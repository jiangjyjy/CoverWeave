from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from .encoding import ActionCodec, TokenCodec, collate_batch, encode_actions, encode_source, legal_action_mask
from .env import CoverageEnv, EnvState
from .graph import CoverageGraph, tree_from_dict
from .io import read_jsonl, sha256_file, write_json
from .model_sft import CodecBundle, MergeTransformer, TransformerConfig, decode_episode

CHECKPOINT_SCHEMA_VERSION = 1
_SPLITS = ("train", "valid", "test")


@dataclass
class TrainingBundle:
    model: MergeTransformer
    codecs: CodecBundle
    optimizer: torch.optim.Optimizer
    model_config: TransformerConfig
    split_hashes: dict[str, str]
    config: dict[str, object]


def seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _model_value(config: Mapping[str, object], key: str, default: object) -> object:
    if key in config:
        return config[key]
    model_config = config.get("model")
    if isinstance(model_config, Mapping) and key in model_config:
        return model_config[key]
    return default


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_float(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _split_path(config: Mapping[str, object], split: str) -> Path:
    for key in (f"{split}_path", f"{split}_jsonl"):
        value = config.get(key)
        if value is not None:
            return Path(value)
    dataset_dir = config.get("dataset_dir") or config.get("data_dir")
    dataset = config.get("dataset")
    if dataset_dir is None and isinstance(dataset, Mapping):
        dataset_dir = dataset.get("dir") or dataset.get("path")
    if dataset_dir is None:
        raise ValueError("config must provide dataset_dir or split paths")
    return Path(dataset_dir) / f"{split}.jsonl"


def _load_splits(config: Mapping[str, object]) -> tuple[dict[str, list[dict[str, object]]], dict[str, str]]:
    paths = {split: _split_path(config, split) for split in _SPLITS}
    manifest_path_value = config.get("manifest_path")
    manifest_path = (
        Path(manifest_path_value)
        if manifest_path_value is not None
        else paths["train"].parent / "manifest.json"
    )
    manifest: Mapping[str, object] = {}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text())
        if not isinstance(loaded, Mapping):
            raise ValueError("dataset manifest must be a mapping")
        manifest = loaded
    manifest_hashes = manifest.get("hashes", {})
    if manifest_hashes is not None and not isinstance(manifest_hashes, Mapping):
        raise ValueError("dataset manifest hashes must be a mapping")
    rows: dict[str, list[dict[str, object]]] = {}
    split_hashes: dict[str, str] = {}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        rows[split] = read_jsonl(path)
        if not rows[split]:
            raise ValueError(f"{split} split is empty")
        actual_hash = sha256_file(path)
        expected_hash = manifest_hashes.get(split) if isinstance(manifest_hashes, Mapping) else None
        if expected_hash is not None and str(expected_hash) != actual_hash:
            raise ValueError(f"manifest hash mismatch for {split}")
        split_hashes[split] = actual_hash
    return rows, split_hashes


def _build_codecs(
    rows: Mapping[str, Iterable[Mapping[str, object]]],
    config: Mapping[str, object],
) -> tuple[CodecBundle, TransformerConfig]:
    all_rows = [row for split in _SPLITS for row in rows[split]]
    max_inventory = max(
        int(row.get("leaf_count", len(row["inventory"])))
        for row in all_rows
    )
    max_n = max(int(row["N"]) for row in all_rows)
    if max_inventory < 1 or max_n < 1:
        raise ValueError("dataset dimensions must be positive")
    token_codec = TokenCodec()
    action_codec = ActionCodec(max_inventory=max_inventory)
    max_source_length = max(
        len(encode_source(row, token_codec)) for row in all_rows
    )
    max_target_length = max(
        len(encode_actions(row, action_codec)) for row in all_rows
    )
    configured_source_length = _positive_int(
        _model_value(config, "max_src_len", max_source_length),
        "max_src_len",
    )
    configured_target_length = _positive_int(
        _model_value(config, "max_target_len", max_target_length),
        "max_target_len",
    )
    model_config = TransformerConfig(
        src_vocab_size=token_codec.vocab_size(max_n - 1),
        action_vocab_size=action_codec.vocab_size,
        d_model=_positive_int(_model_value(config, "d_model", 128), "d_model"),
        nhead=_positive_int(_model_value(config, "nhead", 4), "nhead"),
        num_layers=_positive_int(_model_value(config, "num_layers", 2), "num_layers"),
        dim_feedforward=_positive_int(
            _model_value(config, "dim_feedforward", 256),
            "dim_feedforward",
        ),
        dropout=_finite_float(_model_value(config, "dropout", 0.1), "dropout"),
        max_src_len=max(configured_source_length, max_source_length),
        max_target_len=max(configured_target_length, max_target_length),
    )
    if model_config.d_model % model_config.nhead:
        raise ValueError("d_model must be divisible by nhead")
    return CodecBundle(token_codec, action_codec), model_config


def _optimizer_kwargs(config: Mapping[str, object]) -> dict[str, object]:
    betas = _model_value(config, "betas", (0.9, 0.999))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("betas must contain two values")
    return {
        "lr": _finite_float(_model_value(config, "learning_rate", 1e-3), "learning_rate"),
        "weight_decay": _finite_float(_model_value(config, "weight_decay", 0.01), "weight_decay"),
        "betas": (float(betas[0]), float(betas[1])),
        "eps": _finite_float(_model_value(config, "eps", 1e-8), "eps"),
    }


def _make_bundle(
    config: Mapping[str, object],
    rows: Mapping[str, list[dict[str, object]]],
    split_hashes: dict[str, str],
    device: torch.device,
) -> TrainingBundle:
    codecs, model_config = _build_codecs(rows, config)
    model = MergeTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **_optimizer_kwargs(config))
    return TrainingBundle(
        model=model,
        codecs=codecs,
        optimizer=optimizer,
        model_config=model_config,
        split_hashes=dict(split_hashes),
        config=dict(config),
    )


def _row_state(row: Mapping[str, object]) -> tuple[CoverageGraph, EnvState]:
    graph = CoverageGraph(
        int(row["N"]),
        [tuple(edge) for edge in row.get("graph_edges", [])],
    )
    inventory = tuple(tree_from_dict(item) for item in row["inventory"])
    return graph, EnvState(inventory)


def _teacher_valid_action_rate(
    rows: list[Mapping[str, object]],
    logits: torch.Tensor,
    action_codec: ActionCodec,
) -> tuple[int, int]:
    valid = 0
    total = 0
    for row_index, row in enumerate(rows):
        graph, state = _row_state(row)
        env = CoverageEnv(graph)
        target_tokens = encode_actions(row, action_codec)
        row_logits = logits[row_index].detach().cpu()
        for step, target in enumerate(target_tokens):
            prediction = int(row_logits[step].argmax().item())
            valid += int(bool(legal_action_mask(state, action_codec)[prediction]))
            total += 1
            if target == action_codec.eos_token:
                break
            action = action_codec.decode(target)
            if action is None:
                break
            state = env.apply(state, action)
    return valid, total


def _run_epoch(
    model: MergeTransformer,
    rows: list[dict[str, object]],
    codecs: CodecBundle,
    device: torch.device,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    warmup_steps: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_valid = 0
    total_predictions = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = collate_batch(batch_rows, codecs.token, codecs.action)
        batch = {key: value.to(device) for key, value in batch.items()}
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["src"],
            batch["decoder_input"],
            batch["src_padding_mask"],
            batch["decoder_padding_mask"],
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["targets"].reshape(-1),
            ignore_index=codecs.action.pad_token,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        if optimizer is not None:
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("non-finite gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite gradient norm")
            global_step += 1
            if warmup_steps:
                scale = min(1.0, global_step / warmup_steps)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = float(optimizer.defaults["lr"]) * scale
            optimizer.step()
        target_mask = batch["targets"].ne(codecs.action.pad_token)
        predictions = logits.argmax(dim=-1)
        token_count = int(target_mask.sum().item())
        total_loss += float(loss.detach().cpu()) * token_count
        total_tokens += token_count
        total_correct += int(((predictions == batch["targets"]) & target_mask).sum().item())
        valid, prediction_count = _teacher_valid_action_rate(
            batch_rows,
            logits,
            codecs.action,
        )
        total_valid += valid
        total_predictions += prediction_count
    if total_tokens == 0:
        raise ValueError("batch contains no target tokens")
    return {
        "loss": total_loss / total_tokens,
        "token_accuracy": total_correct / total_tokens,
        "valid_action_rate": total_valid / total_predictions if total_predictions else 0.0,
    }, global_step


def _decode_metrics(
    model: MergeTransformer,
    rows: list[dict[str, object]],
    codecs: CodecBundle,
    device: torch.device,
    action_budget: int,
) -> dict[str, float]:
    exact = 0
    valid_action_rate = 0.0
    for row in rows:
        result = decode_episode(model, row, codecs, device, action_budget)
        exact += int(result.success)
        valid_action_rate += result.valid_action_rate
    count = len(rows)
    return {
        "exact_solve_rate": exact / count,
        "valid_action_rate": valid_action_rate / count,
    }


def _assert_finite_metrics(metrics: Mapping[str, object]) -> None:
    for name, value in metrics.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise FloatingPointError(f"non-finite metric: {name}")


def _optimizer_payload(optimizer: torch.optim.Optimizer) -> dict[str, object]:
    group = optimizer.param_groups[0]
    return {
        "name": optimizer.__class__.__name__,
        "config": {
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "betas": [float(value) for value in group["betas"]],
            "eps": float(group["eps"]),
        },
        "state_dict": optimizer.state_dict(),
    }


def _checkpoint_payload(
    bundle: TrainingBundle,
    epoch: int,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    _assert_finite_metrics(metrics)
    max_atom = (
        bundle.model_config.src_vocab_size
        - TokenCodec.ATOM_OFFSET
        - 1
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_config": asdict(bundle.model_config),
        "codecs": {
            "token": {
                "max_atom": max_atom,
                "vocab_size": bundle.codecs.token.vocab_size(max_atom),
            },
            "action": {
                "max_inventory": bundle.codecs.action.max_inventory,
                "vocab_size": bundle.codecs.action.vocab_size,
            },
        },
        "split_hashes": dict(bundle.split_hashes),
        "optimizer": _optimizer_payload(bundle.optimizer),
        "model_state_dict": bundle.model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "training_config": dict(bundle.config),
        "model_training": bool(bundle.model.training),
    }


def save_checkpoint(
    path: str | Path,
    bundle: TrainingBundle,
    epoch: int,
    metrics: Mapping[str, object],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(bundle, epoch, metrics), checkpoint_path)


def _load_torch_checkpoint(path: Path, device: torch.device) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    return payload


def load_checkpoint(
    path: str | Path,
    device: str | torch.device,
    expected_split_hashes: Mapping[str, str] | None = None,
) -> TrainingBundle:
    checkpoint_path = Path(path)
    payload = _load_torch_checkpoint(checkpoint_path, torch.device(device))
    required = {
        "schema_version",
        "model_config",
        "codecs",
        "split_hashes",
        "optimizer",
        "model_state_dict",
        "epoch",
        "metrics",
    }
    if not required <= set(payload):
        raise ValueError("checkpoint schema is incomplete")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is incompatible")
    split_hashes = payload["split_hashes"]
    if not isinstance(split_hashes, Mapping):
        raise ValueError("checkpoint split hashes are incompatible")
    split_hashes = {str(key): str(value) for key, value in split_hashes.items()}
    if expected_split_hashes is not None and split_hashes != dict(expected_split_hashes):
        raise ValueError("checkpoint dataset split hashes are incompatible")
    model_payload = payload["model_config"]
    if not isinstance(model_payload, Mapping):
        raise ValueError("checkpoint model config is incompatible")
    expected_fields = {field.name for field in fields(TransformerConfig)}
    if set(model_payload) != expected_fields:
        raise ValueError("checkpoint model config is incompatible")
    try:
        model_config = TransformerConfig(**dict(model_payload))
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint model config is incompatible") from error
    codecs_payload = payload["codecs"]
    if not isinstance(codecs_payload, Mapping):
        raise ValueError("checkpoint codecs are incompatible")
    token_payload = codecs_payload.get("token")
    action_payload = codecs_payload.get("action")
    if not isinstance(token_payload, Mapping) or not isinstance(action_payload, Mapping):
        raise ValueError("checkpoint codecs are incompatible")
    max_atom = _positive_int(token_payload.get("max_atom"), "checkpoint max_atom")
    max_inventory = _positive_int(
        action_payload.get("max_inventory"),
        "checkpoint max_inventory",
    )
    token_codec = TokenCodec()
    action_codec = ActionCodec(max_inventory)
    if token_codec.vocab_size(max_atom) != model_config.src_vocab_size:
        raise ValueError("checkpoint token codec is incompatible")
    if action_codec.vocab_size != model_config.action_vocab_size:
        raise ValueError("checkpoint action codec is incompatible")
    optimizer_payload = payload["optimizer"]
    if not isinstance(optimizer_payload, Mapping) or optimizer_payload.get("name") != "AdamW":
        raise ValueError("checkpoint optimizer is incompatible")
    optimizer_config = optimizer_payload.get("config")
    if not isinstance(optimizer_config, Mapping):
        raise ValueError("checkpoint optimizer is incompatible")
    try:
        optimizer = torch.optim.AdamW(
            MergeTransformer(model_config).parameters(),
            lr=float(optimizer_config["lr"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=tuple(float(value) for value in optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint optimizer is incompatible") from error
    model = MergeTransformer(model_config).to(torch.device(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
    )
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(optimizer_payload["state_dict"])
    except (RuntimeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint state is incompatible") from error
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(torch.device(device))
    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError("checkpoint metrics are incompatible")
    _assert_finite_metrics(metrics)
    epoch = payload["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("checkpoint epoch is incompatible")
    training_config = payload.get("training_config", {})
    if not isinstance(training_config, Mapping):
        raise ValueError("checkpoint training config is incompatible")
    model.train(bool(payload.get("model_training", True)))
    return TrainingBundle(
        model=model,
        codecs=CodecBundle(token_codec, action_codec),
        optimizer=optimizer,
        model_config=model_config,
        split_hashes=split_hashes,
        config=dict(training_config),
    )


def train(
    config: Mapping[str, object],
    run_dir: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    epochs = _positive_int(config.get("epochs", 1), "epochs")
    batch_size = _positive_int(config.get("batch_size", 32), "batch_size")
    grad_clip = _finite_float(config.get("grad_clip", 1.0), "grad_clip")
    if grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    seed = _nonnegative_int(config.get("seed", 0), "seed")
    seed_everything(seed)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    rows, split_hashes = _load_splits(config)
    bundle = _make_bundle(config, rows, split_hashes, target_device)
    warmup_steps = _nonnegative_int(config.get("warmup_steps", 0), "warmup_steps")
    global_step = 0
    action_budget = _positive_int(
        config.get("action_budget", bundle.model_config.max_target_len),
        "action_budget",
    )
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_path / "train.log.jsonl"
    metrics_path.write_text("")
    best_exact = -1.0
    best_loss = float("inf")
    best_epoch = 0
    last_metrics: dict[str, object] = {}
    for epoch in range(1, epochs + 1):
        train_loss, global_step = _run_epoch(
            bundle.model,
            rows["train"],
            bundle.codecs,
            target_device,
            batch_size,
            bundle.optimizer,
            grad_clip,
            warmup_steps,
            global_step,
        )
        train_decode = _decode_metrics(
            bundle.model,
            rows["train"],
            bundle.codecs,
            target_device,
            action_budget,
        )
        valid_loss, _ = _run_epoch(
            bundle.model,
            rows["valid"],
            bundle.codecs,
            target_device,
            batch_size,
            None,
            grad_clip,
            0,
            global_step,
        )
        valid_decode = _decode_metrics(
            bundle.model,
            rows["valid"],
            bundle.codecs,
            target_device,
            action_budget,
        )
        epoch_metrics = {
            "epoch": epoch,
            "loss": train_loss["loss"],
            "token_accuracy": train_loss["token_accuracy"],
            "valid_action_rate": train_decode["valid_action_rate"],
            "exact_solve_rate": train_decode["exact_solve_rate"],
            "train_loss": train_loss["loss"],
            "train_token_accuracy": train_loss["token_accuracy"],
            "train_valid_action_rate": train_decode["valid_action_rate"],
            "train_exact_solve_rate": train_decode["exact_solve_rate"],
            "valid_loss": valid_loss["loss"],
            "valid_token_accuracy": valid_loss["token_accuracy"],
            "valid_valid_action_rate": valid_decode["valid_action_rate"],
            "valid_exact_solve_rate": valid_decode["exact_solve_rate"],
        }
        _assert_finite_metrics(epoch_metrics)
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
        last_metrics = epoch_metrics
        if (
            valid_decode["exact_solve_rate"] > best_exact
            or (
                valid_decode["exact_solve_rate"] == best_exact
                and valid_loss["loss"] < best_loss
            )
        ):
            best_exact = valid_decode["exact_solve_rate"]
            best_loss = valid_loss["loss"]
            best_epoch = epoch
            save_checkpoint(checkpoint_dir / "best.pt", bundle, epoch, epoch_metrics)
        save_checkpoint(checkpoint_dir / "last.pt", bundle, epoch, epoch_metrics)
    result = dict(last_metrics)
    result.update(
        {
            "best_epoch": best_epoch,
            "best_valid_exact_solve_rate": best_exact,
            "max_inventory": bundle.codecs.action.max_inventory,
            "max_source_vocab": bundle.model_config.src_vocab_size,
            "metrics_path": str(metrics_path),
            "best_checkpoint": str(checkpoint_dir / "best.pt"),
            "last_checkpoint": str(checkpoint_dir / "last.pt"),
            "split_hashes": dict(split_hashes),
        }
    )
    write_json(run_path / "config.json", dict(config))
    return result
