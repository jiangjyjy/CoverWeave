"""Constrained Transformer decoding for CompForge SFT."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from coverage_repro.compforge import CompForgeEnv, CompForgeState, ObjectInstance, Rule, RuleAction, RuleKind, check_compforge_trace
from coverage_repro.compforge_encoding import (
    CompForgeActionCodec,
    CompForgeTokenCodec,
    compforge_legal_action_mask,
    encode_compforge_source,
    collate_compforge_batch,
)
from coverage_repro.io import read_jsonl, sha256_file, write_json
from coverage_repro.model_sft import MergeTransformer, TransformerConfig
from coverage_repro.train_sft import seed_everything


@dataclass(frozen=True)
class CompForgeCodecBundle:
    token: CompForgeTokenCodec
    action: CompForgeActionCodec


@dataclass(frozen=True)
class CompForgeDecodeResult:
    success: bool
    actions: tuple[RuleAction, ...]
    legal_actions_at_step: tuple[tuple[RuleAction, ...], ...]
    valid_action_rate: float
    checker_valid: bool
    failure_reason: str
    emitted_tokens: tuple[int, ...]


@dataclass
class CompForgeTrainingBundle:
    model: MergeTransformer
    codecs: CompForgeCodecBundle
    optimizer: torch.optim.Optimizer
    model_config: TransformerConfig
    split_hashes: dict[str, str]
    config: dict[str, object]


def _unpack_codecs(
    codecs: CompForgeCodecBundle | tuple[CompForgeTokenCodec, CompForgeActionCodec],
) -> CompForgeCodecBundle:
    if isinstance(codecs, CompForgeCodecBundle):
        return codecs
    if isinstance(codecs, tuple) and len(codecs) == 2:
        token, action = codecs
        if isinstance(token, CompForgeTokenCodec) and isinstance(action, CompForgeActionCodec):
            return CompForgeCodecBundle(token, action)
    raise TypeError("codecs must contain CompForgeTokenCodec and CompForgeActionCodec")


def _result(
    success: bool,
    actions: list[RuleAction],
    legal_actions: list[tuple[RuleAction, ...]],
    checker_valid: bool,
    failure_reason: str,
    emitted_tokens: list[int],
) -> CompForgeDecodeResult:
    valid = sum(action in legal_actions[index] for index, action in enumerate(actions))
    return CompForgeDecodeResult(
        success=success,
        actions=tuple(actions),
        legal_actions_at_step=tuple(legal_actions),
        valid_action_rate=valid / len(actions) if actions else 1.0,
        checker_valid=checker_valid,
        failure_reason=failure_reason,
        emitted_tokens=tuple(emitted_tokens),
    )


def decode_compforge_episode(
    model: nn.Module,
    row: Mapping[str, object],
    env: CompForgeEnv,
    inventory: Sequence[ObjectInstance],
    codecs: CompForgeCodecBundle | tuple[CompForgeTokenCodec, CompForgeActionCodec],
    device: str | torch.device,
    action_budget: int,
) -> CompForgeDecodeResult:
    """Greedily decode public observations while masking state-illegal actions."""
    if isinstance(action_budget, bool) or not isinstance(action_budget, int) or action_budget < 1:
        raise ValueError("action_budget must be a positive integer")
    bundle = _unpack_codecs(codecs)
    goal = row.get("goal")
    if not isinstance(goal, str) or not goal:
        raise ValueError("row goal must be a non-empty string")
    target_device = torch.device(device)
    state: CompForgeState = env.initial_state(tuple(inventory))
    source = torch.tensor(encode_compforge_source(row, bundle.token), dtype=torch.long, device=target_device).unsqueeze(0)
    source_mask = source.eq(bundle.token.PAD)
    decoder_tokens = [bundle.action.bos_token]
    actions: list[RuleAction] = []
    legal_actions: list[tuple[RuleAction, ...]] = []
    emitted_tokens: list[int] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(action_budget):
                decoder_input = torch.tensor(decoder_tokens, dtype=torch.long, device=target_device).unsqueeze(0)
                logits = model(source, decoder_input, source_mask, decoder_input.eq(bundle.action.pad_token))
                if logits.ndim != 3 or logits.shape != (1, len(decoder_tokens), bundle.action.vocab_size):
                    return _result(False, actions, legal_actions, False, "malformed_action", emitted_tokens)
                current_legal = tuple(env.valid_actions(state))
                legal_actions.append(current_legal)
                mask = compforge_legal_action_mask(env, state, bundle.action, goal_state=goal).to(target_device)
                masked_logits = logits[0, -1].masked_fill(~mask, float("-inf"))
                if not torch.isfinite(masked_logits).any():
                    return _result(False, actions, legal_actions, False, "malformed_action", emitted_tokens)
                token = int(masked_logits.argmax().item())
                emitted_tokens.append(token)
                decoder_tokens.append(token)
                if token == bundle.action.eos_token:
                    checked = check_compforge_trace(env, tuple(inventory), goal, tuple(actions))
                    if checked.valid:
                        return _result(True, actions, legal_actions, True, "none", emitted_tokens)
                    return _result(False, actions, legal_actions, False, "wrong_final_state", emitted_tokens)
                try:
                    action = bundle.action.decode(token)
                except ValueError:
                    return _result(False, actions, legal_actions, False, "malformed_action", emitted_tokens)
                if action is None or action not in current_legal:
                    return _result(False, actions, legal_actions, False, "malformed_action", emitted_tokens)
                state = env.apply(state, action)
                actions.append(action)
    finally:
        model.train(was_training)
    return _result(False, actions, legal_actions, False, "budget_exhausted", emitted_tokens)


def _env_from_row(row: Mapping[str, object]) -> CompForgeEnv:
    instance = row.get("instance")
    if not isinstance(instance, Mapping) or not isinstance(instance.get("rules"), Sequence):
        raise ValueError("CompForge row is missing rule audit data")
    rules: list[Rule] = []
    for payload in instance["rules"]:
        if not isinstance(payload, Mapping):
            raise ValueError("rule audit entry is invalid")
        rules.append(
            Rule(
                str(payload["rule_id"]),
                RuleKind(str(payload["kind"])),
                tuple(str(value) for value in payload["inputs"]),
                tuple(str(value) for value in payload["outputs"]),
                int(payload["cost"]),
            )
        )
    return CompForgeEnv(tuple(rules))


def _inventory_from_row(row: Mapping[str, object]) -> tuple[ObjectInstance, ...]:
    values = row.get("inventory")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("row inventory is invalid")
    return tuple(ObjectInstance(index, str(state), (index,)) for index, state in enumerate(values))


def _load_splits(config: Mapping[str, object]) -> tuple[dict[str, list[dict[str, object]]], dict[str, str]]:
    dataset_dir = config.get("dataset_dir")
    if not isinstance(dataset_dir, str) or not dataset_dir:
        raise ValueError("config must contain dataset_dir")
    directory = Path(dataset_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    hashes = manifest.get("hashes") if isinstance(manifest, Mapping) else None
    if not isinstance(hashes, Mapping):
        raise ValueError("dataset manifest hashes are invalid")
    rows: dict[str, list[dict[str, object]]] = {}
    verified: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        path = directory / f"{split}.jsonl"
        rows[split] = read_jsonl(path)
        if not rows[split]:
            raise ValueError(f"{split} split is empty")
        actual = sha256_file(path)
        if hashes.get(split) != actual:
            raise ValueError(f"manifest hash mismatch for {split}")
        verified[split] = actual
    return rows, verified


def _make_bundle(config: Mapping[str, object], rows: Mapping[str, list[dict[str, object]]], hashes: dict[str, str], device: torch.device) -> CompForgeTrainingBundle:
    all_rows = [row for split in ("train", "valid", "test") for row in rows[split]]
    env = _env_from_row(all_rows[0])
    schema = tuple((rule.rule_id, len(rule.inputs), len(rule.outputs)) for rule in env.rules)
    if any(tuple((rule.rule_id, len(rule.inputs), len(rule.outputs)) for rule in _env_from_row(row).rules) != schema for row in all_rows):
        raise ValueError("all CompForge rows must share one rule schema")
    token = CompForgeTokenCodec(tuple(sorted({state for row in all_rows for state in ([*row["inventory"], row["goal"]])} | {state for rule in env.rules for state in (*rule.inputs, *rule.outputs)})))
    action = CompForgeActionCodec.from_env(env, max(len(row["inventory"]) for row in all_rows))
    max_src = max(len(encode_compforge_source(row, token)) for row in all_rows)
    max_tgt = max(len(row["actions"]) + 1 for row in all_rows)
    d_model = int(config.get("d_model", 32))
    nhead = int(config.get("nhead", 4))
    if d_model < 1 or nhead < 1 or d_model % nhead:
        raise ValueError("d_model must be positive and divisible by nhead")
    model_config = TransformerConfig(
        src_vocab_size=token.vocab_size, action_vocab_size=action.vocab_size,
        d_model=d_model, nhead=nhead, num_layers=int(config.get("num_layers", 1)),
        dim_feedforward=int(config.get("dim_feedforward", 64)), dropout=float(config.get("dropout", 0.0)),
        max_src_len=max_src, max_target_len=max(max_tgt, int(config.get("action_budget", max_tgt)) + 1),
    )
    model = MergeTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    return CompForgeTrainingBundle(model, CompForgeCodecBundle(token, action), optimizer, model_config, dict(hashes), dict(config))


def _run_epoch(bundle: CompForgeTrainingBundle, rows: list[dict[str, object]], device: torch.device, batch_size: int, train_mode: bool) -> dict[str, float]:
    bundle.model.train(train_mode)
    losses: list[float] = []
    correct = total = 0
    for start in range(0, len(rows), batch_size):
        batch = {key: value.to(device) for key, value in collate_compforge_batch(rows[start:start + batch_size], bundle.codecs.token, bundle.codecs.action).items()}
        if train_mode:
            bundle.optimizer.zero_grad(set_to_none=True)
        logits = bundle.model(batch["src"], batch["decoder_input"], batch["src_padding_mask"], batch["decoder_padding_mask"])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["targets"].reshape(-1), ignore_index=bundle.codecs.action.pad_token)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), 1.0)
            bundle.optimizer.step()
        mask = batch["targets"].ne(bundle.codecs.action.pad_token)
        losses.append(float(loss.detach().cpu()))
        correct += int(((logits.argmax(dim=-1) == batch["targets"]) & mask).sum().item())
        total += int(mask.sum().item())
    return {"loss": sum(losses) / len(losses), "token_accuracy": correct / total}


def evaluate_compforge_splits(bundle: CompForgeTrainingBundle, rows: Mapping[str, list[dict[str, object]]], device: str | torch.device, action_budget: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split, split_rows in rows.items():
        solved = 0
        for row in split_rows:
            result = decode_compforge_episode(bundle.model, row, _env_from_row(row), _inventory_from_row(row), bundle.codecs, device, action_budget)
            solved += int(result.success)
        metrics[split] = solved / len(split_rows)
    return metrics


def _checkpoint_payload(bundle: CompForgeTrainingBundle, epoch: int, metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1, "model_config": asdict(bundle.model_config),
        "codecs": {"states": list(bundle.codecs.token.states), "rules": [
            {"rule_id": rule.rule_id, "kind": rule.kind.value, "inputs": list(rule.inputs), "outputs": list(rule.outputs), "cost": rule.cost}
            for rule in bundle.codecs.action.rules], "max_inventory": bundle.codecs.action.max_inventory},
        "split_hashes": bundle.split_hashes, "optimizer": bundle.optimizer.state_dict(),
        "model_state_dict": bundle.model.state_dict(), "epoch": epoch, "metrics": dict(metrics), "training_config": bundle.config,
    }


def save_compforge_checkpoint(path: str | Path, bundle: CompForgeTrainingBundle, epoch: int, metrics: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(bundle, epoch, metrics), destination)


def load_compforge_checkpoint(path: str | Path, device: str | torch.device, expected_split_hashes: Mapping[str, str] | None = None) -> CompForgeTrainingBundle:
    payload = torch.load(Path(path), map_location=torch.device(device), weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("checkpoint schema is incompatible")
    hashes = payload.get("split_hashes")
    if not isinstance(hashes, Mapping) or (expected_split_hashes is not None and dict(hashes) != dict(expected_split_hashes)):
        raise ValueError("checkpoint dataset split hashes are incompatible")
    codec_payload = payload.get("codecs")
    model_payload = payload.get("model_config")
    if not isinstance(codec_payload, Mapping) or not isinstance(model_payload, Mapping):
        raise ValueError("checkpoint codecs are incompatible")
    expected_fields = {field.name for field in fields(TransformerConfig)}
    if set(model_payload) != expected_fields:
        raise ValueError("checkpoint model config is incompatible")
    rules = tuple(Rule(str(item["rule_id"]), RuleKind(str(item["kind"])), tuple(item["inputs"]), tuple(item["outputs"]), int(item["cost"])) for item in codec_payload["rules"])
    token = CompForgeTokenCodec(tuple(codec_payload["states"]))
    action = CompForgeActionCodec(rules, int(codec_payload["max_inventory"]))
    model_config = TransformerConfig(**dict(model_payload))
    if token.vocab_size != model_config.src_vocab_size or action.vocab_size != model_config.action_vocab_size:
        raise ValueError("checkpoint codecs are incompatible")
    model = MergeTransformer(model_config).to(torch.device(device))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(payload["optimizer"])
    return CompForgeTrainingBundle(model, CompForgeCodecBundle(token, action), optimizer, model_config, dict(hashes), dict(payload.get("training_config", {})))


def train_compforge_sft(config: Mapping[str, object], run_dir: str | Path, device: str | torch.device) -> dict[str, object]:
    seed_everything(int(config.get("seed", 0)))
    epochs, batch_size = int(config.get("epochs", 1)), int(config.get("batch_size", 8))
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    target_device = torch.device(device)
    rows, hashes = _load_splits(config)
    bundle = _make_bundle(config, rows, hashes, target_device)
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    budget = int(config.get("action_budget", bundle.model_config.max_target_len))
    best_exact, best_path, last = -1.0, output / "checkpoints" / "best.pt", {}
    log_path = output / "train.log.jsonl"
    with log_path.open("w") as log:
        for epoch in range(1, epochs + 1):
            train_metrics = _run_epoch(bundle, rows["train"], target_device, batch_size, True)
            valid_metrics = _run_epoch(bundle, rows["valid"], target_device, batch_size, False)
            exact = evaluate_compforge_splits(bundle, {"valid": rows["valid"]}, target_device, budget)["valid"]
            last = {"epoch": epoch, "train_loss": train_metrics["loss"], "valid_loss": valid_metrics["loss"], "valid_exact_solve_rate": exact}
            log.write(json.dumps(last, sort_keys=True) + "\n")
            if exact >= best_exact:
                best_exact = exact
                save_compforge_checkpoint(best_path, bundle, epoch, last)
    loaded = load_compforge_checkpoint(best_path, target_device, hashes)
    split_metrics = evaluate_compforge_splits(loaded, rows, target_device, budget)
    result = {**last, "best_checkpoint": str(best_path), "split_hashes": hashes, "split_metrics": split_metrics, "valid_exact_solve_rate": split_metrics["valid"]}
    write_json(output / "result.json", result)
    return result
