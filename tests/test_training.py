from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from coverage_repro.dataset import build_splits
from coverage_repro.encoding import ActionCodec, TokenCodec
from coverage_repro.io import read_jsonl
from coverage_repro.model_sft import CodecBundle, MergeTransformer, TransformerConfig
from coverage_repro.train_sft import TrainingBundle, load_checkpoint, save_checkpoint, train


def tiny_initialized_bundle(seed: int = 7) -> TrainingBundle:
    torch.manual_seed(seed)
    model_config = TransformerConfig(
        src_vocab_size=32,
        action_vocab_size=9,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_src_len=32,
        max_target_len=16,
    )
    model = MergeTransformer(model_config)
    codecs = CodecBundle(TokenCodec(), ActionCodec(max_inventory=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return TrainingBundle(
        model=model,
        codecs=codecs,
        optimizer=optimizer,
        model_config=model_config,
        split_hashes={"train": "train-hash", "valid": "valid-hash", "test": "test-hash"},
        config={"seed": seed},
    )


def dataset_config(dataset_dir: Path, *, train_samples: int = 8, epochs: int = 1) -> dict[str, object]:
    data = {
        "N": 8,
        "lambda_values": [1.5],
        "graph_seeds": 1,
        "train_depths": [3],
        "test_depths": [8],
        "train_samples": train_samples,
        "valid_samples": 4,
        "test_samples": 2,
        "seed_base": 20260806,
    }
    build_splits(data, dataset_dir)
    return {
        "dataset_dir": str(dataset_dir),
        "epochs": epochs,
        "batch_size": min(16, train_samples),
        "learning_rate": 3e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "seed": 7,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "action_budget": 24,
    }


def test_checkpoint_reload_preserves_logits(tmp_path):
    bundle = tiny_initialized_bundle()
    bundle.model.eval()
    src = torch.tensor([[1, 3, 8, 4, 5, 8, 7, 9, 6, 2]], dtype=torch.long)
    decoder_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    src_mask = src.eq(0)
    decoder_mask = decoder_input.eq(0)
    before = bundle.model(src, decoder_input, src_mask, decoder_mask).detach().clone()
    path = tmp_path / "model.pt"
    save_checkpoint(path, bundle, epoch=0, metrics={"exact_solve_rate": 0.0})
    loaded = load_checkpoint(path, "cpu")
    loaded.model.eval()
    after = loaded.model(src, decoder_input, src_mask, decoder_mask).detach()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert {"schema_version", "model_config", "codecs", "split_hashes", "optimizer", "epoch", "metrics"} <= payload.keys()


def test_load_checkpoint_rejects_incompatible_schema(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, tiny_initialized_bundle(), epoch=0, metrics={})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema_version"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(path, "cpu")


def test_training_uses_max_inventory_from_all_splits(tmp_path):
    config = dataset_config(tmp_path / "dataset")
    result = train(config, tmp_path / "run", "cpu")
    loaded = load_checkpoint(tmp_path / "run" / "checkpoints" / "best.pt", "cpu")
    assert result["max_inventory"] == 16
    assert loaded.codecs.action.max_inventory == 16
    assert loaded.model.config.src_vocab_size >= TokenCodec().vocab_size(7)


def test_training_is_deterministic_for_same_seed(tmp_path):
    first = train(dataset_config(tmp_path / "dataset", epochs=2), tmp_path / "run-a", "cpu")
    second = train(dataset_config(tmp_path / "dataset", epochs=2), tmp_path / "run-b", "cpu")
    assert first["train_loss"] == second["train_loss"]
    assert first["valid_loss"] == second["valid_loss"]
    assert first["train_exact_solve_rate"] == second["train_exact_solve_rate"]
    log_a = (tmp_path / "run-a" / "train.log.jsonl").read_bytes()
    log_b = (tmp_path / "run-b" / "train.log.jsonl").read_bytes()
    assert log_a == log_b
    assert len(log_a.splitlines()) == 2
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_training_applies_configured_linear_warmup(tmp_path):
    config = dataset_config(tmp_path / "dataset")
    config["batch_size"] = 8
    config["warmup_steps"] = 4
    result = train(config, tmp_path / "run", "cpu")
    loaded = load_checkpoint(result["last_checkpoint"], "cpu")
    assert loaded.optimizer.param_groups[0]["lr"] == pytest.approx(
        config["learning_rate"] / config["warmup_steps"]
    )


def test_training_fails_on_nonfinite_loss(tmp_path, monkeypatch):
    config = dataset_config(tmp_path / "dataset")

    def nonfinite_loss(*args, **kwargs):
        return torch.full((), float("nan"), requires_grad=True)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", nonfinite_loss)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train(config, tmp_path / "run", "cpu")


def test_overfit_64_samples_reaches_exact_solve_gate(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("the bounded overfit gate requires CUDA")
    config = dataset_config(tmp_path / "dataset", train_samples=64, epochs=300)
    config.update(
        {
            "batch_size": 64,
            "d_model": 64,
            "dim_feedforward": 128,
            "num_layers": 2,
            "dropout": 0.0,
        }
    )
    sequences = {
        tuple(tuple(action) for action in row["actions"])
        for row in read_jsonl(tmp_path / "dataset" / "train.jsonl")
    }
    assert len(sequences) > 1
    metrics = train(config, tmp_path / "run", "cuda")
    assert metrics["train_exact_solve_rate"] >= 0.95
