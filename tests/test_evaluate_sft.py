from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import yaml

import coverage_repro.evaluate_sft as evaluate_sft
from coverage_repro.cli import main as cli_main
from coverage_repro.dataset import build_splits
from coverage_repro.encoding import ActionCodec, TokenCodec
from coverage_repro.evaluate_sft import evaluate_checkpoint
from coverage_repro.io import read_jsonl, sha256_file, write_json, write_jsonl
from coverage_repro.model_sft import CodecBundle, MergeTransformer, TransformerConfig
from coverage_repro.train_sft import TrainingBundle, save_checkpoint


def _dataset_config() -> dict[str, object]:
    return {
        "N": 4,
        "lambda_values": [0.5, 1.0],
        "graph_seeds": 2,
        "train_depths": [2],
        "test_depths": [2, 3],
        "train_samples": 2,
        "valid_samples": 1,
        "test_samples": 1,
        "seed_base": 31,
    }


def _save_checkpoint(
    path: Path,
    manifest: dict[str, object],
    *,
    seed: int = 17,
    schema_version: int | None = None,
) -> None:
    torch.manual_seed(5)
    token_codec = TokenCodec()
    action_codec = ActionCodec(max_inventory=6)
    model_config = TransformerConfig(
        src_vocab_size=token_codec.vocab_size(3),
        action_vocab_size=action_codec.vocab_size,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_src_len=128,
        max_target_len=16,
    )
    model = MergeTransformer(model_config)
    bundle = TrainingBundle(
        model=model,
        codecs=CodecBundle(token_codec, action_codec),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        model_config=model_config,
        split_hashes=dict(manifest["hashes"]),
        config={"seed": seed, "action_budget": 16},
    )
    save_checkpoint(path, bundle, epoch=1, metrics={"exact_solve_rate": 0.0})
    if schema_version is not None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["schema_version"] = schema_version
        torch.save(payload, path)


def _build_evaluation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset_dir = tmp_path / "dataset"
    manifest = build_splits(_dataset_config(), dataset_dir)
    checkpoint = tmp_path / "best.pt"
    _save_checkpoint(checkpoint, manifest)
    return dataset_dir, checkpoint


def test_evaluator_writes_every_cell_and_exact_metrics(tmp_path):
    dataset_dir, checkpoint = _build_evaluation_fixture(tmp_path)
    output_dir = tmp_path / "evaluation"

    summary = evaluate_checkpoint(checkpoint, dataset_dir, output_dir, "cpu")
    rows = read_jsonl(output_dir / "model_results.jsonl")

    assert len(rows) == 2 * 2 * 2 * 1
    assert summary["result_rows"] == len(rows)
    assert summary["finite"] is True
    assert (output_dir / "analysis.json").exists()
    cells = {
        (row["lambda"], row["depth"], row["graph_seed"], row["model_seed"])
        for row in rows
    }
    assert len(cells) == len(rows)
    for row in rows:
        assert row["episodes"] == 1
        assert row["success"] in (0, 1)
        assert 0.0 <= row["exact_solve_rate"] <= 1.0
        assert 0.0 <= row["valid_action_rate"] <= 1.0
        assert row["bfs_oracle_success"] == 1
        assert row["bfs_oracle_success_rate"] == 1.0
        assert math.isfinite(row["mean_action_count"])
        assert math.isfinite(row["bfs_cost_ratio"])
        assert sum(row["failure_reason_counts"].values()) == row["episodes"]
        assert "oracle_success" not in row


def test_evaluator_rejects_manifest_hash_mismatch(tmp_path):
    dataset_dir, checkpoint = _build_evaluation_fixture(tmp_path)
    with (dataset_dir / "test.jsonl").open("a") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        evaluate_checkpoint(checkpoint, dataset_dir, tmp_path / "output", "cpu")


def test_evaluator_rejects_checkpoint_compatibility_mismatch(tmp_path):
    dataset_dir, _ = _build_evaluation_fixture(tmp_path)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    manifest["hashes"]["test"] = "0" * 64
    checkpoint = tmp_path / "incompatible.pt"
    _save_checkpoint(checkpoint, manifest)

    with pytest.raises(ValueError, match="split hashes"):
        evaluate_checkpoint(checkpoint, dataset_dir, tmp_path / "output", "cpu")


@pytest.mark.parametrize("mutation", ["remove", "tamper"])
def test_evaluator_predictions_do_not_read_target_actions(tmp_path, mutation):
    dataset_dir, checkpoint = _build_evaluation_fixture(tmp_path)
    first_output = tmp_path / "first"
    evaluate_checkpoint(checkpoint, dataset_dir, first_output, "cpu")
    expected = read_jsonl(first_output / "model_results.jsonl")

    rows = read_jsonl(dataset_dir / "test.jsonl")
    for row in rows:
        if mutation == "remove":
            row.pop("actions")
        else:
            row["actions"] = [[999, 999]]
    write_jsonl(dataset_dir / "test.jsonl", rows)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    manifest["hashes"]["test"] = sha256_file(dataset_dir / "test.jsonl")
    write_json(dataset_dir / "manifest.json", manifest)
    matching_checkpoint = tmp_path / f"{mutation}.pt"
    _save_checkpoint(matching_checkpoint, manifest)

    second_output = tmp_path / "second"
    evaluate_checkpoint(matching_checkpoint, dataset_dir, second_output, "cpu")
    assert read_jsonl(second_output / "model_results.jsonl") == expected


def test_evaluator_replays_model_and_bfs_traces_with_checker(tmp_path, monkeypatch):
    dataset_dir, checkpoint = _build_evaluation_fixture(tmp_path)
    episode_count = len(read_jsonl(dataset_dir / "test.jsonl"))
    calls = []
    original = evaluate_sft.check_trace

    def recording_check(graph, goal, actions, inventory=None):
        calls.append((tuple(actions), tuple(inventory or ())))
        return original(graph, goal, actions, inventory=inventory)

    monkeypatch.setattr(evaluate_sft, "check_trace", recording_check)
    evaluate_checkpoint(
        checkpoint,
        dataset_dir,
        tmp_path / "evaluation",
        "cpu",
    )

    assert len(calls) == 2 * episode_count
    assert all(inventory for _, inventory in calls)


def test_cli_build_train_evaluate_minimal_flow_and_dataset_override(tmp_path):
    config = {
        "N": 4,
        "lambda_values": [0.5],
        "graph_seeds": 1,
        "train_depths": [1],
        "test_depths": [1],
        "train_samples": 1,
        "valid_samples": 1,
        "test_samples": 1,
        "seed_base": 9,
        "dataset_dir": str(tmp_path / "wrong-dataset"),
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "seed": 3,
        "d_model": 8,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "action_budget": 4,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"

    assert cli_main(["build-dataset", "--config", str(config_path), "--output", str(dataset_dir)]) == 0
    assert cli_main([
        "train-sft",
        "--config",
        str(config_path),
        "--dataset",
        str(dataset_dir),
        "--run-dir",
        str(run_dir),
        "--device",
        "cpu",
    ]) == 0
    assert cli_main([
        "evaluate-sft",
        "--checkpoint",
        str(run_dir / "checkpoints" / "best.pt"),
        "--dataset",
        str(dataset_dir),
        "--output",
        str(output_dir),
        "--device",
        "cpu",
    ]) == 0
    assert (output_dir / "model_results.jsonl").exists()
    assert not (tmp_path / "wrong-dataset").exists()

    assert cli_main([
        "evaluate-sft",
        "--checkpoint",
        str(tmp_path / "missing.pt"),
        "--dataset",
        str(dataset_dir),
        "--output",
        str(tmp_path / "bad-output"),
        "--device",
        "cpu",
    ]) != 0
