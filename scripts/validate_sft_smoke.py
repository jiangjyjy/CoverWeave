#!/usr/bin/env python3
"""Independently validate one durable SFT smoke artifact directory."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coverage_repro.baselines import bfs_oracle
from coverage_repro.checker import check_trace
from coverage_repro.env import MergeAction
from coverage_repro.graph import CoverageGraph, tree_from_dict
from coverage_repro.io import read_jsonl, sha256_file, write_json
from coverage_repro.train_sft import load_checkpoint

_SPLITS = ("train", "valid", "test")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"non-finite value at {path}")


def _expected_split_sizes(config: Mapping[str, object]) -> dict[str, int]:
    cells = len(config["lambda_values"]) * int(config["graph_seeds"])
    return {
        "train": cells * len(config["train_depths"]) * int(config["train_samples"]),
        "valid": cells * len(config["train_depths"]) * int(config["valid_samples"]),
        "test": cells * len(config["test_depths"]) * int(config["test_samples"]),
    }


def _validate_dataset(run_dir: Path, config: Mapping[str, object]) -> dict[str, object]:
    dataset_dir = run_dir / "dataset"
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    _require(manifest["config"] == config, "dataset manifest config differs from resolved config")
    expected_sizes = _expected_split_sizes(config)
    _require(manifest["split_sizes"] == expected_sizes, "manifest split sizes differ from config")

    ids: dict[str, set[str]] = {}
    replayed = 0
    for split in _SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        _require(len(rows) == expected_sizes[split], f"{split} row count is wrong")
        _require(manifest["hashes"][split] == sha256_file(path), f"{split} hash mismatch")
        ids[split] = {str(row["sample_id"]) for row in rows}
        _require(len(ids[split]) == len(rows), f"{split} has duplicate sample IDs")
        for row in rows:
            _require(row.get("checker_valid") is True, f"dataset checker flag failed: {row['sample_id']}")
            graph = CoverageGraph(int(row["N"]), [tuple(edge) for edge in row["graph_edges"]])
            goal = tree_from_dict(row["goal"])
            actions = tuple(MergeAction(*action) for action in row["actions"])
            _require(bfs_oracle(graph, goal).success, f"BFS failed: {row['sample_id']}")
            _require(check_trace(graph, goal, actions).valid, f"checker failed: {row['sample_id']}")
            replayed += 1
    _require(ids["train"].isdisjoint(ids["valid"]), "train/valid IDs overlap")
    _require(ids["train"].isdisjoint(ids["test"]), "train/test IDs overlap")
    _require(ids["valid"].isdisjoint(ids["test"]), "valid/test IDs overlap")
    hashes_path = run_dir / "data_hashes.json"
    _require(json.loads(hashes_path.read_text()) == manifest["hashes"], "data_hashes.json disagrees")
    return {"split_sizes": expected_sizes, "hashes": manifest["hashes"], "replayed_rows": replayed}


def _validate_model_outputs(run_dir: Path, config: Mapping[str, object]) -> dict[str, object]:
    checkpoints = run_dir / "training" / "checkpoints"
    best = checkpoints / "best.pt"
    last = checkpoints / "last.pt"
    _require(best.exists(), "best checkpoint missing")
    _require(last.exists(), "last checkpoint missing")
    load_checkpoint(best, "cpu")
    load_checkpoint(last, "cpu")

    rows = read_jsonl(run_dir / "evaluation" / "model_results.jsonl")
    expected_count = (
        len(config["lambda_values"])
        * len(config["test_depths"])
        * int(config["graph_seeds"])
        * int(config["model_seeds"])
    )
    _require(len(rows) == expected_count, "model result cell count is wrong")
    expected_cells = {
        (
            float(lam),
            int(depth),
            int(config["seed_base"]) + lam_index * int(config["graph_seeds"]) + graph_index,
            int(config["seed"]),
        )
        for lam_index, lam in enumerate(config["lambda_values"])
        for depth in config["test_depths"]
        for graph_index in range(int(config["graph_seeds"]))
    }
    observed_cells = {
        (float(row["lambda"]), int(row["depth"]), int(row["graph_seed"]), int(row["model_seed"]))
        for row in rows
    }
    _require(observed_cells == expected_cells, "model result cells are incomplete")
    _finite(rows)

    analysis = json.loads((run_dir / "evaluation" / "analysis.json").read_text())
    _require(analysis.get("finite") is True, "analysis finite flag is false")
    _require({"p1", "p2", "p3"} <= set(analysis), "analysis lacks P1/P2/P3")
    _finite(analysis)
    return {"result_cells": expected_count, "analysis": analysis}


def validate(run_dir: str | Path) -> dict[str, object]:
    root = Path(run_dir)
    config = yaml.safe_load((root / "resolved-config.yaml").read_text())
    _require(isinstance(config, dict), "resolved config must be a mapping")
    for path in (
        root / "bash.log",
        root / "commands.sh",
        root / "environment.txt",
        root / "nvidia-smi.txt",
        root / "git-commit.txt",
        root / "git-status.txt",
    ):
        _require(path.exists(), f"missing run metadata: {path.name}")
    commands = (root / "commands.sh").read_text()
    for command in ("build-dataset", "train-sft", "evaluate-sft"):
        _require(command in commands, f"missing recorded command: {command}")
    environment = (root / "environment.txt").read_text()
    _require("torch=" in environment and "gpu=" in environment, "incomplete environment record")

    result = {
        "run_dir": str(root.resolve()),
        "dataset": _validate_dataset(root, config),
        "model": _validate_model_outputs(root, config),
        "validated": True,
    }
    write_json(root / "validation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
