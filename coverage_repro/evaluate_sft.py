from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping

from .checker import check_trace
from .env import CoverageEnv, EnvState, MergeAction
from .graph import CoverageGraph, Node, Tree, tree_from_dict, tree_subtrees
from .io import assert_finite, read_jsonl, sha256_file, write_json, write_jsonl
from .model_sft import DecodeResult, decode_episode
from .percolation import summarize_results
from .train_sft import load_checkpoint

_SPLITS = ("train", "valid", "test")
_FAILURE_REASONS = ("none", "budget_exhausted", "wrong_final_tree", "malformed_action")


@dataclass(frozen=True)
class _OracleResult:
    success: bool
    actions: tuple[MergeAction, ...]


def _load_verified_manifest(dataset_dir: Path) -> tuple[dict[str, object], dict[str, str]]:
    manifest_path = dataset_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}") from None
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be a mapping")
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(_SPLITS):
        raise ValueError("dataset manifest must contain exactly train/valid/test hashes")
    verified: dict[str, str] = {}
    for split in _SPLITS:
        split_path = dataset_dir / f"{split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        expected = hashes[split]
        if not isinstance(expected, str):
            raise ValueError(f"manifest hash for {split} must be a string")
        actual = sha256_file(split_path)
        if actual != expected:
            raise ValueError(f"manifest hash mismatch for {split}")
        verified[split] = actual
    return manifest, verified


def _bfs_from_inventory(
    graph: CoverageGraph,
    goal: Tree,
    inventory: tuple[Tree, ...],
    max_states: int = 100000,
) -> _OracleResult:
    env = CoverageEnv(graph)
    target_nodes = set(tree_subtrees(goal))
    initial = EnvState(inventory)
    queue = deque([(initial, tuple())])
    seen = {initial.inventory}
    explored = 0
    while queue and explored < max_states:
        state, actions = queue.popleft()
        explored += 1
        if len(state.inventory) == 1:
            if state.inventory[0] == goal:
                return _OracleResult(True, actions)
            continue
        for action in env.valid_actions(state):
            merged = Node(
                state.inventory[action.left_index],
                state.inventory[action.right_index],
            )
            if merged not in target_nodes:
                continue
            next_state = env.apply(state, action)
            if next_state.inventory in seen:
                continue
            seen.add(next_state.inventory)
            queue.append((next_state, actions + (action,)))
    return _OracleResult(False, tuple())


def _expected_cells(
    manifest: Mapping[str, object],
    model_seed: int,
) -> set[tuple[float, int, int, int]]:
    config = manifest.get("config")
    coverage_cells = manifest.get("coverage_cells")
    if not isinstance(config, Mapping) or not isinstance(coverage_cells, list):
        raise ValueError("dataset manifest is missing config or coverage cells")
    depths = config.get("test_depths")
    if not isinstance(depths, list) or not depths:
        raise ValueError("dataset manifest has no test depths")
    expected: set[tuple[float, int, int, int]] = set()
    for cell in coverage_cells:
        if not isinstance(cell, Mapping):
            raise ValueError("dataset coverage cell must be a mapping")
        for depth in depths:
            expected.add(
                (
                    float(cell["lambda"]),
                    int(depth),
                    int(cell["graph_seed"]),
                    model_seed,
                )
            )
    if not expected:
        raise ValueError("dataset manifest contains no evaluation cells")
    return expected


def _decode_without_targets(
    row: Mapping[str, object],
    model: object,
    codecs: object,
    device: str,
    action_budget: int,
) -> DecodeResult:
    source_row = {key: value for key, value in row.items() if key != "actions"}
    return decode_episode(model, source_row, codecs, device, action_budget)


def _aggregate_cell(
    key: tuple[float, int, int, int],
    episodes: list[dict[str, object]],
) -> dict[str, object]:
    successes = sum(int(episode["success"]) for episode in episodes)
    oracle_successes = sum(int(episode["bfs_success"]) for episode in episodes)
    count = len(episodes)
    failure_counts = Counter({reason: 0 for reason in _FAILURE_REASONS})
    failure_counts.update(str(episode["failure_reason"]) for episode in episodes)
    return {
        "lambda": key[0],
        "depth": key[1],
        "graph_seed": key[2],
        "model_seed": key[3],
        "episodes": count,
        "success": successes,
        "exact_solve_rate": successes / count,
        "success_rate": successes / count,
        "checker_valid_rate": mean(float(episode["checker_valid"]) for episode in episodes),
        "valid_action_rate": mean(float(episode["valid_action_rate"]) for episode in episodes),
        "mean_action_count": mean(int(episode["action_count"]) for episode in episodes),
        "bfs_oracle_success": oracle_successes,
        "bfs_oracle_success_rate": oracle_successes / count,
        "bfs_oracle_mean_action_count": mean(
            int(episode["bfs_action_count"]) for episode in episodes
        ),
        "bfs_cost_ratio": mean(float(episode["bfs_cost_ratio"]) for episode in episodes),
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "backend": "transformer-sft",
    }


def evaluate_checkpoint(
    checkpoint: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    device: str,
) -> dict[str, object]:
    dataset_path = Path(dataset_dir)
    manifest, split_hashes = _load_verified_manifest(dataset_path)
    bundle = load_checkpoint(checkpoint, device, expected_split_hashes=split_hashes)
    model_seed = bundle.config.get("seed")
    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise ValueError("checkpoint training config must contain an integer seed")
    action_budget_value = bundle.config.get("action_budget", bundle.model_config.max_target_len)
    if (
        isinstance(action_budget_value, bool)
        or not isinstance(action_budget_value, int)
        or action_budget_value < 1
    ):
        raise ValueError("checkpoint action budget must be a positive integer")
    action_budget = action_budget_value

    rows = read_jsonl(dataset_path / "test.jsonl")
    if not rows:
        raise ValueError("test split is empty")
    expected_cells = _expected_cells(manifest, model_seed)
    grouped: dict[tuple[float, int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        graph = CoverageGraph(
            int(row["N"]),
            [tuple(edge) for edge in row.get("graph_edges", [])],
        )
        goal = tree_from_dict(row["goal"])
        inventory = tuple(tree_from_dict(item) for item in row["inventory"])
        oracle = _bfs_from_inventory(graph, goal, inventory)
        if not oracle.success:
            raise ValueError(f"BFS oracle failed for sample {row.get('sample_id', '<unknown>')}")
        oracle_check = check_trace(graph, goal, oracle.actions, inventory=inventory)
        if not oracle_check.valid:
            raise ValueError(f"BFS oracle trace failed checker: {oracle_check.reason}")

        decoded = _decode_without_targets(
            row,
            bundle.model,
            bundle.codecs,
            device,
            action_budget,
        )
        model_check = check_trace(graph, goal, decoded.actions, inventory=inventory)
        oracle_cost = len(oracle.actions)
        cost_ratio = len(decoded.actions) / max(oracle_cost, 1)
        key = (
            float(row["lambda"]),
            int(row["depth"]),
            int(row["graph_seed"]),
            model_seed,
        )
        grouped[key].append(
            {
                "success": decoded.success and model_check.valid,
                "checker_valid": model_check.valid,
                "valid_action_rate": decoded.valid_action_rate,
                "action_count": len(decoded.actions),
                "failure_reason": decoded.failure_reason,
                "bfs_success": oracle.success,
                "bfs_action_count": oracle_cost,
                "bfs_cost_ratio": cost_ratio,
            }
        )

    observed_cells = set(grouped)
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ValueError(f"evaluation cells are incomplete: missing={missing}, extra={extra}")

    result_rows = [
        _aggregate_cell(key, grouped[key])
        for key in sorted(grouped)
    ]
    n_value = manifest.get("config", {}).get("N") if isinstance(manifest.get("config"), Mapping) else None
    if isinstance(n_value, bool) or not isinstance(n_value, int) or n_value < 1:
        raise ValueError("dataset manifest config must contain a positive N")
    analysis = summarize_results(result_rows, n_value)
    analysis.update(
        {
            "finite": True,
            "result_rows": len(result_rows),
            "episodes": sum(int(row["episodes"]) for row in result_rows),
            "model_seed": model_seed,
            "split_hashes": split_hashes,
        }
    )
    assert_finite(result_rows)
    assert_finite(analysis)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path / "model_results.jsonl", result_rows)
    write_json(output_path / "analysis.json", analysis)
    return analysis
