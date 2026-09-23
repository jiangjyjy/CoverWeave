from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from random import Random

from .baselines import bfs_oracle
from .checker import check_trace
from .env import CoverageEnv
from .graph import CoverageGraph, Node, Tree, make_goal, tree_to_dict
from .io import sha256_file, write_json, write_jsonl


_SPLIT_SEED_OFFSETS = {"train": 0, "valid": 1_000_000, "test": 2_000_000}


@dataclass(frozen=True)
class DatasetRecord:
    """Auditable trajectory record; graph_edges and rho are not model inputs."""

    sample_id: str
    split: str
    N: int
    lambda_value: float
    rho: float
    graph_edges: tuple[tuple[int, int], ...]
    inventory: tuple[object, ...]
    goal: object
    actions: tuple[tuple[int, int], ...]
    realized_pairs: tuple[tuple[int, int], ...]
    depth: int
    leaf_count: int
    graph_seed: int
    episode_seed: int
    checker_valid: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "N": self.N,
            "lambda": self.lambda_value,
            "rho": self.rho,
            "graph_edges": self.graph_edges,
            "inventory": self.inventory,
            "goal": self.goal,
            "actions": self.actions,
            "realized_pairs": self.realized_pairs,
            "depth": self.depth,
            "leaf_count": self.leaf_count,
            "graph_seed": self.graph_seed,
            "episode_seed": self.episode_seed,
            "checker_valid": self.checker_valid,
        }


@dataclass(frozen=True)
class _GraphCell:
    lambda_value: float
    graph_seed: int
    graph: CoverageGraph


def _positive_int(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _depths(config: dict[str, object], key: str) -> tuple[int, ...]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list of positive integers")
    if any(isinstance(depth, bool) or not isinstance(depth, int) or depth < 1 for depth in value):
        raise ValueError(f"{key} must be a non-empty list of positive integers")
    return tuple(value)


def _lambda_values(config: dict[str, object]) -> tuple[float, ...]:
    value = config.get("lambda_values")
    if not isinstance(value, list) or not value:
        raise ValueError("lambda_values must be a non-empty list")
    values = tuple(float(item) for item in value)
    if any(item < 0 for item in values):
        raise ValueError("lambda_values must be non-negative")
    return values


def _seed_base(config: dict[str, object]) -> int:
    value = config.get("seed_base", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed_base must be an integer")
    return value


def _pair(left: int, right: int) -> tuple[int, int]:
    return tuple(sorted((left, right)))


def _edge_rho(n: int, edges: set[tuple[int, int]]) -> float:
    return 2.0 * len(edges) / (n * (n - 1))


def _map_occurrence_tree(tree: Tree, labels: tuple[int, ...]) -> Tree:
    if isinstance(tree, int):
        return labels[tree]
    return Node(
        _map_occurrence_tree(tree.left, labels),
        _map_occurrence_tree(tree.right, labels),
    )


def _walk_leaves(
    n: int,
    target_edges: tuple[tuple[int, int], ...],
    leaf_count: int,
    rng: Random,
    forced_edge: tuple[int, int] | None = None,
) -> tuple[int, ...]:
    if not target_edges:
        raise ValueError("train and valid splits require a non-empty target edge set")
    adjacency = {atom: [] for atom in range(n)}
    for left, right in target_edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    if forced_edge is None:
        start = rng.choice([atom for atom, neighbors in adjacency.items() if neighbors])
        leaves = [start]
    else:
        left, right = forced_edge
        leaves = [left, right] if rng.randrange(2) == 0 else [right, left]
    while len(leaves) < leaf_count:
        leaves.append(rng.choice(adjacency[leaves[-1]]))
    return tuple(leaves)


def _test_leaves(
    n: int,
    target_edges: tuple[tuple[int, int], ...],
    leaf_count: int,
    rng: Random,
) -> tuple[int, ...]:
    all_edges = set(combinations(range(n), 2))
    complement = tuple(sorted(all_edges - set(target_edges)))
    if complement:
        left, right = rng.choice(complement)
        leaves = [left, right] if rng.randrange(2) == 0 else [right, left]
    else:
        leaves = [rng.randrange(n)]
    while len(leaves) < leaf_count:
        candidate = rng.randrange(n - 1)
        if candidate >= leaves[-1]:
            candidate += 1
        leaves.append(candidate)
    return tuple(leaves)


def _make_record(
    split: str,
    sample_id: str,
    cell: _GraphCell,
    depth: int,
    episode_index: int,
    episode_seed: int,
) -> DatasetRecord:
    target_edges = tuple(sorted(cell.graph.edges))
    leaf_count = 2 * depth
    rng = Random(episode_seed)
    if split == "train":
        leaves = _walk_leaves(
            cell.graph.n, target_edges, leaf_count, rng, target_edges[episode_index % len(target_edges)]
        )
    elif split == "valid":
        leaves = _walk_leaves(cell.graph.n, target_edges, leaf_count, rng)
    else:
        leaves = _test_leaves(cell.graph.n, target_edges, leaf_count, rng)

    occurrence_goal = make_goal(leaf_count, depth, seed=episode_seed)
    goal = _map_occurrence_tree(occurrence_goal, leaves)
    realized_pairs = tuple(_pair(left, right) for left, right in zip(leaves, leaves[1:]))
    target_set = set(target_edges)
    complement = set(combinations(range(cell.graph.n), 2)) - target_set
    if split in {"train", "valid"} and not set(realized_pairs) <= target_set:
        raise AssertionError("walk yielded a pair outside the target edge set")
    if split == "test" and complement and not set(realized_pairs) & complement:
        raise AssertionError("test trajectory omitted a held-out pair")

    oracle = bfs_oracle(cell.graph, goal)
    if not oracle.success:
        raise AssertionError("BFS failed for an unrestricted merge target")
    checked = check_trace(cell.graph, goal, oracle.actions)
    if not checked.valid:
        raise AssertionError(f"checker rejected BFS trace: {checked.reason}")

    inventory = CoverageEnv(cell.graph).initial_state(goal).inventory
    return DatasetRecord(
        sample_id=sample_id,
        split=split,
        N=cell.graph.n,
        lambda_value=cell.lambda_value,
        rho=cell.graph.rho,
        graph_edges=target_edges,
        inventory=tuple(tree_to_dict(item) for item in inventory),
        goal=tree_to_dict(goal),
        actions=tuple((action.left_index, action.right_index) for action in oracle.actions),
        realized_pairs=realized_pairs,
        depth=depth,
        leaf_count=leaf_count,
        graph_seed=cell.graph_seed,
        episode_seed=episode_seed,
        checker_valid=checked.valid,
    )


def build_splits(config: dict[str, object], output_dir: str | Path) -> dict[str, object]:
    """Build deterministic, checker-verified data from cooccurrence graph targets."""
    n = _positive_int(config, "N")
    graph_seed_count = _positive_int(config, "graph_seeds")
    lambda_values = _lambda_values(config)
    train_depths = _depths(config, "train_depths")
    test_depths = _depths(config, "test_depths")
    train_samples = _positive_int(config, "train_samples")
    valid_samples = _positive_int(config, "valid_samples")
    test_samples = _positive_int(config, "test_samples")
    seed_base = _seed_base(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cells: list[_GraphCell] = []
    for lambda_index, lambda_value in enumerate(lambda_values):
        for graph_index in range(graph_seed_count):
            graph_seed = seed_base + lambda_index * graph_seed_count + graph_index
            graph = CoverageGraph.random(n=n, lam=lambda_value, seed=graph_seed)
            if graph.m == 0:
                raise ValueError("train and valid splits require target graphs with at least one edge")
            if train_samples < graph.m:
                raise ValueError("train_samples must be at least the target edge count for every graph cell")
            cells.append(_GraphCell(lambda_value, graph_seed, graph))

    split_specs = {
        "train": (train_samples, train_depths, _SPLIT_SEED_OFFSETS["train"]),
        "valid": (valid_samples, train_depths, _SPLIT_SEED_OFFSETS["valid"]),
        "test": (test_samples, test_depths, _SPLIT_SEED_OFFSETS["test"]),
    }
    hashes: dict[str, str] = {}
    split_sizes: dict[str, int] = {}
    depth_histograms: dict[str, dict[str, int]] = {}
    realized_by_cell = [{split: set() for split in split_specs} for _ in cells]

    for split, (samples_per_cell, depths, offset) in split_specs.items():
        rows: list[dict[str, object]] = []
        cell_number = 0
        for cell_index, cell in enumerate(cells):
            for depth in depths:
                for episode_index in range(samples_per_cell):
                    episode_seed = seed_base + offset + cell_number * samples_per_cell + episode_index
                    record = _make_record(
                        split,
                        f"{split}-{offset + cell_number:07d}-{episode_index:06d}",
                        cell,
                        depth,
                        episode_index,
                        episode_seed,
                    )
                    rows.append(record.to_dict())
                    realized_by_cell[cell_index][split].update(record.realized_pairs)
                cell_number += 1
        split_path = output_path / f"{split}.jsonl"
        write_jsonl(split_path, rows)
        hashes[split] = sha256_file(split_path)
        split_sizes[split] = len(rows)
        depth_histograms[split] = {
            str(depth): count for depth, count in sorted(Counter(row["depth"] for row in rows).items())
        }

    coverage_cells = []
    for cell, realized in zip(cells, realized_by_cell):
        coverage_cells.append(
            {
                "lambda": cell.lambda_value,
                "graph_seed": cell.graph_seed,
                "target_edges": tuple(sorted(cell.graph.edges)),
                "target_rho": cell.graph.rho,
                "realized_edges": {split: tuple(sorted(edges)) for split, edges in realized.items()},
                "realized_rho": {split: _edge_rho(n, edges) for split, edges in realized.items()},
            }
        )
    manifest = {
        "config": config,
        "split_sizes": split_sizes,
        "depth_histograms": depth_histograms,
        "hashes": hashes,
        "coverage_cells": coverage_cells,
    }
    write_json(output_path / "manifest.json", manifest)
    return manifest
