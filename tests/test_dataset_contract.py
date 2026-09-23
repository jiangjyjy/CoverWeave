from itertools import combinations

import pytest

from coverage_repro.baselines import bfs_oracle
from coverage_repro.checker import check_trace
from coverage_repro.dataset import build_splits
from coverage_repro.env import MergeAction
from coverage_repro.graph import CoverageGraph, tree_from_dict
from coverage_repro.io import read_jsonl


def contract_config():
    return {
        "N": 5,
        "lambda_values": [1.0],
        "graph_seeds": 1,
        "train_depths": [3],
        "test_depths": [3],
        "train_samples": 2,
        "valid_samples": 1,
        "test_samples": 1,
        "seed_base": 19,
    }


def _replay(row):
    graph = CoverageGraph(row["N"], [tuple(edge) for edge in row["graph_edges"]])
    goal = tree_from_dict(row["goal"])
    actions = [MergeAction(*action) for action in row["actions"]]
    assert bfs_oracle(graph, goal).success is True
    assert check_trace(graph, goal, actions).valid is True


def test_coverage_graph_contract_shares_target_edges_and_tracks_held_out_pairs(tmp_path):
    manifest = build_splits(contract_config(), tmp_path)
    rows_by_split = {
        split: read_jsonl(tmp_path / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }
    target_edges = {tuple(edge) for edge in rows_by_split["train"][0]["graph_edges"]}
    assert target_edges == {tuple(edge) for edge in rows_by_split["valid"][0]["graph_edges"]}
    assert target_edges == {tuple(edge) for edge in rows_by_split["test"][0]["graph_edges"]}
    train_pairs = {tuple(pair) for row in rows_by_split["train"] for pair in row["realized_pairs"]}
    valid_pairs = {tuple(pair) for row in rows_by_split["valid"] for pair in row["realized_pairs"]}
    test_pairs = {tuple(pair) for row in rows_by_split["test"] for pair in row["realized_pairs"]}
    all_pairs = set(combinations(range(contract_config()["N"]), 2))

    assert train_pairs == target_edges
    assert valid_pairs <= target_edges
    assert test_pairs & (all_pairs - target_edges)
    cell = manifest["coverage_cells"][0]
    assert {tuple(edge) for edge in cell["target_edges"]} == target_edges
    assert {tuple(edge) for edge in cell["realized_edges"]["train"]} == target_edges
    assert cell["realized_rho"]["train"] == cell["target_rho"]

    for rows in rows_by_split.values():
        for row in rows:
            assert row["checker_valid"] is True
            _replay(row)


def test_train_samples_must_cover_every_target_edge(tmp_path):
    config = contract_config()
    config["train_samples"] = 1
    with pytest.raises(ValueError, match="train_samples"):
        build_splits(config, tmp_path)

def test_sparse_smoke_cells_do_not_require_large_components(tmp_path):
    config = {
        "N": 12,
        "lambda_values": [0.6, 1.0, 1.5],
        "graph_seeds": 1,
        "train_depths": [3],
        "test_depths": [3],
        "train_samples": 9,
        "valid_samples": 1,
        "test_samples": 1,
        "seed_base": 20260806,
    }
    build_splits(config, tmp_path)

    for split in ("valid", "test"):
        rows = read_jsonl(tmp_path / f"{split}.jsonl")
        assert len(rows) == len(config["lambda_values"])
        for row in rows:
            assert row["checker_valid"] is True
            _replay(row)
