from pathlib import Path

from coverage_repro.baselines import bfs_oracle
from coverage_repro.checker import check_trace
from coverage_repro.dataset import build_splits
from coverage_repro.env import CoverageEnv, MergeAction
from coverage_repro.graph import CoverageGraph, Node, tree_depth, tree_from_dict
from coverage_repro.io import read_jsonl


def diversity_config():
    return {
        "N": 12,
        "lambda_values": [1.5],
        "graph_seeds": 1,
        "train_depths": [3, 4],
        "test_depths": [3, 4],
        "train_samples": 9,
        "valid_samples": 1,
        "test_samples": 1,
        "seed_base": 20260806,
    }


def test_dataset_records_keep_depth_and_leaf_count_contract(tmp_path):
    build_splits(diversity_config(), tmp_path)
    rows = read_jsonl(Path(tmp_path) / "train.jsonl")
    for row in rows:
        goal = tree_from_dict(row["goal"])
        assert row["leaf_count"] == 2 * row["depth"]
        assert tree_depth(goal) == row["depth"]
        assert len(row["inventory"]) == row["leaf_count"]
        assert tuple(tree_from_dict(item) for item in row["inventory"]) == goal.leaves()
        assert check_trace(
            CoverageGraph(row["N"], [tuple(edge) for edge in row["graph_edges"]]),
            goal,
            [MergeAction(*action) for action in row["actions"]],
        ).valid is True


def test_train_action_sequences_are_not_constant_by_depth(tmp_path):
    build_splits(diversity_config(), tmp_path)
    rows = read_jsonl(Path(tmp_path) / "train.jsonl")
    for depth in (3, 4):
        sequences = {
            tuple(tuple(action) for action in row["actions"])
            for row in rows
            if row["depth"] == depth
        }
        assert len(sequences) > 1


def test_same_inventory_different_goals_produce_different_bfs_labels():
    graph = CoverageGraph.complete(4)
    goal_a = Node(Node(0, 1), Node(2, 3))
    goal_b = Node(0, Node(1, Node(2, 3)))
    assert CoverageEnv(graph).initial_state(goal_a).inventory == CoverageEnv(graph).initial_state(goal_b).inventory
    trace_a = bfs_oracle(graph, goal_a)
    trace_b = bfs_oracle(graph, goal_b)
    assert trace_a.success is True
    assert trace_b.success is True
    assert trace_a.actions != trace_b.actions
