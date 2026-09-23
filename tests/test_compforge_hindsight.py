import json

import pytest

from coverage_repro.compforge import CompForgeEnv, ObjectInstance, Rule, RuleKind, check_compforge_trace
from coverage_repro.compforge_hindsight import (
    CompForgeDatasetConfig,
    RecordedAction,
    build_compforge_splits,
    compile_actions,
    connected_coverage_graph,
    prune_causal_trace,
    policy_view,
    realized_pairs,
)


def test_connected_coverage_graph_has_requested_edge_count_and_one_component():
    graph = connected_coverage_graph(atom_count=6, edge_count=7, seed=11)

    assert graph.m == 7
    assert len(set(graph.component_labels())) == 1


def test_connected_coverage_graph_rejects_impossible_sparse_target():
    with pytest.raises(ValueError, match="at least atom_count - 1"):
        connected_coverage_graph(atom_count=6, edge_count=4, seed=11)


def test_realized_pairs_tracks_cross_operand_provenance_only():
    actions = (
        RecordedAction("transform", (0,), (3,), ((0, 2),)),
        RecordedAction("bind", (3, 1), (4,), ((0, 2), (1,))),
    )

    assert realized_pairs(actions) == {(0, 1), (1, 2)}


def test_causal_pruning_drops_unrelated_actions_and_compiles_a_replayable_trace():
    rules = (
        Rule("make-t0", RuleKind.TRANSFORM, ("a0",), ("t0",), 1),
        Rule("bind-goal", RuleKind.BIND, ("t0", "a1"), ("goal",), 1),
        Rule("unrelated", RuleKind.TRANSFORM, ("a2",), ("u2",), 1),
    )
    env = CompForgeEnv(rules)
    initial = (
        ObjectInstance(0, "a0", (0,)),
        ObjectInstance(1, "a1", (1,)),
        ObjectInstance(2, "a2", (2,)),
    )
    actions = (
        RecordedAction("make-t0", (0,), (3,), ((0,),)),
        RecordedAction("unrelated", (2,), (4,), ((2,),)),
        RecordedAction("bind-goal", (3, 1), (5,), ((0,), (1,))),
    )
    goal = ObjectInstance(5, "goal", (0, 1))

    trace = prune_causal_trace(initial, goal, actions)
    compiled = compile_actions(env, trace)
    replay = check_compforge_trace(env, trace.initial_inventory, trace.goal_object.state, compiled)

    assert tuple(action.rule_id for action in trace.actions) == ("make-t0", "bind-goal")
    assert tuple(item.instance_id for item in trace.initial_inventory) == (0, 1)
    assert tuple(action.rule_id for action in compiled) == ("make-t0", "bind-goal")
    assert replay.valid is True


def test_hindsight_splits_are_deterministic_certified_and_hide_audit_metadata(tmp_path):
    config = CompForgeDatasetConfig(
        atom_count=5,
        edge_count=10,
        trajectory_budget=2,
        max_rollout_steps=8,
        max_attempts=20,
        seed=73,
    )
    first = build_compforge_splits(config, tmp_path / "first")
    second = build_compforge_splits(config, tmp_path / "second")

    assert first["hashes"] == second["hashes"]
    assert (tmp_path / "first" / "train.jsonl").read_bytes() == (tmp_path / "second" / "train.jsonl").read_bytes()
    for split in ("train", "valid", "test"):
        rows = [json.loads(line) for line in (tmp_path / "first" / f"{split}.jsonl").read_text().splitlines()]
        assert len(rows) == 2
        for row in rows:
            assert row["checker_valid"] is True
            assert row["oracle_success"] is True
            assert set(map(tuple, row["realized_pairs"])) <= set(map(tuple, row["coverage_edges"]))
            assert policy_view(row) == {"inventory": row["inventory"], "goal": row["goal"]}
            assert set(policy_view(row)) == {"inventory", "goal"}
