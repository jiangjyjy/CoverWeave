import json
import torch

from coverage_repro.compforge import RuleAction, check_compforge_trace
from coverage_repro.compforge_encoding import CompForgeActionCodec
from coverage_repro.compforge_v3 import (
    CompForgeV3Config,
    build_branching_instance,
    build_compforge_v3_splits,
    validate_compforge_v3_manifest,
    certify_branching_task,
    classify_legal_actions,
    environment_from_v3_row,
    goal_blind_legal_action_mask,
    inventory_from_v3_row,
)


def test_same_inventory_different_goals_require_different_first_actions():
    config = CompForgeV3Config()

    left = build_branching_instance(config, seed=11, goal_variant=0)
    right = build_branching_instance(config, seed=11, goal_variant=1)

    assert left["inventory"] == right["inventory"]
    assert left["oracle_actions"][0] != right["oracle_actions"][0]
    assert min(left["legal_action_counts"]) >= 3


def test_v3_legality_mask_is_goal_blind_for_identical_states():
    config = CompForgeV3Config()
    left = build_branching_instance(config, seed=11, goal_variant=0)
    right = build_branching_instance(config, seed=11, goal_variant=1)
    env = environment_from_v3_row(left)
    state = env.initial_state(inventory_from_v3_row(left))
    codec = CompForgeActionCodec.from_env(env, max_inventory=3)

    left_mask = goal_blind_legal_action_mask(env, state, codec)
    right_mask = goal_blind_legal_action_mask(env, state, codec)

    assert torch.equal(left_mask, right_mask)
    assert left_mask[codec.eos_token].item() is False

    intermediate = env.apply(state, next(action for action in env.valid_actions(state) if action.rule_id == "start:red"))
    intermediate_mask = goal_blind_legal_action_mask(env, intermediate, codec)
    assert intermediate_mask[codec.eos_token].item() is False

    red_state = intermediate
    for rule_id in ("red:step:0", "red:finish"):
        red_state = env.apply(red_state, next(action for action in env.valid_actions(red_state) if action.rule_id == rule_id))
    assert goal_blind_legal_action_mask(env, red_state, codec)[codec.eos_token].item() is True

    blue_state = state
    for rule_id in ("start:blue", "blue:step:0", "blue:finish"):
        blue_state = env.apply(blue_state, next(action for action in env.valid_actions(blue_state) if action.rule_id == rule_id))
    assert goal_blind_legal_action_mask(env, blue_state, codec)[codec.eos_token].item() is True


def test_certification_proves_oracle_and_harmful_legal_branches():
    row = build_branching_instance(CompForgeV3Config(), seed=11, goal_variant=0)
    env = environment_from_v3_row(row)
    inventory = inventory_from_v3_row(row)
    state = env.initial_state(inventory)
    classes = classify_legal_actions(env, state, row["goal"], max_expansions=100)

    assert classes["useful"]
    assert classes["distractor"]
    assert classes["harmful"]
    assert certify_branching_task(row)["certified"] is True
    assert check_compforge_trace(
        env,
        inventory,
        row["goal"],
        tuple(
            RuleAction(value["rule_id"], tuple(value["operand_slots"]))
            for value in row["oracle_actions"]
        ),
    ).valid is True
    assert check_compforge_trace(
        env,
        inventory,
        row["goal"],
        (classes["harmful"][0],),
    ).valid is False

def test_v3_splits_are_disjoint_branching_and_checker_certified(tmp_path):
    config = CompForgeV3Config(atom_count=5, train_edge_count=6, samples_per_split=2)
    manifest = build_compforge_v3_splits(config, tmp_path)
    rows = {
        split: [json.loads(line) for line in (tmp_path / f"{split}.jsonl").read_text().splitlines()]
        for split in manifest["splits"]
    }
    signatures = [row["task_signature"] for split_rows in rows.values() for row in split_rows]
    assert manifest["summary"]["signature_overlap_count"] == 0
    assert manifest["summary"]["minimum_decision_branching"] >= 3
    assert manifest["summary"]["oracle_exact"] == 1.0
    assert len(signatures) == len(set(signatures))
    assert validate_compforge_v3_manifest(manifest, rows)["valid"] is True
