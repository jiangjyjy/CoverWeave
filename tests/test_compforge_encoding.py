import pytest

from coverage_repro.compforge import CompForgeEnv, ObjectInstance, Rule, RuleKind
from coverage_repro.compforge_hindsight import policy_view
from coverage_repro.compforge_encoding import (
    CompForgeActionCodec,
    CompForgeTokenCodec,
    collate_compforge_batch,
    compforge_legal_action_mask,
    encode_compforge_actions,
    encode_compforge_source,
)


@pytest.fixture
def env():
    return CompForgeEnv(
        (
            Rule("transform", RuleKind.TRANSFORM, ("a",), ("t",), 1),
            Rule("bind", RuleKind.BIND, ("t", "b"), ("goal",), 1),
        )
    )


@pytest.fixture
def row():
    return {
        "inventory": ["a", "b"],
        "goal": "goal",
        "actions": [
            {"rule_id": "transform", "operand_slots": [0]},
            {"rule_id": "bind", "operand_slots": [0, 1]},
        ],
        "coverage_edges": [[0, 1]],
        "seed": 4,
        "instance": {"private": True},
    }


def test_source_uses_only_public_policy_fields(row):
    codec = CompForgeTokenCodec.from_rows([row])
    changed = dict(row, actions=[], coverage_edges=[], seed=99, instance={"private": False})

    assert policy_view(row) == {"inventory": ["a", "b"], "goal": "goal"}
    assert encode_compforge_source(row, codec) == encode_compforge_source(changed, codec)


def test_action_codec_round_trips_and_mask_matches_legal_actions(env, row):
    state = env.initial_state((ObjectInstance(0, "a", (0,)), ObjectInstance(1, "b", (1,))))
    codec = CompForgeActionCodec.from_env(env, max_inventory=3)

    assert {codec.decode(codec.encode(action)) for action in env.valid_actions(state)} == set(env.valid_actions(state))
    mask = compforge_legal_action_mask(env, state, codec, goal_state=row["goal"])
    enabled = {codec.decode(token) for token in range(codec.vocab_size) if mask[token] and token != codec.eos_token}
    assert enabled == set(env.valid_actions(state))
    assert mask[codec.eos_token].item() is False


def test_collation_prepends_bos_and_targets_actions(row, env):
    token_codec = CompForgeTokenCodec.from_rows([row])
    action_codec = CompForgeActionCodec.from_env(env, max_inventory=3)

    batch = collate_compforge_batch([row], token_codec, action_codec)

    assert batch["decoder_input"][0, 0].item() == action_codec.bos_token
    assert batch["targets"][0].tolist()[:3] == encode_compforge_actions(row, action_codec)
