import torch

from coverage_repro.encoding import (
    ActionCodec,
    TokenCodec,
    collate_batch,
    encode_actions,
    encode_source,
    legal_action_mask,
)
from coverage_repro.env import CoverageEnv, EnvState, MergeAction
from coverage_repro.graph import CoverageGraph


def training_row(**overrides):
    row = {
        "inventory": [0, 1, 2],
        "goal": {"left": {"left": 0, "right": 1}, "right": 2},
        "actions": [[0, 1], [0, 1]],
        "rho": 0.1,
        "lambda": 0.5,
        "graph_edges": [[0, 1]],
        "graph_seed": 1,
        "split": "train",
    }
    row.update(overrides)
    return row


def test_action_codec_round_trip():
    codec = ActionCodec(max_inventory=6)
    action = MergeAction(1, 4)
    assert codec.decode(codec.encode(action)) == action
    assert codec.decode(codec.eos_token) is None


def test_mask_contains_exact_environment_actions_and_only_terminal_eos():
    graph = CoverageGraph(n=4, edges={(0, 1), (2, 3)})
    state = CoverageEnv(graph).initial_state()
    codec = ActionCodec(max_inventory=4)
    mask = legal_action_mask(state, codec)
    allowed = {codec.decode(index) for index, flag in enumerate(mask) if flag}
    assert allowed == set(CoverageEnv(graph).valid_actions(state))
    assert mask[codec.eos_token].item() is False

    terminal_mask = legal_action_mask(EnvState((0,)), codec)
    assert terminal_mask[codec.eos_token].item() is True
    assert terminal_mask[: codec.eos_token].sum().item() == 0


def test_serialization_is_deterministic_and_excludes_coverage_metadata():
    first = training_row()
    second = training_row(
        inventory=[0, 1, 2],
        goal={"right": 2, "left": {"right": 1, "left": 0}},
        rho=0.9,
        **{"lambda": 9.0, "graph_edges": [[0, 2], [1, 2]], "graph_seed": 99, "split": "test"},
    )
    assert encode_source(first) == encode_source(second)


def test_collator_builds_shifted_decoder_inputs_and_padding_masks():
    token_codec = TokenCodec()
    action_codec = ActionCodec(max_inventory=3)
    batch = collate_batch([training_row()], token_codec, action_codec)

    assert batch["src"].dtype == torch.long
    assert batch["src_padding_mask"].dtype == torch.bool
    assert batch["decoder_input"][0, 0].item() == action_codec.bos_token
    assert batch["decoder_input"][0, 1].item() == action_codec.encode(MergeAction(0, 1))
    assert batch["targets"][0, -1].item() == action_codec.eos_token
    assert batch["target_padding_mask"].dtype == torch.bool
    assert encode_actions(training_row(), action_codec)[-1] == action_codec.eos_token
