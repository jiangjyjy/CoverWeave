import torch
from torch import nn

from coverage_repro.encoding import ActionCodec, TokenCodec
from coverage_repro.env import MergeAction
from coverage_repro.model_sft import (
    MergeTransformer,
    TransformerConfig,
    decode_episode,
)


def row_for_goal(goal, actions=None):
    return {
        "N": 3,
        "inventory": [0, 1, 2],
        "goal": goal,
        "actions": actions or [],
        "graph_edges": [[0, 1]],
        "rho": 0.1,
        "lambda": 0.5,
        "graph_seed": 1,
        "split": "test",
    }


def codecs():
    return TokenCodec(), ActionCodec(max_inventory=3)


def solved_row():
    return row_for_goal(
        {"left": {"left": 0, "right": 1}, "right": 2},
        actions=[[0, 1], [0, 1]],
    )


class ScriptedModel(nn.Module):
    def __init__(self, action_codec, plans):
        super().__init__()
        self.action_codec = action_codec
        self.plans = plans

    def forward(self, src, decoder_input, src_padding_mask, decoder_padding_mask):
        logits = torch.full(
            (src.size(0), decoder_input.size(1), self.action_codec.vocab_size),
            -10.0,
            device=src.device,
        )
        for token, score in self.plans[min(decoder_input.size(1) - 1, len(self.plans) - 1)]:
            logits[:, -1, token] = score
        return logits


def plan_for(codec, *items):
    result = []
    for item in items:
        if item == "eos":
            result.append([(codec.eos_token, 1.0)])
        else:
            result.append([(codec.encode(MergeAction(*item)), 1.0)])
    return result


def test_transformer_logits_shape():
    model = MergeTransformer(
        TransformerConfig(
            src_vocab_size=128,
            action_vocab_size=18,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )
    )
    src = torch.randint(0, 128, (2, 7))
    decoder_input = torch.randint(0, 18, (2, 5))
    src_padding_mask = torch.zeros((2, 7), dtype=torch.bool)
    decoder_padding_mask = torch.zeros((2, 5), dtype=torch.bool)
    logits = model(src, decoder_input, src_padding_mask, decoder_padding_mask)
    assert logits.shape == (2, 5, 18)


def test_decode_masks_illegal_high_logits_and_solves_exact_goal():
    row = solved_row()
    token_codec = TokenCodec()
    action_codec = ActionCodec(max_inventory=4)
    illegal = action_codec.encode(MergeAction(2, 3))
    legal = action_codec.encode(MergeAction(0, 1))
    model = ScriptedModel(
        action_codec,
        [[(illegal, 100.0), (legal, 1.0)], [(illegal, 100.0), (legal, 1.0)], [(action_codec.eos_token, 1.0)]],
    )
    result = decode_episode(model, row, (token_codec, action_codec), "cpu", 8)
    assert result.success is True
    assert result.failure_reason == "none"
    assert result.checker_valid is True
    assert result.valid_action_rate == 1.0
    assert result.actions == (MergeAction(0, 1), MergeAction(0, 1))
    assert all(
        action in result.legal_actions_at_step[index]
        for index, action in enumerate(result.actions)
    )


def test_decode_reports_wrong_final_tree():
    row = solved_row()
    token_codec, action_codec = codecs()
    model = ScriptedModel(action_codec, plan_for(action_codec, (0, 2), (0, 1), "eos"))
    result = decode_episode(model, row, (token_codec, action_codec), "cpu", 8)
    assert result.success is False
    assert result.failure_reason == "wrong_final_tree"
    assert result.checker_valid is False


def test_decode_reports_budget_exhaustion():
    row = solved_row()
    token_codec, action_codec = codecs()
    model = ScriptedModel(action_codec, plan_for(action_codec, (0, 1), (0, 1)))
    result = decode_episode(model, row, (token_codec, action_codec), "cpu", 1)
    assert result.success is False
    assert result.failure_reason == "budget_exhausted"


def test_decode_reports_malformed_action():
    row = solved_row()
    token_codec = TokenCodec()
    action_codec = ActionCodec(max_inventory=1)
    model = ScriptedModel(action_codec, [[(action_codec.pad_token, 10.0)]])
    result = decode_episode(model, row, (token_codec, action_codec), "cpu", 8)
    assert result.success is False
    assert result.failure_reason == "malformed_action"


def test_decode_uses_row_inventory_instead_of_goal_leaves():
    row = row_for_goal(
        {"left": {"left": 2, "right": 1}, "right": 0},
        actions=[[0, 2], [0, 1]],
    )
    row["inventory"] = [2, 0, 1]
    token_codec, action_codec = codecs()
    model = ScriptedModel(action_codec, plan_for(action_codec, (0, 2), (0, 1), "eos"))
    result = decode_episode(model, row, (token_codec, action_codec), "cpu", 8)
    assert result.success is True
    assert result.actions == (MergeAction(0, 2), MergeAction(0, 1))


def test_transformer_passes_source_padding_to_decoder_cross_attention(monkeypatch):
    model = MergeTransformer(
        TransformerConfig(
            src_vocab_size=32,
            action_vocab_size=8,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
    )
    src = torch.randint(0, 32, (1, 5))
    decoder_input = torch.randint(0, 8, (1, 3))
    src_padding_mask = torch.tensor([[False, False, True, True, True]])
    decoder_padding_mask = torch.zeros((1, 3), dtype=torch.bool)
    captured = {}
    original_forward = model.transformer.forward

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.transformer, "forward", spy)
    model(src, decoder_input, src_padding_mask, decoder_padding_mask)
    assert torch.equal(captured["memory_key_padding_mask"], src_padding_mask)
