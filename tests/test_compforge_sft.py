import torch
from torch import nn
from pathlib import Path

from coverage_repro.compforge import CompForgeEnv, ObjectInstance, Rule, RuleAction, RuleKind
from coverage_repro.compforge_encoding import CompForgeActionCodec, CompForgeTokenCodec
from coverage_repro.compforge_sft import decode_compforge_episode
from coverage_repro.compforge_hindsight import CompForgeDatasetConfig, build_compforge_splits
from coverage_repro.compforge_sft import load_compforge_checkpoint, train_compforge_sft


class ScriptedModel(nn.Module):
    def __init__(self, vocab_size, choices):
        super().__init__()
        self.vocab_size = vocab_size
        self.choices = choices

    def forward(self, src, decoder_input, src_padding_mask=None, decoder_padding_mask=None):
        logits = torch.full((1, decoder_input.shape[1], self.vocab_size), -100.0)
        for step, choices in enumerate(self.choices[: decoder_input.shape[1]]):
            for token, score in choices:
                logits[0, step, token] = score
        return logits


def _environment_and_row():
    env = CompForgeEnv(
        (
            Rule("transform", RuleKind.TRANSFORM, ("a",), ("t",), 1),
            Rule("bind", RuleKind.BIND, ("t", "b"), ("goal",), 1),
        )
    )
    row = {
        "inventory": ["a", "b"],
        "goal": "goal",
        "actions": [
            {"rule_id": "transform", "operand_slots": [0]},
            {"rule_id": "bind", "operand_slots": [0, 1]},
        ],
    }
    inventory = (ObjectInstance(0, "a", (0,)), ObjectInstance(1, "b", (1,)))
    return env, row, inventory


def test_decode_masks_illegal_high_logit():
    env, row, inventory = _environment_and_row()
    action_codec = CompForgeActionCodec.from_env(env, max_inventory=2)
    token_codec = CompForgeTokenCodec.from_rows([row])
    transform = action_codec.encode(RuleAction("transform", (0,)))
    bind = action_codec.encode(RuleAction("bind", (0, 1)))
    illegal_bind = action_codec.encode(RuleAction("bind", (0, 1)))
    model = ScriptedModel(
        action_codec.vocab_size,
        [[(illegal_bind, 100.0), (transform, 1.0)], [(bind, 1.0)], [(action_codec.eos_token, 1.0)]],
    )

    result = decode_compforge_episode(model, row, env, inventory, (token_codec, action_codec), "cpu", 8)

    assert result.success is True
    assert result.checker_valid is True
    assert result.actions == (RuleAction("transform", (0,)), RuleAction("bind", (0, 1)))


def test_decode_masks_early_eos_until_goal_is_present():
    env, row, inventory = _environment_and_row()
    action_codec = CompForgeActionCodec.from_env(env, max_inventory=2)
    token_codec = CompForgeTokenCodec.from_rows([row])
    model = ScriptedModel(action_codec.vocab_size, [[(action_codec.eos_token, 1.0)]])

    result = decode_compforge_episode(model, row, env, inventory, (token_codec, action_codec), "cpu", 8)

    assert result.success is True
    assert result.emitted_tokens[0] != action_codec.eos_token


def test_training_saves_a_loadable_codec_checked_checkpoint(tmp_path):
    data_dir = tmp_path / "data"
    build_compforge_splits(
        CompForgeDatasetConfig(5, 10, trajectory_budget=2, max_rollout_steps=8, max_attempts=8, seed=73),
        data_dir,
    )
    result = train_compforge_sft(
        {"dataset_dir": str(data_dir), "epochs": 1, "batch_size": 2, "d_model": 16, "nhead": 4, "num_layers": 1, "dim_feedforward": 32, "seed": 7},
        tmp_path / "run",
        "cpu",
    )

    loaded = load_compforge_checkpoint(result["best_checkpoint"], "cpu", result["split_hashes"])

    assert Path(result["best_checkpoint"]).is_file()
    assert loaded.codecs.token.states
