"""CompForge-specific public-observation and stateful-action codecs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import ClassVar, Mapping, Sequence

import torch

from coverage_repro.compforge import CompForgeEnv, CompForgeState, Rule, RuleAction
from coverage_repro.compforge_hindsight import policy_view


def _public_strings(row: Mapping[str, object]) -> tuple[tuple[str, ...], str]:
    public = policy_view(dict(row))
    inventory = public["inventory"]
    goal = public["goal"]
    if not isinstance(inventory, Sequence) or isinstance(inventory, (str, bytes)):
        raise ValueError("inventory must be a sequence of states")
    if not inventory or not all(isinstance(state, str) and state for state in inventory):
        raise ValueError("inventory states must be non-empty strings")
    if not isinstance(goal, str) or not goal:
        raise ValueError("goal must be a non-empty string")
    return tuple(inventory), goal


@dataclass(frozen=True)
class CompForgeTokenCodec:
    states: tuple[str, ...]

    PAD: ClassVar[int] = 0
    BOS: ClassVar[int] = 1
    EOS: ClassVar[int] = 2
    INVENTORY: ClassVar[int] = 3
    GOAL: ClassVar[int] = 4
    SEPARATOR: ClassVar[int] = 5
    STATE_OFFSET: ClassVar[int] = 6

    def __post_init__(self) -> None:
        if not self.states or tuple(sorted(set(self.states))) != self.states:
            raise ValueError("states must be a non-empty sorted unique tuple")

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, object]]) -> "CompForgeTokenCodec":
        names: set[str] = set()
        for row in rows:
            inventory, goal = _public_strings(row)
            names.update(inventory)
            names.add(goal)
        return cls(tuple(sorted(names)))

    @property
    def vocab_size(self) -> int:
        return self.STATE_OFFSET + len(self.states)

    def encode_state(self, state: str) -> int:
        try:
            return self.STATE_OFFSET + self.states.index(state)
        except ValueError as error:
            raise ValueError(f"state is outside this codec: {state}") from error

    def encode_inventory_and_goal(self, inventory: Sequence[str], goal: str) -> list[int]:
        if not inventory:
            raise ValueError("inventory must not be empty")
        tokens = [self.BOS, self.INVENTORY]
        for index, state in enumerate(inventory):
            if index:
                tokens.append(self.SEPARATOR)
            tokens.append(self.encode_state(state))
        return [*tokens, self.GOAL, self.encode_state(goal), self.EOS]


@dataclass(frozen=True)
class CompForgeActionCodec:
    rules: tuple[Rule, ...]
    max_inventory: int

    PAD: ClassVar[int] = 0
    BOS: ClassVar[int] = 1
    ACTION_OFFSET: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if not self.rules or len({rule.rule_id for rule in self.rules}) != len(self.rules):
            raise ValueError("rules must be non-empty with unique IDs")
        if isinstance(self.max_inventory, bool) or not isinstance(self.max_inventory, int) or self.max_inventory < 1:
            raise ValueError("max_inventory must be a positive integer")

    @classmethod
    def from_env(cls, env: CompForgeEnv, max_inventory: int) -> "CompForgeActionCodec":
        return cls(env.rules, max_inventory)

    @property
    def action_space(self) -> tuple[RuleAction, ...]:
        return tuple(
            RuleAction(rule.rule_id, slots)
            for rule in self.rules
            for slots in permutations(range(self.max_inventory), len(rule.inputs))
        )

    @property
    def pad_token(self) -> int:
        return self.PAD

    @property
    def bos_token(self) -> int:
        return self.BOS

    @property
    def eos_token(self) -> int:
        return self.ACTION_OFFSET + len(self.action_space)

    @property
    def vocab_size(self) -> int:
        return self.eos_token + 1

    def encode(self, action: RuleAction) -> int:
        try:
            return self.ACTION_OFFSET + self.action_space.index(action)
        except ValueError as error:
            raise ValueError("action is outside this codec") from error

    def decode(self, token: int) -> RuleAction | None:
        if token == self.eos_token:
            return None
        index = token - self.ACTION_OFFSET
        if index < 0 or index >= len(self.action_space):
            raise ValueError("token is not an action or EOS")
        return self.action_space[index]


def encode_compforge_source(row: Mapping[str, object], codec: CompForgeTokenCodec) -> list[int]:
    inventory, goal = _public_strings(row)
    return codec.encode_inventory_and_goal(inventory, goal)


def _as_rule_action(value: object) -> RuleAction:
    if isinstance(value, RuleAction):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("actions must be RuleAction values or mappings")
    rule_id = value.get("rule_id")
    slots = value.get("operand_slots")
    if not isinstance(rule_id, str) or not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
        raise ValueError("action mapping is invalid")
    return RuleAction(rule_id, tuple(int(slot) for slot in slots))


def encode_compforge_actions(row: Mapping[str, object], codec: CompForgeActionCodec) -> list[int]:
    values = row.get("actions")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("actions must be a sequence")
    return [*(codec.encode(_as_rule_action(value)) for value in values), codec.eos_token]


def _pad_sequences(sequences: Sequence[Sequence[int]], pad_token: int) -> torch.Tensor:
    if not sequences:
        raise ValueError("cannot collate an empty batch")
    width = max(len(sequence) for sequence in sequences)
    result = torch.full((len(sequences), width), pad_token, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        result[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return result


def collate_compforge_batch(
    rows: Sequence[Mapping[str, object]], token_codec: CompForgeTokenCodec, action_codec: CompForgeActionCodec
) -> dict[str, torch.Tensor]:
    sources = [encode_compforge_source(row, token_codec) for row in rows]
    targets = [encode_compforge_actions(row, action_codec) for row in rows]
    src = _pad_sequences(sources, token_codec.PAD)
    target = _pad_sequences(targets, action_codec.pad_token)
    decoder_input = _pad_sequences(
        [[action_codec.bos_token, *tokens[:-1]] for tokens in targets], action_codec.pad_token
    )
    return {
        "src": src,
        "src_padding_mask": src.eq(token_codec.PAD),
        "decoder_input": decoder_input,
        "decoder_padding_mask": decoder_input.eq(action_codec.pad_token),
        "targets": target,
        "target_padding_mask": target.eq(action_codec.pad_token),
    }


def compforge_legal_action_mask(
    env: CompForgeEnv,
    state: CompForgeState,
    codec: CompForgeActionCodec,
    *,
    goal_state: str,
) -> torch.BoolTensor:
    mask = torch.zeros(codec.vocab_size, dtype=torch.bool)
    for action in env.valid_actions(state):
        mask[codec.encode(action)] = True
    if any(item.state == goal_state for item in state.inventory):
        mask[codec.eos_token] = True
    return mask
