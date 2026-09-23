from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping, Sequence

import torch

from .env import EnvState, MergeAction


@dataclass(frozen=True)
class TokenCodec:
    """Deterministic source codec for inventory and goal trees only."""

    PAD: ClassVar[int] = 0
    BOS: ClassVar[int] = 1
    EOS: ClassVar[int] = 2
    INVENTORY: ClassVar[int] = 3
    GOAL: ClassVar[int] = 4
    LPAREN: ClassVar[int] = 5
    RPAREN: ClassVar[int] = 6
    COMMA: ClassVar[int] = 7
    ATOM_OFFSET: ClassVar[int] = 8

    def encode_atom(self, atom: int) -> int:
        if isinstance(atom, bool) or not isinstance(atom, int) or atom < 0:
            raise ValueError("atom ids must be non-negative integers")
        return self.ATOM_OFFSET + atom

    def encode_tree(self, tree: object) -> list[int]:
        if isinstance(tree, bool):
            raise ValueError("tree atoms must be integers")
        if isinstance(tree, int):
            return [self.encode_atom(tree)]
        if not isinstance(tree, Mapping) or set(tree) != {"left", "right"}:
            raise ValueError("tree nodes must contain exactly left and right")
        return [
            self.LPAREN,
            *self.encode_tree(tree["left"]),
            self.COMMA,
            *self.encode_tree(tree["right"]),
            self.RPAREN,
        ]

    def vocab_size(self, max_atom: int) -> int:
        return self.encode_atom(max_atom) + 1


@dataclass(frozen=True)
class ActionCodec:
    max_inventory: int

    PAD: ClassVar[int] = 0
    BOS: ClassVar[int] = 1
    PAIR_OFFSET: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if isinstance(self.max_inventory, bool) or self.max_inventory < 1:
            raise ValueError("max_inventory must be positive")

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (left, right)
            for left in range(self.max_inventory)
            for right in range(left + 1, self.max_inventory)
        )

    @property
    def pad_token(self) -> int:
        return self.PAD

    @property
    def bos_token(self) -> int:
        return self.BOS

    @property
    def eos_token(self) -> int:
        return self.PAIR_OFFSET + len(self.pairs)

    @property
    def vocab_size(self) -> int:
        return self.eos_token + 1

    def encode(self, action: MergeAction) -> int:
        pair = (action.left_index, action.right_index)
        try:
            return self.PAIR_OFFSET + self.pairs.index(pair)
        except ValueError as error:
            raise ValueError("action is outside this codec's inventory range") from error

    def decode(self, token: int) -> MergeAction | None:
        if token == self.eos_token:
            return None
        pair_index = token - self.PAIR_OFFSET
        if pair_index < 0 or pair_index >= len(self.pairs):
            raise ValueError("token is not an action or EOS")
        left, right = self.pairs[pair_index]
        return MergeAction(left, right)


def encode_source(row: Mapping[str, object], token_codec: TokenCodec | None = None) -> list[int]:
    """Serialize exactly row['inventory'] and row['goal']; audit metadata is ignored."""
    codec = token_codec if token_codec is not None else TokenCodec()
    inventory = row["inventory"]
    if not isinstance(inventory, Sequence) or isinstance(inventory, (str, bytes)):
        raise ValueError("inventory must be a sequence of trees")
    tokens = [codec.BOS, codec.INVENTORY]
    for index, tree in enumerate(inventory):
        if index:
            tokens.append(codec.COMMA)
        tokens.extend(codec.encode_tree(tree))
    tokens.extend((codec.GOAL, *codec.encode_tree(row["goal"]), codec.EOS))
    return tokens


def _as_action(value: object) -> MergeAction:
    if isinstance(value, MergeAction):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("actions must be MergeAction values or index pairs")
    return MergeAction(int(value[0]), int(value[1]))


def encode_actions(row: Mapping[str, object], action_codec: ActionCodec) -> list[int]:
    actions = row["actions"]
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        raise ValueError("actions must be a sequence")
    return [*(action_codec.encode(_as_action(action)) for action in actions), action_codec.eos_token]


def _pad_sequences(sequences: Sequence[Sequence[int]], pad_token: int) -> torch.Tensor:
    if not sequences:
        raise ValueError("cannot collate an empty batch")
    width = max(len(sequence) for sequence in sequences)
    result = torch.full((len(sequences), width), pad_token, dtype=torch.long)
    for row_index, sequence in enumerate(sequences):
        result[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return result


def collate_batch(
    rows: Sequence[Mapping[str, object]], token_codec: TokenCodec, action_codec: ActionCodec
) -> dict[str, torch.Tensor]:
    src = _pad_sequences([encode_source(row, token_codec) for row in rows], token_codec.PAD)
    targets = _pad_sequences([encode_actions(row, action_codec) for row in rows], action_codec.pad_token)
    decoder_sequences = [
        [action_codec.bos_token, *target[:-1]]
        for target in (encode_actions(row, action_codec) for row in rows)
    ]
    decoder_input = _pad_sequences(decoder_sequences, action_codec.pad_token)
    return {
        "src": src,
        "src_padding_mask": src.eq(token_codec.PAD),
        "decoder_input": decoder_input,
        "decoder_padding_mask": decoder_input.eq(action_codec.pad_token),
        "targets": targets,
        "target_padding_mask": targets.eq(action_codec.pad_token),
    }


def legal_action_mask(state: EnvState, action_codec: ActionCodec) -> torch.BoolTensor:
    mask = torch.zeros(action_codec.vocab_size, dtype=torch.bool)
    for action in (
        MergeAction(left, right)
        for left in range(len(state.inventory))
        for right in range(left + 1, len(state.inventory))
    ):
        if action.left_index < action_codec.max_inventory and action.right_index < action_codec.max_inventory:
            mask[action_codec.encode(action)] = True
    if len(state.inventory) == 1:
        mask[action_codec.eos_token] = True
    return mask
