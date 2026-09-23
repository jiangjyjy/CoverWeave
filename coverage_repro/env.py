from __future__ import annotations

from dataclasses import dataclass

from .graph import CoverageGraph, Node, Tree, tree_leaves


@dataclass(frozen=True)
class MergeAction:
    left_index: int
    right_index: int

    def __post_init__(self) -> None:
        if self.left_index < 0 or self.right_index < 0:
            raise ValueError("action indices must be non-negative")
        if self.left_index == self.right_index:
            raise ValueError("MERGE requires two distinct inventory items")


@dataclass(frozen=True)
class EnvState:
    inventory: tuple[Tree, ...]
    steps: int = 0


class CoverageEnv:
    def __init__(self, graph: CoverageGraph):
        self.graph = graph

    def initial_state(self, goal: Tree | None = None) -> EnvState:
        if goal is not None:
            return EnvState(tuple(tree_leaves(goal)))
        return EnvState(tuple(range(self.graph.n)))

    def can_merge(self, left: Tree, right: Tree) -> bool:
        del left, right
        return True

    def valid_actions(self, state: EnvState) -> tuple[MergeAction, ...]:
        return tuple(
            MergeAction(left_index, right_index)
            for left_index in range(len(state.inventory))
            for right_index in range(left_index + 1, len(state.inventory))
        )

    def apply(self, state: EnvState, action: MergeAction) -> EnvState:
        if action.right_index >= len(state.inventory):
            raise ValueError("action index is outside the inventory")
        inventory = list(state.inventory)
        merged = Node(inventory[action.left_index], inventory[action.right_index])
        inventory[action.left_index] = merged
        del inventory[action.right_index]
        return EnvState(tuple(inventory), state.steps + 1)
