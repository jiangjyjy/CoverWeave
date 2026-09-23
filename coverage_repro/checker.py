from __future__ import annotations

from dataclasses import dataclass

from .env import CoverageEnv, EnvState, MergeAction
from .graph import CoverageGraph, Tree


@dataclass(frozen=True)
class CheckResult:
    valid: bool
    final_tree: Tree | None
    steps: int
    reason: str


def check_trace(
    graph: CoverageGraph,
    goal: Tree,
    actions: list[MergeAction] | tuple[MergeAction, ...],
    inventory: tuple[Tree, ...] | list[Tree] | None = None,
) -> CheckResult:
    env = CoverageEnv(graph)
    state: EnvState = (
        env.initial_state(goal)
        if inventory is None
        else EnvState(tuple(inventory))
    )
    for step, action in enumerate(actions, start=1):
        if action.left_index >= len(state.inventory) or action.right_index >= len(state.inventory):
            return CheckResult(False, None, step - 1, "action index is outside inventory")
        state = env.apply(state, action)
    final_tree = state.inventory[0] if len(state.inventory) == 1 else None
    if final_tree != goal:
        return CheckResult(
            False,
            final_tree,
            state.steps,
            "final inventory is not the exact goal tree",
        )
    return CheckResult(True, final_tree, state.steps, "exact goal replay verified")
