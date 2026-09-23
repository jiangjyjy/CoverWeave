from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .baselines import bfs_oracle
from .env import CoverageEnv, EnvState, MergeAction
from .graph import CoverageGraph, Tree


@dataclass(frozen=True)
class Trace:
    actions: tuple[MergeAction, ...]
    success: bool
    budget: int
    attempts: int
    final_state: EnvState


def generate_trace(
    graph: CoverageGraph,
    goal: Tree,
    policy: str = "random",
    seed: int = 0,
    budget: int | None = None,
) -> Trace:
    action_budget = budget if budget is not None else max(1, 2 * graph.n)
    if action_budget < 1:
        raise ValueError("budget must be positive")
    if policy == "oracle":
        result = bfs_oracle(graph, goal)
        env = CoverageEnv(graph)
        state = env.initial_state(goal)
        if result.success:
            for action in result.actions:
                state = env.apply(state, action)
        return Trace(result.actions, result.success, action_budget, len(result.actions), state)
    if policy != "random":
        raise ValueError("policy must be random or oracle")
    env = CoverageEnv(graph)
    state = env.initial_state(goal)
    rng = Random(seed)
    actions: list[MergeAction] = []
    while len(actions) < action_budget:
        valid = env.valid_actions(state)
        if not valid:
            break
        action = rng.choice(valid)
        actions.append(action)
        state = env.apply(state, action)
    return Trace(
        tuple(actions),
        len(state.inventory) == 1 and state.inventory[0] == goal,
        action_budget,
        len(actions),
        state,
    )
