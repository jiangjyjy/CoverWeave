from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .env import CoverageEnv, EnvState, MergeAction
from .graph import CoverageGraph, Node, Tree, tree_leaves, tree_subtrees


@dataclass(frozen=True)
class BFSResult:
    success: bool
    actions: tuple[MergeAction, ...]
    checked: bool
    states_explored: int


def component_oracle(graph: CoverageGraph, goal: Tree) -> bool:
    return bfs_oracle(graph, goal).success


def bfs_oracle(graph: CoverageGraph, goal: Tree, max_states: int = 100000) -> BFSResult:
    env = CoverageEnv(graph)
    target_nodes = set(tree_subtrees(goal))
    initial = env.initial_state(goal)
    queue = deque([(initial, tuple())])
    seen = {initial.inventory}
    explored = 0
    while queue and explored < max_states:
        state, actions = queue.popleft()
        explored += 1
        if len(state.inventory) == 1:
            return BFSResult(state.inventory[0] == goal, actions, True, explored)
        for action in env.valid_actions(state):
            merged = Node(state.inventory[action.left_index], state.inventory[action.right_index])
            if merged not in target_nodes:
                continue
            next_state = env.apply(state, action)
            if next_state.inventory in seen:
                continue
            seen.add(next_state.inventory)
            queue.append((next_state, actions + (action,)))
    return BFSResult(False, tuple(), True, explored)
