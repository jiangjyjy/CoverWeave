"""Deterministic rule-governed state transitions for the CompForge v1 environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
from itertools import count, permutations


class RuleKind(str, Enum):
    TRANSFORM = "transform"
    BIND = "bind"
    SYNCHRONIZE = "synchronize"
    SYNTHESIZE = "synthesize"
    DECOMPOSE = "decompose"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    kind: RuleKind
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cost: int

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError("rule_id must be a non-empty string")
        if not isinstance(self.kind, RuleKind):
            raise ValueError("kind must be a RuleKind")
        if not all(isinstance(state, str) and state for state in (*self.inputs, *self.outputs)):
            raise ValueError("rule states must be non-empty strings")
        if isinstance(self.cost, bool) or not isinstance(self.cost, int) or self.cost < 1:
            raise ValueError("rule cost must be a positive integer")
        input_count = len(self.inputs)
        output_count = len(self.outputs)
        valid = {
            RuleKind.TRANSFORM: input_count == 1 and output_count == 1,
            RuleKind.BIND: input_count == 2 and output_count == 1,
            RuleKind.SYNCHRONIZE: input_count == 2 and output_count == 2,
            RuleKind.SYNTHESIZE: input_count in {2, 3} and output_count == 1,
            RuleKind.DECOMPOSE: input_count == 1 and output_count in {2, 3},
        }
        if not valid[self.kind]:
            raise ValueError(f"invalid arity for {self.kind.value}")


@dataclass(frozen=True)
class ObjectInstance:
    instance_id: int
    state: str
    atoms: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.instance_id, bool) or not isinstance(self.instance_id, int) or self.instance_id < 0:
            raise ValueError("instance_id must be a non-negative integer")
        if not isinstance(self.state, str) or not self.state:
            raise ValueError("state must be a non-empty string")
        if not self.atoms or any(isinstance(atom, bool) or not isinstance(atom, int) or atom < 0 for atom in self.atoms):
            raise ValueError("atoms must be a non-empty tuple of non-negative integers")


@dataclass(frozen=True)
class RuleAction:
    rule_id: str
    operand_slots: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError("rule_id must be a non-empty string")
        if any(isinstance(slot, bool) or not isinstance(slot, int) or slot < 0 for slot in self.operand_slots):
            raise ValueError("operand_slots must be non-negative integers")


@dataclass(frozen=True)
class CompForgeState:
    inventory: tuple[ObjectInstance, ...]
    next_instance_id: int
    total_cost: int = 0

    def __post_init__(self) -> None:
        identifiers = [item.instance_id for item in self.inventory]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("inventory contains duplicate instance IDs")
        if isinstance(self.next_instance_id, bool) or not isinstance(self.next_instance_id, int) or self.next_instance_id < 0:
            raise ValueError("next_instance_id must be a non-negative integer")
        if identifiers and self.next_instance_id <= max(identifiers):
            raise ValueError("next_instance_id must exceed all inventory IDs")
        if isinstance(self.total_cost, bool) or not isinstance(self.total_cost, int) or self.total_cost < 0:
            raise ValueError("total_cost must be a non-negative integer")


@dataclass(frozen=True)
class CompForgeCheckResult:
    valid: bool
    steps: int
    total_cost: int
    reason: str
    final_state: CompForgeState


@dataclass(frozen=True)
class OracleResult:
    success: bool
    actions: tuple[RuleAction, ...]
    total_cost: int
    expanded_states: int
    reason: str


class CompForgeEnv:
    def __init__(self, rules: tuple[Rule, ...]) -> None:
        if not rules:
            raise ValueError("CompForgeEnv requires at least one rule")
        if len({rule.rule_id for rule in rules}) != len(rules):
            raise ValueError("rule IDs must be unique")
        self.rules = tuple(rules)
        self._by_id = {rule.rule_id: rule for rule in self.rules}

    def initial_state(self, inventory: tuple[ObjectInstance, ...]) -> CompForgeState:
        inventory = tuple(inventory)
        next_id = max((item.instance_id for item in inventory), default=-1) + 1
        return CompForgeState(inventory, next_id)

    def valid_actions(self, state: CompForgeState) -> tuple[RuleAction, ...]:
        actions: list[RuleAction] = []
        for rule in self.rules:
            for slots in permutations(range(len(state.inventory)), len(rule.inputs)):
                if tuple(state.inventory[slot].state for slot in slots) == rule.inputs:
                    actions.append(RuleAction(rule.rule_id, slots))
        return tuple(actions)

    def apply(self, state: CompForgeState, action: RuleAction) -> CompForgeState:
        if action not in self.valid_actions(state):
            raise ValueError("action is not legal for this state")
        rule = self._by_id[action.rule_id]
        operands = tuple(state.inventory[slot] for slot in action.operand_slots)
        atoms = tuple(atom for operand in operands for atom in operand.atoms)
        outputs = tuple(
            ObjectInstance(state.next_instance_id + index, output_state, atoms)
            for index, output_state in enumerate(rule.outputs)
        )
        consumed = set(action.operand_slots)
        insertion = min(consumed)
        remaining = [item for index, item in enumerate(state.inventory) if index not in consumed]
        next_inventory = tuple((*remaining[:insertion], *outputs, *remaining[insertion:]))
        return CompForgeState(
            inventory=next_inventory,
            next_instance_id=state.next_instance_id + len(outputs),
            total_cost=state.total_cost + rule.cost,
        )


def check_compforge_trace(
    env: CompForgeEnv,
    inventory: tuple[ObjectInstance, ...],
    goal_state: str,
    actions: tuple[RuleAction, ...],
) -> CompForgeCheckResult:
    state = env.initial_state(inventory)
    for step, action in enumerate(actions):
        try:
            state = env.apply(state, action)
        except ValueError as error:
            return CompForgeCheckResult(False, step, state.total_cost, str(error), state)
    if not any(item.state == goal_state for item in state.inventory):
        return CompForgeCheckResult(False, len(actions), state.total_cost, "goal state is absent", state)
    return CompForgeCheckResult(True, len(actions), state.total_cost, "goal state is present", state)


def _state_key(state: CompForgeState) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((item.state, item.atoms) for item in state.inventory)


def uniform_cost_oracle(
    env: CompForgeEnv,
    inventory: tuple[ObjectInstance, ...],
    goal_state: str,
    *,
    max_expansions: int,
) -> OracleResult:
    if isinstance(max_expansions, bool) or not isinstance(max_expansions, int) or max_expansions < 0:
        raise ValueError("max_expansions must be a non-negative integer")
    initial = env.initial_state(inventory)
    queue: list[tuple[int, int, CompForgeState, tuple[RuleAction, ...]]] = []
    tie_breaker = count()
    heappush(queue, (0, next(tie_breaker), initial, tuple()))
    best_cost = {_state_key(initial): 0}
    expanded = 0

    while queue:
        cost, _, state, actions = heappop(queue)
        if cost != best_cost.get(_state_key(state)):
            continue
        if any(item.state == goal_state for item in state.inventory):
            return OracleResult(True, actions, cost, expanded, "goal reached")
        if expanded >= max_expansions:
            return OracleResult(False, tuple(), cost, expanded, "expansion limit reached")
        expanded += 1
        for action in env.valid_actions(state):
            next_state = env.apply(state, action)
            next_cost = cost + env._by_id[action.rule_id].cost
            key = _state_key(next_state)
            if next_cost < best_cost.get(key, float("inf")):
                best_cost[key] = next_cost
                heappush(queue, (next_cost, next(tie_breaker), next_state, (*actions, action)))
    return OracleResult(False, tuple(), 0, expanded, "goal is unreachable")
