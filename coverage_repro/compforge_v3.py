"""Goal-dependent, branching CompForge v3 task generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Mapping, Sequence

import torch

from .compforge import (
    CompForgeEnv,
    CompForgeState,
    ObjectInstance,
    Rule,
    RuleAction,
    RuleKind,
    check_compforge_trace,
    uniform_cost_oracle,
)
from .compforge_encoding import CompForgeActionCodec
from .compforge_hindsight import connected_coverage_graph
from .io import sha256_file, write_json, write_jsonl


@dataclass(frozen=True)
class CompForgeV3Config:
    atom_count: int = 6
    train_edge_count: int = 12
    branch_depth: int = 2
    samples_per_split: int = 4
    oracle_max_expansions: int = 1_000
    seed: int = 73

    def __post_init__(self) -> None:
        complete_edges = self.atom_count * (self.atom_count - 1) // 2
        if self.atom_count < 4:
            raise ValueError("atom_count must be at least four")
        if not 1 <= self.train_edge_count < complete_edges:
            raise ValueError("train_edge_count must leave a held-out edge")
        if self.branch_depth < 1 or self.oracle_max_expansions < 1:
            raise ValueError("branch_depth and oracle_max_expansions must be positive")
        if self.samples_per_split < 1:
            raise ValueError("samples_per_split must be positive")


def _action_payload(action: RuleAction) -> dict[str, object]:
    return {"rule_id": action.rule_id, "operand_slots": list(action.operand_slots)}


def _rule_payload(rule: Rule) -> dict[str, object]:
    return {
        "rule_id": rule.rule_id,
        "kind": rule.kind.value,
        "inputs": list(rule.inputs),
        "outputs": list(rule.outputs),
        "cost": rule.cost,
    }


def _pair_for_seed(config: CompForgeV3Config, seed: int) -> tuple[int, int]:
    selected = Random(seed).sample(range(config.atom_count), 2)
    return tuple(sorted(selected))


def _make_rules(left: str, right: str, depth: int) -> tuple[Rule, ...]:
    rules: list[Rule] = [
        Rule("start:red", RuleKind.BIND, (left, right), ("red:0",), 1),
        Rule("start:blue", RuleKind.BIND, (left, right), ("blue:0",), 1),
        Rule("harm:initial", RuleKind.BIND, (left, right), ("waste",), 1),
        Rule("distract:left", RuleKind.TRANSFORM, ("idle",), ("idle:left",), 1),
        Rule("distract:right", RuleKind.TRANSFORM, ("idle",), ("idle:right",), 1),
    ]
    for branch in ("red", "blue"):
        for index in range(depth):
            output = f"{branch}:{index + 1}" if index + 1 < depth else f"goal:{branch}"
            suffix = f"step:{index}" if index + 1 < depth else "finish"
            rules.extend(
                (
                    Rule(f"{branch}:{suffix}", RuleKind.TRANSFORM, (f"{branch}:{index}",), (output,), 1),
                    Rule(f"harm:{branch}:{index}", RuleKind.TRANSFORM, (f"{branch}:{index}",), ("waste",), 1),
                )
            )
    return tuple(rules)


def _inventory_for_pair(pair: tuple[int, int]) -> tuple[ObjectInstance, ...]:
    left, right = pair
    idle_atom = max(pair) + 1
    return (
        ObjectInstance(0, f"atom:{left}", (left,)),
        ObjectInstance(1, f"atom:{right}", (right,)),
        ObjectInstance(2, "idle", (idle_atom,)),
    )


def build_branching_instance(
    config: CompForgeV3Config,
    seed: int,
    goal_variant: int = 0,
    *,
    pair: tuple[int, int] | None = None,
    branch_depth: int | None = None,
) -> dict[str, object]:
    """Build one deterministic task with useful, distractor, and harmful choices."""
    if goal_variant not in (0, 1):
        raise ValueError("goal_variant must be zero or one")
    chosen_pair = _pair_for_seed(config, seed) if pair is None else tuple(sorted(pair))
    if len(chosen_pair) != 2 or chosen_pair[0] == chosen_pair[1]:
        raise ValueError("pair must contain two distinct atoms")
    if any(atom < 0 or atom >= config.atom_count for atom in chosen_pair):
        raise ValueError("pair atom is outside config.atom_count")
    depth = config.branch_depth if branch_depth is None else branch_depth
    if depth < 1:
        raise ValueError("branch_depth must be positive")
    inventory = _inventory_for_pair(chosen_pair)
    env = CompForgeEnv(_make_rules(inventory[0].state, inventory[1].state, depth))
    goal = ("goal:red", "goal:blue")[goal_variant]
    oracle = uniform_cost_oracle(env, inventory, goal, max_expansions=config.oracle_max_expansions)
    if not oracle.success:
        raise AssertionError("branching v3 oracle could not reach its goal")
    checked = check_compforge_trace(env, inventory, goal, oracle.actions)
    if not checked.valid:
        raise AssertionError("branching v3 oracle failed checker replay")
    state = env.initial_state(inventory)
    legal_counts: list[int] = []
    for action in oracle.actions:
        legal_counts.append(len(env.valid_actions(state)))
        state = env.apply(state, action)
    if min(legal_counts, default=0) < 3:
        raise AssertionError("branching v3 task has a forced oracle state")
    signature = (
        tuple(item.state for item in inventory),
        goal,
        tuple(action.rule_id for action in oracle.actions),
    )
    return {
        "inventory": [item.state for item in inventory],
        "goal": goal,
        "actions": [_action_payload(action) for action in oracle.actions],
        "oracle_actions": [_action_payload(action) for action in oracle.actions],
        "oracle_cost": oracle.total_cost,
        "checker_valid": checked.valid,
        "legal_action_counts": legal_counts,
        "depth": len(oracle.actions),
        "task_signature": repr(signature),
        "realized_pairs": [list(chosen_pair)],
        "instance": {"rules": [_rule_payload(rule) for rule in env.rules]},
        "audit_initial_objects": [
            {"instance_id": item.instance_id, "state": item.state, "atoms": list(item.atoms)}
            for item in inventory
        ],
    }


def environment_from_v3_row(row: Mapping[str, object]) -> CompForgeEnv:
    instance = row.get("instance")
    if not isinstance(instance, Mapping):
        raise ValueError("v3 row instance is invalid")
    rule_payloads = instance.get("rules")
    if not isinstance(rule_payloads, Sequence) or isinstance(rule_payloads, (str, bytes)):
        raise ValueError("v3 row rules are invalid")
    rules: list[Rule] = []
    for payload in rule_payloads:
        if not isinstance(payload, Mapping):
            raise ValueError("v3 rule payload is invalid")
        rules.append(
            Rule(
                str(payload["rule_id"]),
                RuleKind(str(payload["kind"])),
                tuple(str(value) for value in payload["inputs"]),
                tuple(str(value) for value in payload["outputs"]),
                int(payload["cost"]),
            )
        )
    return CompForgeEnv(tuple(rules))


def inventory_from_v3_row(row: Mapping[str, object]) -> tuple[ObjectInstance, ...]:
    payloads = row.get("audit_initial_objects")
    if not isinstance(payloads, Sequence) or isinstance(payloads, (str, bytes)):
        raise ValueError("v3 initial object audit is invalid")
    inventory: list[ObjectInstance] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise ValueError("v3 object payload is invalid")
        inventory.append(
            ObjectInstance(
                int(payload["instance_id"]),
                str(payload["state"]),
                tuple(int(atom) for atom in payload["atoms"]),
            )
        )
    return tuple(inventory)


def goal_blind_legal_action_mask(
    env: CompForgeEnv,
    state: CompForgeState,
    codec: CompForgeActionCodec,
) -> torch.BoolTensor:
    mask = torch.zeros(codec.vocab_size, dtype=torch.bool)
    for action in env.valid_actions(state):
        mask[codec.encode(action)] = True
    if any(item.state.startswith("goal:") for item in state.inventory):
        mask[codec.eos_token] = True
    return mask


def classify_legal_actions(
    env: CompForgeEnv,
    state: CompForgeState,
    goal_state: object,
    *,
    max_expansions: int,
) -> dict[str, tuple[RuleAction, ...]]:
    goal = str(goal_state)
    baseline = uniform_cost_oracle(env, state.inventory, goal, max_expansions=max_expansions)
    if not baseline.success:
        raise ValueError("goal is unreachable from classification state")
    costs = {rule.rule_id: rule.cost for rule in env.rules}
    buckets: dict[str, list[RuleAction]] = {"useful": [], "distractor": [], "harmful": []}
    for action in env.valid_actions(state):
        next_state = env.apply(state, action)
        continuation = uniform_cost_oracle(
            env,
            next_state.inventory,
            goal,
            max_expansions=max_expansions,
        )
        if not continuation.success:
            buckets["harmful"].append(action)
        elif costs[action.rule_id] + continuation.total_cost == baseline.total_cost:
            buckets["useful"].append(action)
        else:
            buckets["distractor"].append(action)
    return {name: tuple(actions) for name, actions in buckets.items()}


def _actions_from_payload(payloads: object) -> tuple[RuleAction, ...]:
    if not isinstance(payloads, Sequence) or isinstance(payloads, (str, bytes)):
        raise ValueError("v3 actions are invalid")
    actions: list[RuleAction] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise ValueError("v3 action payload is invalid")
        actions.append(
            RuleAction(
                str(payload["rule_id"]),
                tuple(int(slot) for slot in payload["operand_slots"]),
            )
        )
    return tuple(actions)


def certify_branching_task(row: Mapping[str, object]) -> dict[str, object]:
    env = environment_from_v3_row(row)
    inventory = inventory_from_v3_row(row)
    goal = str(row["goal"])
    actions = _actions_from_payload(row.get("oracle_actions"))
    oracle = uniform_cost_oracle(env, inventory, goal, max_expansions=1_000)
    checked = check_compforge_trace(env, inventory, goal, actions)
    state = env.initial_state(inventory)
    legal_counts: list[int] = []
    classification_counts: list[dict[str, int]] = []
    decisions_valid = True
    for action in actions:
        legal = env.valid_actions(state)
        classes = classify_legal_actions(env, state, goal, max_expansions=1_000)
        legal_counts.append(len(legal))
        classification_counts.append({name: len(values) for name, values in classes.items()})
        decisions_valid &= len(legal) >= 3
        decisions_valid &= all(classes[name] for name in ("useful", "distractor", "harmful"))
        decisions_valid &= action in classes["useful"]
        state = env.apply(state, action)
    certified = oracle.success and checked.valid and oracle.actions == actions and decisions_valid
    return {
        "certified": certified,
        "oracle_cost": oracle.total_cost,
        "legal_action_counts": legal_counts,
        "classification_counts": classification_counts,
    }


def _v3_split_rows(
    config: CompForgeV3Config,
    split: str,
    pairs: tuple[tuple[int, int], ...],
    offset: int,
    depth: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(config.samples_per_split):
        row = build_branching_instance(
            config,
            config.seed + offset + index,
            (index + offset // len(pairs)) % 2,
            pair=pairs[(index + offset) % len(pairs)],
            branch_depth=depth,
        )
        row.update({"sample_id": f"{split}-{index:06d}", "split": split})
        rows.append(row)
    return rows


def build_compforge_v3_splits(
    config: CompForgeV3Config,
    output_dir: str | Path,
) -> dict[str, object]:
    graph = connected_coverage_graph(config.atom_count, config.train_edge_count, config.seed)
    train_edges = tuple(sorted(graph.edges))
    all_edges = {
        (left, right)
        for left in range(config.atom_count)
        for right in range(left + 1, config.atom_count)
    }
    held_edges = tuple(sorted(all_edges - set(train_edges)))
    rows = {
        "train": _v3_split_rows(config, "train", train_edges, 0, config.branch_depth),
        "valid": _v3_split_rows(config, "valid", train_edges, config.samples_per_split, config.branch_depth),
        "held_out_same_depth": _v3_split_rows(config, "held_out_same_depth", held_edges, 0, config.branch_depth),
        "held_out_deep": _v3_split_rows(
            config,
            "held_out_deep",
            held_edges,
            config.samples_per_split,
            config.branch_depth + 2,
        ),
    }
    signatures = [str(row["task_signature"]) for values in rows.values() for row in values]
    if len(signatures) != len(set(signatures)):
        raise ValueError("v3 manifest has duplicate public signatures")
    if not all(certify_branching_task(row)["certified"] for values in rows.values() for row in values):
        raise ValueError("v3 dataset contains an uncertified task")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for split, values in rows.items():
        path = output / f"{split}.jsonl"
        write_jsonl(path, values)
        hashes[split] = sha256_file(path)
    summary = {
        "signature_overlap_count": 0,
        "minimum_decision_branching": min(
            count
            for values in rows.values()
            for row in values
            for count in row["legal_action_counts"]
        ),
        "oracle_exact": 1.0,
    }
    manifest = {
        "config": config.__dict__,
        "splits": list(rows),
        "hashes": hashes,
        "train_coverage_edges": [list(edge) for edge in train_edges],
        "held_out_edges": [list(edge) for edge in held_edges],
        "summary": summary,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_compforge_v3_manifest(
    manifest: Mapping[str, object],
    rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    train_edges = {tuple(edge) for edge in manifest["train_coverage_edges"]}
    held_edges = {tuple(edge) for edge in manifest["held_out_edges"]}
    all_rows = [row for values in rows.values() for row in values]
    signatures = [str(row["task_signature"]) for row in all_rows]
    signatures_unique = len(signatures) == len(set(signatures))
    all_certified = all(certify_branching_task(row)["certified"] for row in all_rows)
    train_membership = all(
        set(map(tuple, row["realized_pairs"])) <= train_edges
        for split in ("train", "valid")
        for row in rows[split]
    )
    held_membership = all(
        bool(set(map(tuple, row["realized_pairs"])) & held_edges)
        for split in ("held_out_same_depth", "held_out_deep")
        for row in rows[split]
    )
    return {
        "valid": signatures_unique and all_certified and train_membership and held_membership,
        "signature_overlap_count": len(signatures) - len(set(signatures)),
    }
