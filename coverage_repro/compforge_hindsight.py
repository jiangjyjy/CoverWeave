"""Coverage and causal-trace primitives for deterministic CompForge hindsight data."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import combinations
from random import Random
from pathlib import Path

from .compforge import CompForgeEnv, CompForgeState, ObjectInstance, Rule, RuleAction, RuleKind, check_compforge_trace, uniform_cost_oracle
from .graph import CoverageGraph
from .io import sha256_file, write_json, write_jsonl


@dataclass(frozen=True)
class RecordedAction:
    rule_id: str
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    input_atoms: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("recorded rule_id must be non-empty")
        if not self.input_ids or not self.output_ids:
            raise ValueError("recorded actions require inputs and outputs")
        if len(self.input_ids) != len(self.input_atoms):
            raise ValueError("input_ids and input_atoms must have the same length")


@dataclass(frozen=True)
class CausalTrace:
    initial_inventory: tuple[ObjectInstance, ...]
    goal_object: ObjectInstance
    actions: tuple[RecordedAction, ...]


def connected_coverage_graph(atom_count: int, edge_count: int, seed: int) -> CoverageGraph:
    if atom_count < 2:
        raise ValueError("atom_count must be at least 2")
    minimum = atom_count - 1
    maximum = atom_count * (atom_count - 1) // 2
    if edge_count < minimum:
        raise ValueError("edge_count must be at least atom_count - 1 for a connected graph")
    if edge_count > maximum:
        raise ValueError("edge_count exceeds the complete graph limit")
    rng = Random(seed)
    prufer = [rng.randrange(atom_count) for _ in range(atom_count - 2)]
    degrees = [1] * atom_count
    for node in prufer:
        degrees[node] += 1
    leaves = [node for node, degree in enumerate(degrees) if degree == 1]
    heapify(leaves)
    edges: set[tuple[int, int]] = set()
    for parent in prufer:
        leaf = heappop(leaves)
        edges.add(tuple(sorted((leaf, parent))))
        degrees[leaf] -= 1
        degrees[parent] -= 1
        if degrees[parent] == 1:
            heappush(leaves, parent)
    first, second = sorted(leaves)
    edges.add((first, second))
    remaining = sorted(set(combinations(range(atom_count), 2)) - edges)
    edges.update(rng.sample(remaining, edge_count - len(edges)))
    return CoverageGraph(atom_count, edges)


def realized_pairs(actions: tuple[RecordedAction, ...]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for action in actions:
        for left_index, left_atoms in enumerate(action.input_atoms):
            for right_atoms in action.input_atoms[left_index + 1 :]:
                for left in left_atoms:
                    for right in right_atoms:
                        if left != right:
                            pairs.add(tuple(sorted((left, right))))
    return pairs


def prune_causal_trace(
    initial_inventory: tuple[ObjectInstance, ...],
    goal_object: ObjectInstance,
    actions: tuple[RecordedAction, ...],
) -> CausalTrace:
    producers: dict[int, RecordedAction] = {}
    for action in actions:
        for output_id in action.output_ids:
            if output_id in producers:
                raise ValueError("an object instance may have only one producer")
            producers[output_id] = action
    if goal_object.instance_id not in producers:
        raise ValueError("goal object has no producer")
    required = {goal_object.instance_id}
    retained_reversed: list[RecordedAction] = []
    for action in reversed(actions):
        if set(action.output_ids) & required:
            retained_reversed.append(action)
            required.update(action.input_ids)
    retained = tuple(reversed(retained_reversed))
    produced = {output_id for action in retained for output_id in action.output_ids}
    source_ids = required - produced
    sources = tuple(item for item in initial_inventory if item.instance_id in source_ids)
    if source_ids != {item.instance_id for item in sources}:
        raise ValueError("causal sources are missing from initial inventory")
    if any(item.state == goal_object.state for item in sources):
        raise ValueError("goal state already occurs in causal initial inventory")
    return CausalTrace(sources, goal_object, retained)


def _slot_for_instance(state: CompForgeState, instance_id: int) -> int:
    for index, item in enumerate(state.inventory):
        if item.instance_id == instance_id:
            return index
    raise ValueError(f"causal instance {instance_id} is unavailable")


def compile_actions(env: CompForgeEnv, trace: CausalTrace) -> tuple[RuleAction, ...]:
    state = env.initial_state(trace.initial_inventory)
    stable_to_actual = {item.instance_id: item.instance_id for item in trace.initial_inventory}
    compiled: list[RuleAction] = []
    for record in trace.actions:
        try:
            slots = tuple(_slot_for_instance(state, stable_to_actual[item_id]) for item_id in record.input_ids)
        except KeyError as error:
            raise ValueError(f"causal instance {error.args[0]} is unavailable") from error
        action = RuleAction(record.rule_id, slots)
        before = state
        state = env.apply(state, action)
        created = tuple(item for item in state.inventory if item.instance_id >= before.next_instance_id)
        if len(created) != len(record.output_ids):
            raise ValueError("recorded output arity differs from the rule output arity")
        stable_to_actual.update(
            {stable_id: actual.instance_id for stable_id, actual in zip(record.output_ids, created)}
        )
        compiled.append(action)
    return tuple(compiled)


@dataclass(frozen=True)
class CompForgeDatasetConfig:
    atom_count: int
    edge_count: int
    trajectory_budget: int
    max_rollout_steps: int
    max_attempts: int
    seed: int

    def __post_init__(self) -> None:
        for name in ("atom_count", "edge_count", "trajectory_budget", "max_rollout_steps", "max_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.atom_count < 4:
            raise ValueError("atom_count must be at least 4")


def _sample_instance(atom_count: int, seed: int) -> tuple[CompForgeEnv, tuple[ObjectInstance, ...], dict[str, object]]:
    atoms = list(range(atom_count))
    Random(seed).shuffle(atoms)
    first, second, third, fourth = atoms[:4]
    rules = (
        Rule("transform", RuleKind.TRANSFORM, (f"atom:{first}",), (f"t:{first}",), 1),
        Rule("bind", RuleKind.BIND, (f"t:{first}", f"atom:{second}"), ("bound",), 2),
        Rule("sync", RuleKind.SYNCHRONIZE, (f"atom:{third}", f"atom:{fourth}"), ("sync-left", "sync-right"), 3),
        Rule("synth", RuleKind.SYNTHESIZE, ("bound", "sync-left", "sync-right"), ("product",), 4),
        Rule("decompose", RuleKind.DECOMPOSE, ("product",), ("fragment-left", "fragment-right"), 5),
    )
    inventory = tuple(ObjectInstance(atom, f"atom:{atom}", (atom,)) for atom in atoms)
    audit = {
        "atom_order": atoms,
        "rules": [
            {"rule_id": rule.rule_id, "kind": rule.kind.value, "inputs": rule.inputs, "outputs": rule.outputs, "cost": rule.cost}
            for rule in rules
        ],
    }
    return CompForgeEnv(rules), inventory, audit


def _record_rollout(env: CompForgeEnv, inventory: tuple[ObjectInstance, ...]) -> tuple[ObjectInstance, tuple[RecordedAction, ...]]:
    state = env.initial_state(inventory)
    records: list[RecordedAction] = []
    for rule_id in ("transform", "bind", "sync", "synth"):
        action = next(action for action in env.valid_actions(state) if action.rule_id == rule_id)
        inputs = tuple(state.inventory[slot] for slot in action.operand_slots)
        before = state
        state = env.apply(state, action)
        outputs = tuple(item for item in state.inventory if item.instance_id >= before.next_instance_id)
        records.append(RecordedAction(rule_id, tuple(item.instance_id for item in inputs), tuple(item.instance_id for item in outputs), tuple(item.atoms for item in inputs)))
    return next(item for item in state.inventory if item.state == "product"), tuple(records)


def policy_view(record: dict[str, object]) -> dict[str, object]:
    return {"inventory": record["inventory"], "goal": record["goal"]}


def build_compforge_splits(config: CompForgeDatasetConfig, output_dir: str | Path) -> dict[str, object]:
    graph = connected_coverage_graph(config.atom_count, config.edge_count, config.seed)
    env, initial, instance_audit = _sample_instance(config.atom_count, config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        rows: list[dict[str, object]] = []
        attempts = 0
        while len(rows) < config.trajectory_budget and attempts < config.max_attempts:
            attempts += 1
            goal, recorded = _record_rollout(env, initial)
            trace = prune_causal_trace(initial, goal, recorded)
            pairs = realized_pairs(trace.actions)
            if not pairs <= set(graph.edges):
                continue
            actions = compile_actions(env, trace)
            checked = check_compforge_trace(env, trace.initial_inventory, trace.goal_object.state, actions)
            if not checked.valid:
                raise AssertionError("causal CompForge trace failed checker")
            oracle = uniform_cost_oracle(env, trace.initial_inventory, trace.goal_object.state, max_expansions=1_000)
            if not oracle.success:
                raise AssertionError("CompForge oracle failed on an accepted trace")
            rows.append({
                "sample_id": f"{split}-{len(rows):06d}", "split": split,
                "inventory": [item.state for item in trace.initial_inventory], "goal": trace.goal_object.state,
                "actions": [{"rule_id": action.rule_id, "operand_slots": action.operand_slots} for action in actions],
                "checker_valid": True, "oracle_success": True, "oracle_cost": oracle.total_cost,
                "realized_pairs": sorted(pairs), "coverage_edges": sorted(graph.edges), "coverage_rho": graph.rho,
                "instance": instance_audit,
                "audit_initial_objects": [{"instance_id": item.instance_id, "state": item.state, "atoms": item.atoms} for item in trace.initial_inventory],
            })
        if len(rows) != config.trajectory_budget:
            raise ValueError("could not collect CompForge trajectories within max_attempts")
        path = output / f"{split}.jsonl"
        write_jsonl(path, rows)
        hashes[split] = sha256_file(path)
    manifest = {"config": config.__dict__, "hashes": hashes, "coverage_edges": sorted(graph.edges), "instance": instance_audit}
    write_json(output / "manifest.json", manifest)
    return manifest
