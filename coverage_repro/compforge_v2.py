"""Diverse, coverage-controlled CompForge v2 data generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Mapping

from .compforge import check_compforge_trace, uniform_cost_oracle
from .compforge_hindsight import (_record_rollout, _sample_instance, compile_actions,
    connected_coverage_graph, prune_causal_trace, realized_pairs)
from .io import sha256_file, write_json, write_jsonl

@dataclass(frozen=True)
class CompForgeV2Config:
    atom_count: int = 6
    train_edge_count: int = 12
    samples_per_split: int = 16
    train_max_actions: int = 4
    deep_min_actions: int = 5
    max_attempts: int = 20000
    seed: int = 73

    def __post_init__(self):
        complete = self.atom_count * (self.atom_count - 1) // 2
        if self.atom_count < 5 or not (self.atom_count - 1 <= self.train_edge_count < complete):
            raise ValueError("train_edge_count must leave at least one held-out edge")
        if min(self.samples_per_split, self.train_max_actions, self.deep_min_actions, self.max_attempts) < 1:
            raise ValueError("v2 counts must be positive")
        if self.deep_min_actions <= self.train_max_actions:
            raise ValueError("deep_min_actions must exceed train_max_actions")

def _record(config: CompForgeV2Config, seed: int, depth: int) -> dict[str, object]:
    env, initial, audit = _sample_instance(config.atom_count, seed)
    goal, recorded = _record_rollout(env, initial)
    if depth == 5:
        state = env.initial_state(initial)
        for action in compile_actions(env, prune_causal_trace(initial, goal, recorded)):
            state = env.apply(state, action)
        action = next(value for value in env.valid_actions(state) if value.rule_id == "decompose")
        inputs = tuple(state.inventory[slot] for slot in action.operand_slots)
        before = state
        state = env.apply(state, action)
        outputs = tuple(item for item in state.inventory if item.instance_id >= before.next_instance_id)
        from .compforge_hindsight import RecordedAction
        recorded = (*recorded, RecordedAction("decompose", tuple(i.instance_id for i in inputs), tuple(i.instance_id for i in outputs), tuple(i.atoms for i in inputs)))
        goal = outputs[0]
    prefix = {1: 1, 2: 2, 4: 4, 5: 5}[depth]
    if prefix < len(recorded):
        last = recorded[prefix - 1]
        rule = next(rule for rule in env.rules if rule.rule_id == last.rule_id)
        from .compforge import ObjectInstance
        goal = ObjectInstance(last.output_ids[0], rule.outputs[0], tuple(atom for atoms in last.input_atoms for atom in atoms))
    trace = prune_causal_trace(initial, goal, tuple(recorded[:prefix]))
    actions = compile_actions(env, trace)
    checked = check_compforge_trace(env, trace.initial_inventory, trace.goal_object.state, actions)
    if not checked.valid:
        raise AssertionError("v2 trace failed checker")
    pairs = realized_pairs(trace.actions)
    signature = (tuple(item.state for item in trace.initial_inventory), trace.goal_object.state, tuple(a.rule_id for a in actions), tuple(sorted(pairs)))
    return {"inventory": [item.state for item in trace.initial_inventory], "goal": trace.goal_object.state,
        "actions": [{"rule_id": a.rule_id, "operand_slots": a.operand_slots} for a in actions],
        "checker_valid": True, "realized_pairs": sorted(pairs), "depth": len(actions), "task_signature": repr(signature),
        "instance": audit, "audit_initial_objects": [{"instance_id": i.instance_id, "state": i.state, "atoms": i.atoms} for i in trace.initial_inventory]}

def _summary(rows: list[dict[str, object]], train_edges: set[tuple[int,int]], held: set[tuple[int,int]]) -> dict[str, object]:
    depths = [int(row["depth"]) for row in rows]
    pairs = {tuple(pair) for row in rows for pair in row["realized_pairs"]}
    return {"samples": len(rows), "unique_task_signatures": len({row["task_signature"] for row in rows}),
        "min_depth": min(depths), "max_depth": max(depths), "seen_pair_count": len(pairs & train_edges), "held_out_pair_count": len(pairs & held)}

def build_compforge_v2_splits(config: CompForgeV2Config, output_dir: str | Path) -> dict[str, object]:
    graph = connected_coverage_graph(config.atom_count, config.train_edge_count, config.seed)
    train_edges = set(graph.edges)
    all_edges = {(a,b) for a in range(config.atom_count) for b in range(a + 1, config.atom_count)}
    held = all_edges - train_edges
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = {name: [] for name in ("train", "valid", "held_out_same_depth", "held_out_deep")}
    signatures: set[str] = set(); rng = Random(config.seed)
    requirements = {"train": (False, 1, config.train_max_actions), "valid": (False, 1, config.train_max_actions),
        "held_out_same_depth": (True, 1, config.train_max_actions), "held_out_deep": (True, config.deep_min_actions, 5)}
    for split, (needs_held, low, high) in requirements.items():
        attempts = 0
        while len(rows[split]) < config.samples_per_split and attempts < config.max_attempts:
            attempts += 1; depth = rng.choice([value for value in (1,2,4,5) if low <= value <= high])
            row = _record(config, rng.randrange(1 << 30), depth); pairs = {tuple(pair) for pair in row["realized_pairs"]}
            if (needs_held and not (pairs & held)) or (not needs_held and not pairs <= train_edges) or row["task_signature"] in signatures:
                continue
            row["sample_id"] = f"{split}-{len(rows[split]):06d}"; row["split"] = split; rows[split].append(row); signatures.add(row["task_signature"])
        if len(rows[split]) != config.samples_per_split:
            raise ValueError(f"could not build {split} within max_attempts")
    hashes = {}
    for split, values in rows.items():
        path = output / f"{split}.jsonl"; write_jsonl(path, values); hashes[split] = sha256_file(path)
    summary = {split: _summary(values, train_edges, held) for split, values in rows.items()}
    summary["signature_overlap_count"] = 0
    manifest = {"config": asdict(config), "splits": list(rows), "hashes": hashes, "train_coverage_edges": sorted(train_edges), "held_out_edges": sorted(held), "summary": summary}
    write_json(output / "manifest.json", manifest)
    return manifest
