from __future__ import annotations

from pathlib import Path
from typing import Any

from .baselines import bfs_oracle, component_oracle
from .checker import check_trace
from .generator import generate_trace
from .graph import CoverageGraph, make_goal, tree_to_dict
from .io import assert_finite, write_json, write_jsonl
from .percolation import component_summary, summarize_results
from .env import MergeAction


def _seed_values(value: Any) -> list[int]:
    if isinstance(value, int):
        return list(range(value))
    return [int(item) for item in value]


def _action_dict(action: MergeAction) -> dict[str, int]:
    return {"left_index": action.left_index, "right_index": action.right_index}


def _component_goal(graph: CoverageGraph, depth: int, seed: int):
    components: dict[int, list[int]] = {}
    for atom, label in enumerate(graph.component_labels()):
        components.setdefault(label, []).append(atom)
    atoms = sorted(max(components.values(), key=lambda values: (len(values), -min(values))))
    if len(atoms) == 1:
        return atoms[0]
    return make_goal(len(atoms), depth, seed, atom_ids=atoms)

def run_smoke(config: dict[str, Any], output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    n = int(config["N"])
    lambdas = [float(value) for value in config["lambda_values"]]
    test_depths = [int(value) for value in config["test_depths"]]
    train_depths = [int(value) for value in config["train_depths"]]
    graph_seeds = _seed_values(config["graph_seeds"])
    model_seeds = _seed_values(config["model_seeds"])
    train_episodes = int(config["train_episodes"])
    test_episodes = int(config["test_episodes"])
    seed_base = int(config.get("seed_base", 0))
    budget = int(config.get("random_action_budget", 2 * n))

    graphs = []
    graph_by_key = {}
    for lambda_index, lam in enumerate(lambdas):
        for graph_seed in graph_seeds:
            graph = CoverageGraph.random(n, lam, seed_base + 1000 * lambda_index + graph_seed)
            entry = graph.manifest(lam, graph_seed)
            entry["components"] = component_summary(graph)
            graphs.append(entry)
            graph_by_key[(lam, graph_seed)] = graph
    write_json(
        output / "manifest.json",
        {
            "N": n,
            "formula": {
                "m": "round(lambda*N/2)",
                "rho": "2*m/(N*(N-1))",
                "connectedness": "not forced",
            },
            "config": config,
            "graphs": graphs,
        },
    )

    trajectory_rows = []
    for lam in lambdas:
        for graph_seed in graph_seeds:
            graph = graph_by_key[(lam, graph_seed)]
            for depth in test_depths:
                goal = _component_goal(graph, depth, seed_base + graph_seed + depth)
                oracle = bfs_oracle(graph, goal)
                checked = check_trace(graph, goal, oracle.actions)
                oracle_check_pass = int(checked.valid == oracle.success)
                trajectory_rows.append({
                    "lambda": lam,
                    "graph_seed": graph_seed,
                    "depth": depth,
                    "goal": tree_to_dict(goal),
                    "oracle_success": int(oracle.success),
                    "oracle_actions": [_action_dict(action) for action in oracle.actions],
                    "checker_valid": int(checked.valid),
                    "checker_valid_rate": float(checked.valid),
                    "checker_pass": oracle_check_pass,
                })
    write_jsonl(output / "trajectories.jsonl", trajectory_rows)

    train_rows = []
    for lam in lambdas:
        for graph_seed in graph_seeds:
            graph = graph_by_key[(lam, graph_seed)]
            for depth in train_depths:
                goal = _component_goal(graph, depth, seed_base + 50000 + graph_seed + depth)
                successes = 0
                checks = 0
                for episode in range(train_episodes):
                    trace = generate_trace(graph, goal, "random", seed_base + episode + depth)
                    checked = check_trace(graph, goal, trace.actions)
                    successes += int(trace.success)
                    checks += int(checked.valid == trace.success)
                train_rows.append({
                    "lambda": lam,
                    "graph_seed": graph_seed,
                    "depth": depth,
                    "episodes": train_episodes,
                    "success": successes,
                    "checker_pass_rate": checks / max(train_episodes, 1),
                    "backend": "fallback-random",
                })
    write_jsonl(output / "train_metrics.jsonl", train_rows)

    result_rows = []
    for lam in lambdas:
        for graph_seed in graph_seeds:
            graph = graph_by_key[(lam, graph_seed)]
            for depth in test_depths:
                goal = _component_goal(graph, depth, seed_base + graph_seed + depth)
                oracle = bfs_oracle(graph, goal)
                oracle_checked = check_trace(graph, goal, oracle.actions)
                component = component_oracle(graph, goal)
                for model_seed in model_seeds:
                    successes = 0
                    checks = 0
                    for episode in range(test_episodes):
                        trace = generate_trace(
                            graph,
                            goal,
                            "random",
                            seed_base + 100000 + model_seed * 10000 + graph_seed * 100 + episode,
                            budget=budget,
                        )
                        checked = check_trace(graph, goal, trace.actions)
                        successes += int(trace.success)
                        checks += int(checked.valid == trace.success)
                    result_rows.append({
                        "lambda": lam,
                        "depth": depth,
                        "graph_seed": graph_seed,
                        "model_seed": model_seed,
                        "episodes": test_episodes,
                        "success": successes,
                        "success_rate": successes / max(test_episodes, 1),
                        "bfs_success": int(oracle.success),
                        "bfs_oracle_success": int(oracle.success),
                        "component_success": int(component),
                        "checker_valid_rate": float(oracle_checked.valid),
                        "checker_pass_rate": checks / max(test_episodes, 1),
                        "action_budget": budget,
                    })
    write_jsonl(output / "results.jsonl", result_rows)
    checker_valid_values = [
        row["checker_valid_rate"] for row in result_rows
    ] + [row["checker_valid_rate"] for row in trajectory_rows]
    bfs_oracle_values = [row["bfs_success"] for row in result_rows]
    if not checker_valid_values or min(checker_valid_values) != 1.0:
        raise AssertionError("checker_valid_rate must be 1.0 for generated oracle traces")
    if not bfs_oracle_values or min(bfs_oracle_values) != 1:
        raise AssertionError("BFS oracle must succeed for every generated goal")
    summary = summarize_results(result_rows, n)
    summary.update({
        "result_rows": len(result_rows),
        "trajectory_rows": len(trajectory_rows),
        "train_rows": len(train_rows),
        "checker_valid_rate": min(checker_valid_values),
        "checker_pass_rate": min(
            [row["checker_pass_rate"] for row in result_rows] + [row["checker_pass"] for row in trajectory_rows]
        ),
        "bfs_oracle_rate": sum(bfs_oracle_values) / max(len(bfs_oracle_values), 1),
        "component_oracle_rate": sum(row["component_success"] for row in result_rows) / max(len(result_rows), 1),
    })
    assert_finite(summary)
    write_json(output / "summary.json", summary)
    return summary
