from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Iterable

from .graph import CoverageGraph


def component_summary(graph: CoverageGraph) -> dict[str, object]:
    labels = graph.component_labels()
    sizes: dict[int, int] = defaultdict(int)
    for label in labels:
        sizes[label] += 1
    ordered = sorted(sizes.values(), reverse=True)
    return {
        "component_count": len(ordered),
        "component_sizes": ordered,
        "largest_component": ordered[0] if ordered else 0,
        "largest_fraction": (ordered[0] / graph.n) if ordered else 0.0,
    }


def _linear_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, mean([value for _, value in points]) if points else 0.0
    x_bar = mean(x for x, _ in points)
    y_bar = mean(y for _, y in points)
    denominator = sum((x - x_bar) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0, y_bar
    slope = sum((x - x_bar) * (y - y_bar) for x, y in points) / denominator
    return slope, y_bar - slope * x_bar


def summarize_results(rows: Iterable[dict[str, object]], n: int) -> dict[str, object]:
    rows = list(rows)
    by_depth: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_depth[int(row["depth"])].append(row)

    thresholds = []
    for depth, group in sorted(by_depth.items()):
        by_lambda: dict[float, list[dict[str, object]]] = defaultdict(list)
        for row in group:
            by_lambda[float(row["lambda"])].append(row)
        estimate = None
        crossed = False
        for lam in sorted(by_lambda):
            episodes = sum(int(r["episodes"]) for r in by_lambda[lam])
            successes = sum(int(r["success"]) for r in by_lambda[lam])
            rate = successes / max(episodes, 1)
            if rate >= 0.5 and not crossed:
                estimate = lam
                crossed = True
        if estimate is None:
            estimate = max(by_lambda) if by_lambda else 0.0
        thresholds.append({"depth": depth, "threshold": float(estimate), "crossed": crossed})

    p2 = []
    for depth, group in sorted(by_depth.items()):
        points = []
        values = []
        for row in group:
            success = int(row["success"])
            episodes = max(int(row["episodes"]), 1)
            q = (1.0 / max(depth, 1)) * math.log((success + 0.5) / (episodes + 1))
            points.append((float(row["lambda"]), q))
            values.append((float(row["lambda"]), q, episodes))
        slope, intercept = _linear_fit(points)
        for lam, q, episodes in values:
            residual = q - (intercept + slope * lam)
            p2.append({
                "depth": depth,
                "lambda": lam,
                "q": q,
                "residual": residual,
                "scaled_residual": residual * math.sqrt(episodes),
            })

    p3 = []
    for (depth, lam), group in _group_by_depth_lambda(rows).items():
        rates = [
            int(row["success"]) / max(int(row["episodes"]), 1)
            for row in group
        ]
        variance = mean([(rate - mean(rates)) ** 2 for rate in rates]) if rates else 0.0
        p3.append({
            "depth": depth,
            "lambda": lam,
            "chi": float(n * variance),
            "seed_count": len(rates),
        })
    peak_item = max(p3, key=lambda item: item["chi"]) if p3 else {"chi": 0.0}
    finite_values = [item["threshold"] for item in thresholds]
    finite_values.extend(item[key] for item in p2 for key in ("q", "residual", "scaled_residual"))
    finite_values.extend(item["chi"] for item in p3)
    if not all(math.isfinite(float(value)) for value in finite_values):
        raise ValueError("statistics contain non-finite values")
    return {
        "p1": {"thresholds": thresholds},
        "p2": p2,
        "p3": {"rows": p3, "peak": float(peak_item["chi"])},
        "finite_values": finite_values,
    }


def _group_by_depth_lambda(rows: Iterable[dict[str, object]]) -> dict[tuple[int, float], list[dict[str, object]]]:
    grouped: dict[tuple[int, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["depth"]), float(row["lambda"]))].append(row)
    return grouped
