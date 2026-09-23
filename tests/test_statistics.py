import math

from coverage_repro.percolation import summarize_results


def test_statistics_emit_p1_p2_p3_without_nonfinite_values():
    rows = [
        {"lambda": 0.5, "depth": 2, "seed": 0, "success": 0, "episodes": 10},
        {"lambda": 0.5, "depth": 2, "seed": 1, "success": 1, "episodes": 10},
        {"lambda": 1.5, "depth": 2, "seed": 0, "success": 10, "episodes": 10},
        {"lambda": 1.5, "depth": 2, "seed": 1, "success": 9, "episodes": 10},
    ]
    summary = summarize_results(rows, n=12)
    assert summary["p1"]["thresholds"]
    assert all("q" in item and "scaled_residual" in item and "residual" in item for item in summary["p2"])
    assert summary["p3"]["peak"] >= 0
    assert all(math.isfinite(value) for value in summary["finite_values"])
