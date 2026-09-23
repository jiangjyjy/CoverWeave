from pathlib import Path

from coverage_repro.compforge_v3 import CompForgeV3Config, build_branching_instance
from coverage_repro.compforge_v3_eval import (
    build_v3_codecs,
    evaluate_oracle,
    evaluate_checkpoint_sft,
    evaluate_random_legal,
    evaluate_trained_sft,
    evaluate_untrained_transformer,
    run_v3_gate,
)


def _rows():
    config = CompForgeV3Config()
    return [
        build_branching_instance(config, seed=seed, goal_variant=seed % 2)
        for seed in range(10, 18)
    ]


def test_four_real_baselines_and_checkpoint_replay(tmp_path):
    rows = _rows()
    codecs = build_v3_codecs(rows, d_model=16, nhead=4, num_layers=1, dim_feedforward=32)
    oracle = evaluate_oracle(rows[4:])
    random = evaluate_random_legal(rows[4:], codecs=codecs, seed=3, action_budget=4)
    untrained = evaluate_untrained_transformer(rows[4:], codecs=codecs, seed=5, device="cpu", action_budget=4)
    trained = evaluate_trained_sft(
        rows[:4],
        rows[4:],
        codecs=codecs,
        seed=7,
        device="cpu",
        run_dir=tmp_path,
        epochs=30,
        batch_size=4,
        action_budget=4,
    )

    assert oracle["exact_solve_rate"] == 1.0
    assert random["failure_counts"]
    assert "episodes" in untrained and "episodes" in trained
    assert Path(trained["checkpoint"]).is_file()
    replayed = evaluate_checkpoint_sft(
        trained["checkpoint"], rows[4:], device="cpu", action_budget=4
    )
    assert replayed["exact_solve_rate"] == trained["exact_solve_rate"]


def test_gate_preserves_all_baselines_when_separation_fails():
    zero = {"exact_solve_rate": 0.0}
    per_split = {
        split: {
            "oracle": {"exact_solve_rate": 1.0},
            "random": zero,
            "untrained": zero,
            "trained": zero,
        }
        for split in ("valid", "held_out_same_depth", "held_out_deep")
    }
    gate = run_v3_gate(per_split)

    assert gate["passed"] is False
    assert set(gate["per_split"]) == {"valid", "held_out_same_depth", "held_out_deep"}
    assert all(set(metrics) == {"oracle", "random", "untrained", "trained"} for metrics in gate["per_split"].values())
    assert gate["conditions"]["oracle_all_splits"] is True
    assert gate["conditions"]["valid_trained_nonzero"] is False
    assert gate["conditions"]["held_out_same_depth_trained_beats_random"] is False
    assert gate["conditions"]["held_out_same_depth_trained_beats_untrained"] is False
    assert gate["conditions"]["held_out_deep_trained_beats_random"] is False
    assert gate["conditions"]["held_out_deep_trained_beats_untrained"] is False
