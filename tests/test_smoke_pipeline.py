import json

import yaml

from coverage_repro.evaluate import run_smoke
from coverage_repro.io import read_yaml


def test_smoke_pipeline_writes_complete_finite_outputs(tmp_path):
    config = yaml.safe_load(
        """
        N: 6
        lambda_values: [0.5, 1.0]
        train_depths: [2]
        test_depths: [3]
        graph_seeds: 2
        model_seeds: 1
        train_episodes: 4
        test_episodes: 5
        """
    )
    summary = run_smoke(config, tmp_path)
    assert summary["result_rows"] == 2 * 1 * 2 * 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    results = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(manifest["graphs"]) == 4
    assert len(results) == summary["result_rows"]
    assert all(row["bfs_success"] in (0, 1) for row in results)
    assert all(row["bfs_oracle_success"] == 1 for row in results)
    assert all(row["checker_valid_rate"] == 1.0 for row in results)
    assert all(row["checker_pass_rate"] == 1.0 for row in results)
    assert summary["checker_valid_rate"] == 1.0
    assert summary["bfs_oracle_rate"] == 1.0


def test_yaml_reader_returns_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("N: 4\nlambda_values: [1.0]\n")
    assert read_yaml(path)["N"] == 4
