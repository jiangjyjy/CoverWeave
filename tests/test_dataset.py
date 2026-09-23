from coverage_repro.dataset import build_splits
from coverage_repro.io import read_jsonl, sha256_file


def tiny_dataset_config():
    return {
        "N": 6,
        "lambda_values": [6.0],
        "graph_seeds": 1,
        "train_depths": [2, 3],
        "test_depths": [2, 3],
        "train_samples": 15,
        "valid_samples": 2,
        "test_samples": 2,
        "seed_base": 17,
    }


def test_build_splits_is_deterministic_and_checker_valid(tmp_path):
    config = tiny_dataset_config()
    first = build_splits(config, tmp_path / "first")
    second = build_splits(config, tmp_path / "second")

    assert first["hashes"] == second["hashes"]
    for split in ("train", "valid", "test"):
        rows = read_jsonl(tmp_path / "first" / f"{split}.jsonl")
        assert len(rows) == config[f"{split}_samples"] * 2
        assert all(row["checker_valid"] is True for row in rows)


def test_split_ids_and_episode_seeds_do_not_overlap(tmp_path):
    build_splits(tiny_dataset_config(), tmp_path)
    rows_by_split = {
        name: read_jsonl(tmp_path / f"{name}.jsonl")
        for name in ("train", "valid", "test")
    }
    for key in ("sample_id", "episode_seed"):
        values = {name: {row[key] for row in rows} for name, rows in rows_by_split.items()}
        assert values["train"].isdisjoint(values["valid"])
        assert values["train"].isdisjoint(values["test"])
        assert values["valid"].isdisjoint(values["test"])


def test_dataset_rows_have_required_serialized_schema_and_manifest_hashes(tmp_path):
    manifest = build_splits(tiny_dataset_config(), tmp_path)
    required_keys = {
        "sample_id", "split", "N", "lambda", "rho", "graph_edges", "inventory",
        "goal", "actions", "realized_pairs", "depth", "leaf_count", "graph_seed", "episode_seed",
        "checker_valid",
    }
    for split, expected_hash in manifest["hashes"].items():
        rows = read_jsonl(tmp_path / f"{split}.jsonl")
        assert rows
        assert required_keys <= rows[0].keys()
        assert sha256_file(tmp_path / f"{split}.jsonl") == expected_hash
