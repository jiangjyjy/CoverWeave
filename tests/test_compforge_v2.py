import json

from coverage_repro.compforge_v2 import CompForgeV2Config, build_compforge_v2_splits


def test_v2_splits_are_disjoint_and_obey_edge_and_depth_contracts(tmp_path):
    manifest = build_compforge_v2_splits(CompForgeV2Config(samples_per_split=3), tmp_path)
    rows = {
        split: [json.loads(line) for line in (tmp_path / f"{split}.jsonl").read_text().splitlines()]
        for split in manifest["splits"]
    }
    train_edges = set(map(tuple, manifest["train_coverage_edges"]))
    held_out = set(map(tuple, manifest["held_out_edges"]))
    signatures = [row["task_signature"] for split_rows in rows.values() for row in split_rows]

    assert len(signatures) == len(set(signatures))
    assert all(set(map(tuple, row["realized_pairs"])) <= train_edges for split in ("train", "valid") for row in rows[split])
    assert all(set(map(tuple, row["realized_pairs"])) & held_out for split in ("held_out_same_depth", "held_out_deep") for row in rows[split])
    assert min(row["depth"] for row in rows["held_out_deep"]) > max(row["depth"] for row in rows["train"])


def test_v2_builder_is_deterministic(tmp_path):
    config = CompForgeV2Config(samples_per_split=2)
    first = build_compforge_v2_splits(config, tmp_path / "first")
    second = build_compforge_v2_splits(config, tmp_path / "second")

    assert first["hashes"] == second["hashes"]
    assert first["summary"]["signature_overlap_count"] == 0
