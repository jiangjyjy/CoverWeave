from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


def jsonable(value: object) -> object:
    if hasattr(value, "item") and callable(value.item):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def assert_finite(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")


def write_json(path: str | Path, value: object) -> None:
    payload = jsonable(value)
    assert_finite(payload)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[object]) -> None:
    output = []
    for row in rows:
        payload = jsonable(row)
        assert_finite(payload)
        output.append(json.dumps(payload, sort_keys=True))
    Path(path).write_text("\n".join(output) + ("\n" if output else ""))


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: str | Path) -> dict[str, object]:
    import yaml

    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value
