#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

cd "$PROJECT_DIR"
command -v "$PYTHON" >/dev/null 2>&1 || { printf 'Python executable not found: %s\n' "$PYTHON" >&2; exit 1; }

"$PYTHON" - <<'PY'
import importlib
import sys

if sys.version_info < (3, 10):
    raise SystemExit(f"Python >=3.10 required, found {sys.version}")
for package in ("numpy", "yaml", "pytest", "torch"):
    importlib.import_module(package)

import torch

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
name = torch.cuda.get_device_name(0)
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu={name}")
PY

nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
printf '%s\n' "environment check: PASS"
