#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CONFIG_INPUT="${1:-configs/smoke.yaml}"

RECORD_GIT="${RECORD_GIT:-0}"

cd "$PROJECT_DIR"
command -v "$PYTHON" >/dev/null 2>&1 || { printf 'Python executable not found: %s\n' "$PYTHON" >&2; exit 1; }
[[ -f "$CONFIG_INPUT" ]] || { printf 'missing config: %s\n' "$CONFIG_INPUT" >&2; exit 1; }

case "$CONFIG_INPUT" in
    "$PROJECT_DIR"/*) CONFIG_LABEL="${CONFIG_INPUT#"$PROJECT_DIR"/}" ;;
    /*)               CONFIG_LABEL="$(basename "$CONFIG_INPUT")" ;;
    *)                CONFIG_LABEL="$CONFIG_INPUT" ;;
esac

RUN_DIR="artifacts/sft-smoke/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/bash.log"
COMMANDS_PATH="$RUN_DIR/commands.sh"
touch "$LOG_PATH" "$COMMANDS_PATH"
exec > >(tee -a "$LOG_PATH") 2>&1

run_command() {
    printf '%q ' "$@" >> "$COMMANDS_PATH"
    printf '\n' >> "$COMMANDS_PATH"
    "$@"
}

printf 'run_dir=%s\n' "$RUN_DIR"
printf 'config=%s\n' "$CONFIG_LABEL"
run_command bash scripts/env_check.sh

if [[ "$RECORD_GIT" == "1" ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$RUN_DIR/git-commit.txt"
    git status --short > "$RUN_DIR/git-status.txt"
else
    printf 'not-recorded\n' > "$RUN_DIR/git-commit.txt"
    printf 'not-recorded\n' > "$RUN_DIR/git-status.txt"
fi

if ! timeout 30 nvidia-smi --query-gpu=index,name,driver_version,memory.total \
        --format=csv,noheader > "$RUN_DIR/nvidia-smi.txt"; then
    printf 'nvidia-smi timed out or failed\n' >> "$RUN_DIR/nvidia-smi.txt"
fi

"$PYTHON" - <<'PY' > "$RUN_DIR/environment.txt"
import importlib.metadata
import platform
import sys

import torch

# Coarse platform facts only: no node name, no kernel build string.
print(f"python={sys.version.split()[0]}")
print(f"platform={platform.system()}-{platform.machine()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
try:
    print(f"gpu={torch.cuda.get_device_name(0)}")
except Exception:
    print("gpu=unavailable")
for package in ("numpy", "PyYAML", "pytest", "coverage-repro"):
    try:
        print(f"{package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}=not-installed")
PY

"$PYTHON" - "$CONFIG_INPUT" "$RUN_DIR/resolved-config.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
if not isinstance(config, dict):
    raise SystemExit("configuration root must be a mapping")
Path(sys.argv[2]).write_text(yaml.safe_dump(config, sort_keys=True))
PY

COMMAND_ENV="PYTHONPATH=."
run_command env "$COMMAND_ENV" "$PYTHON" -m coverage_repro.cli build-dataset \
    --config "$RUN_DIR/resolved-config.yaml" --output "$RUN_DIR/dataset"
"$PYTHON" - "$RUN_DIR/dataset/manifest.json" "$RUN_DIR/data_hashes.json" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(json.dumps(manifest["hashes"], indent=2, sort_keys=True) + "\n")
PY
run_command env "$COMMAND_ENV" "$PYTHON" -m coverage_repro.cli train-sft \
    --config "$RUN_DIR/resolved-config.yaml" --dataset "$RUN_DIR/dataset" \
    --run-dir "$RUN_DIR/training" --device cuda
run_command env "$COMMAND_ENV" "$PYTHON" -m coverage_repro.cli evaluate-sft \
    --checkpoint "$RUN_DIR/training/checkpoints/best.pt" --dataset "$RUN_DIR/dataset" \
    --output "$RUN_DIR/evaluation" --device cuda
run_command env "$COMMAND_ENV" "$PYTHON" scripts/validate_sft_smoke.py \
    --run-dir "$RUN_DIR"

printf 'smoke artifact: %s\n' "$RUN_DIR"
