# CoverWeave

Anonymous reproducibility code for coverage-governed compositional depth
extrapolation.

The repository contains the symbolic environment, deterministic data
generation, SFT training and evaluation code, checker replay, statistics, and
tests. It does not contain model checkpoints or paper result artifacts.

## Setup

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

## CPU tests

```bash
python -m pytest -q --ignore=tests/test_training.py
```

## Reproduction smoke

The bounded smoke writes all outputs under a user-selected `artifacts/`
directory, which is ignored by Git:

```bash
PYTHON=python CUDA_VISIBLE_DEVICES=0 bash scripts/run_smoke.sh configs/smoke.yaml
```

The smoke is an integration smoke that validates data generation,
training/evaluation wiring, checker replay, checkpoint reload, and durable
artifact hashes. It is not paper evidence.

The command-line entry points are:

```bash
coverage-repro build-dataset --config configs/smoke.yaml --output DATASET_DIR
coverage-repro train-sft --config configs/smoke.yaml --dataset DATASET_DIR --run-dir RUN_DIR --device cuda
coverage-repro evaluate-sft --checkpoint RUN_DIR/checkpoints/best.pt --dataset DATASET_DIR --output OUTPUT_DIR --device cuda
```

## Scope

All code runs locally and makes no external API calls. Checkpoints, datasets,
logs, caches, and generated results are intentionally excluded from this
anonymous source release.
