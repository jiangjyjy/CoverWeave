from pathlib import Path
import subprocess
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_integration_smoke_config_is_small_cuda_grid():
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "smoke.yaml").read_text())

    assert config["N"] == 12
    assert config["lambda_values"] == [0.6, 1.0, 1.5]
    assert config["train_depths"] == [2, 3, 4]
    assert config["test_depths"] == [5, 8]
    assert config["graph_seeds"] == 3
    assert config["model_seeds"] == 1
    assert config["seed"] == 1
    assert config["train_samples"] == 9
    assert config["valid_samples"] == 2
    assert config["test_samples"] == 2
    assert config["epochs"] == 2
    assert config["batch_size"] == 32
    assert config["d_model"] == 128
    assert config["nhead"] == 4
    assert config["num_layers"] == 2
    assert config["dim_feedforward"] == 512
    assert config["dropout"] == 0.1
    assert config["learning_rate"] == 0.001
    assert config["weight_decay"] == 0.01
    assert config["grad_clip"] == 1.0
    assert config["device"] == "cuda"


def test_gpu_scripts_pin_runtime_log_commands_and_validate_artifacts():
    env_check = (PROJECT_ROOT / "scripts" / "env_check.sh").read_text()
    smoke = (PROJECT_ROOT / "scripts" / "run_smoke.sh").read_text()

    assert 'PYTHON="${PYTHON:-python3}"' in env_check
    assert 'PYTHON="${PYTHON:-python3}"' in smoke
    assert "sys.version_info" in env_check
    assert "torch.cuda.is_available()" in env_check
    assert r"\${" not in env_check
    assert "set -euo pipefail" in smoke
    assert "artifacts/sft-smoke" in smoke
    assert "resolved-config.yaml" in smoke
    assert "bash.log" in smoke
    assert "nvidia-smi" in smoke
    assert "timeout 30 nvidia-smi" in smoke
    assert "$RUN_DIR/nvidia-smi.txt" in smoke
    assert "build-dataset" in smoke
    assert "train-sft" in smoke
    assert "evaluate-sft" in smoke
    assert "validate_sft_smoke.py" in smoke
    assert r"\${" not in smoke
    assert (PROJECT_ROOT / "scripts" / "validate_sft_smoke.py").exists()


def test_package_and_readme_document_reproducible_sft_smoke():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
    readme = (PROJECT_ROOT / "README.md").read_text().lower()

    assert "coverage-repro = \"coverage_repro.cli:main\"" in pyproject
    assert "coverage-repro build-dataset" in readme
    assert "coverage-repro train-sft" in readme
    assert "coverage-repro evaluate-sft" in readme
    assert "no external api" in readme
    assert "anonymous reproducibility" in readme
    assert "integration smoke" in readme
    assert "not paper evidence" in readme


def test_artifact_validator_is_invocable_from_project_root():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_sft_smoke.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
