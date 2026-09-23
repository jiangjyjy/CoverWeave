from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .dataset import build_splits
from .evaluate import run_smoke
from .evaluate_sft import evaluate_checkpoint
from .graph import CoverageGraph, make_goal
from .io import read_yaml
from .train_sft import train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coverage-repro")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run the complete MVP pipeline")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--output", required=True)

    generate = subparsers.add_parser("generate", help="print one graph and goal")
    generate.add_argument("--N", type=int, required=True)
    generate.add_argument("--lambda", dest="lam", type=float, required=True)
    generate.add_argument("--depth", type=int, required=True)
    generate.add_argument("--seed", type=int, default=0)

    dataset = subparsers.add_parser("build-dataset", help="build verified JSONL splits")
    dataset.add_argument("--config", required=True)
    dataset.add_argument("--output", required=True)

    training = subparsers.add_parser("train-sft", help="train a merge Transformer")
    training.add_argument("--config", required=True)
    training.add_argument("--dataset", required=True)
    training.add_argument("--run-dir", required=True)
    training.add_argument("--device", required=True)

    evaluation = subparsers.add_parser("evaluate-sft", help="evaluate a trained checkpoint")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "smoke":
            result = run_smoke(read_yaml(args.config), args.output)
        elif args.command == "generate":
            graph = CoverageGraph.random(args.N, args.lam, args.seed)
            goal = make_goal(args.N, args.depth, args.seed)
            result = {
                "graph": graph.manifest(args.lam, args.seed),
                "goal": str(goal),
            }
        elif args.command == "build-dataset":
            result = build_splits(read_yaml(args.config), args.output)
        elif args.command == "train-sft":
            config = read_yaml(args.config)
            config["dataset_dir"] = args.dataset
            result = train(config, args.run_dir, args.device)
        else:
            result = evaluate_checkpoint(
                args.checkpoint,
                args.dataset,
                args.output,
                args.device,
            )
    except Exception as error:
        print(f"coverage-repro: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
