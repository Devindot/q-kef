"""Validate the generated Q-KEF FiQA evolution benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from _phase1_common import load_config, repository_path
from qkef.datasets.evolution import EvolutionBenchmarkConfig, validate_benchmark
from qkef.datasets.fiqa import load_fiqa_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source-dir")
    parser.add_argument("--benchmark-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapping, _ = load_config(args.config)
    dataset_config = mapping["dataset"]
    raw_dir = repository_path(dataset_config["raw_dir"])
    source_dir = Path(args.source_dir) if args.source_dir else raw_dir / "fiqa"
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else repository_path(dataset_config["processed_dir"])
    result = validate_benchmark(
        benchmark_dir,
        load_fiqa_dataset(source_dir),
        EvolutionBenchmarkConfig.from_mapping(mapping),
        archive_path=raw_dir / "fiqa.zip",
        expected_md5=str(dataset_config["expected_md5"]),
    )
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}")
        print("Benchmark validation: FAILED")
        return 1
    print(f"Benchmark validation: PASS ({result.statistics['event_count']} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
