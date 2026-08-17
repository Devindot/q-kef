"""Print observed counts and leakage status for the generated benchmark."""

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
    stats = result.statistics
    print("Q-KEF FiQA Evolution Benchmark")
    print("------------------------------")
    print(f"Events: {stats.get('event_count', 0)}")
    print("\nBy action:")
    for action, count in stats.get("count_per_action", {}).items():
        print(f"{action:<10} {count:>4}")
    print("\nBy split:")
    for split, count in stats.get("count_per_split", {}).items():
        print(f"{split:<10} {count:>4}")
    print(f"\nT0 units: {stats.get('t0_count', 0)}")
    print(f"T1 units: {stats.get('t1_count', 0)}")
    print("\nLeakage:")
    overlaps = stats.get("source_overlap", {})
    print(f"train/dev source overlap: {overlaps.get('train_dev', 0)}")
    print(f"train/test source overlap: {overlaps.get('train_test', 0)}")
    print(f"dev/test source overlap: {overlaps.get('dev_test', 0)}")
    print(f"\nValidation: {'PASS' if result.passed else 'FAIL'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
