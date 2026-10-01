"""Construct and validate the Q-KEF v2 2,400-event benchmark.

This command constructs TEST labels as benchmark ground truth but performs no
model fitting, configuration selection, or TEST evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _phase1_common import load_config, repository_path
from qkef.datasets.evolution import EvolutionBenchmarkConfig, build_benchmark, validate_benchmark
from qkef.datasets.fiqa import load_fiqa_dataset
from qkef.v2.benchmark import partition_by_ancestry
from qkef.v2.canonical import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v2_confirmatory_benchmark.yaml")
    parser.add_argument("--source-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--report-path", default="reports/v2/CONFIRMATORY_BENCHMARK_BUILD.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapping, _ = load_config(args.config)
    benchmark_config = EvolutionBenchmarkConfig.from_mapping(mapping)
    dataset_config = mapping["dataset"]
    raw_dir = repository_path(dataset_config["raw_dir"])
    source_dir = Path(args.source_dir) if args.source_dir else raw_dir / "fiqa"
    output_dir = Path(args.output_dir) if args.output_dir else repository_path(dataset_config["processed_dir"])
    source = load_fiqa_dataset(source_dir)
    partition = partition_by_ancestry(source, benchmark_config.split_targets, seed=benchmark_config.seed)
    manifest, content = build_benchmark(
        partition.dataset,
        benchmark_config,
        output_dir,
        expected_fiqa_md5=str(dataset_config["expected_md5"]),
        archive_path=raw_dir / "fiqa.zip",
        verify_determinism=True,
    )
    validation = validate_benchmark(
        output_dir,
        partition.dataset,
        benchmark_config,
        archive_path=raw_dir / "fiqa.zip",
        expected_md5=str(dataset_config["expected_md5"]),
    )
    overlaps = {**validation.statistics.get("source_overlap", {}), **validation.statistics.get("query_overlap", {})}
    if any(overlaps.values()):
        validation.errors.append(f"v2 requires zero source and query overlap, observed {overlaps}")
    partition_path = output_dir / "ancestry_partition_manifest.json"
    partition_path.write_text(json.dumps(partition.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    manifest.update(
        {
            "benchmark_role": "confirmatory benchmark construction",
            "model_evaluation_status": "not_executed",
            "test_evaluation_status": "not_executed",
            "configuration_selection_allowed_splits": ["train", "dev"],
            "conformal_fit_split": "calibration",
            "ancestry_partition_manifest": partition_path.name,
            "ancestry_partition_manifest_sha256": sha256_file(partition_path),
            "leakage_validation_status": "pass" if not any(overlaps.values()) else "fail",
            "integrity_validation_status": "pass" if validation.passed else "fail",
        }
    )
    (output_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    report_path = repository_path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tracked_manifest_dir = report_path.parent / "confirmatory_benchmark"
    tracked_manifest_dir.mkdir(parents=True, exist_ok=True)
    (tracked_manifest_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (tracked_manifest_dir / "ancestry_partition_manifest.json").write_text(
        json.dumps(partition.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    report_lines = [
        "# Q-KEF v2 Confirmatory Benchmark Build",
        "",
        "This artifact reports benchmark construction and integrity only. No model was fitted or evaluated on TEST.",
        "",
        f"- Requested events: {manifest['requested_number_of_events']}",
        f"- Constructed events: {manifest['actual_number_of_events']}",
        f"- Events per action: `{json.dumps(manifest['count_per_action'], sort_keys=True)}`",
        f"- Events per split: `{json.dumps(manifest['count_per_benchmark_split'], sort_keys=True)}`",
        f"- Unique source documents: {manifest['unique_source_document_count']}",
        f"- Deterministic regeneration: {manifest['deterministic_regeneration_status']}",
        f"- Source/query ancestry overlap: {json.dumps(overlaps, sort_keys=True)}",
        f"- Integrity validation: {'PASS' if validation.passed else 'FAIL'}",
        "- TEST evaluation: NOT EXECUTED",
        f"- Deterministic content SHA-256: `{manifest['deterministic_content_sha256']}`",
        "",
        "The events remain controlled synthetic temporal scenarios and require the planned human audit.",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8", newline="\n")
    if not validation.passed:
        for error in validation.errors:
            print(f"ERROR: {error}")
        return 1
    print(json.dumps({"status": "PASS", "events": len(content.events), "splits": manifest["count_per_benchmark_split"], "test_evaluation": "not_executed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
