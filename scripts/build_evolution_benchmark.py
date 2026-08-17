"""Build the deterministic Q-KEF FiQA temporal evolution benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from _phase1_common import PROJECT_ROOT, load_config, repository_path
from qkef.datasets.evolution import (
    EvolutionBenchmarkConfig,
    build_benchmark,
    validate_benchmark,
)
from qkef.datasets.fiqa import load_fiqa_dataset
from qkef.schemas import EvolutionAction


def render_report(
    manifest: dict[str, object],
    content: object,
    validation: object,
    dataset_document_count: int,
    dataset_query_count: int,
) -> str:
    t0_units = content.t0_units
    t1_units = content.t1_units
    events = content.events
    lengths = [len(unit.text) for unit in t1_units]
    mutation_types = Counter(
        event.mutation_parameters.get("token_type", "unknown")
        for event in events
        if event.expected_action is EvolutionAction.REPLACE
    )
    lines = [
        "# Phase 1 Dataset Report",
        "",
        "This report describes generated benchmark infrastructure and integrity checks; it contains no model-performance results.",
        "",
        "## Source",
        "",
        f"- Expected FiQA MD5: `{manifest['fiqa_expected_md5']}`",
        f"- Observed FiQA MD5: `{manifest['fiqa_observed_md5']}`",
        f"- Checksum verified: `{manifest['fiqa_checksum_verified']}`",
        f"- Source corpus documents: {dataset_document_count}",
        f"- Source queries: {dataset_query_count}",
        "",
        "## Generated benchmark",
        "",
        f"- Events: {len(events)}",
        f"- T0 knowledge units: {len(t0_units)}",
        f"- T1 incoming units: {len(t1_units)}",
        f"- Unique FiQA source documents: {manifest['unique_source_document_count']}",
        f"- Human-review sample rows: {content.statistics['review_sample_count']}",
        "",
        "### Events by action",
        "",
    ]
    for action, count in manifest["count_per_action"].items():
        lines.append(f"- {action}: {count}")
    lines.extend(["", "### Events by split", ""])
    for split, count in manifest["count_per_benchmark_split"].items():
        lines.append(f"- {split}: {count}")
    lines.extend(
        [
            "",
            "### T1 text-length statistics (characters)",
            "",
            f"- Minimum: {min(lengths) if lengths else 0}",
            f"- Mean: {statistics.fmean(lengths):.2f}" if lengths else "- Mean: 0",
            f"- Median: {statistics.median(lengths):.2f}" if lengths else "- Median: 0",
            f"- Maximum: {max(lengths) if lengths else 0}",
            "",
            "### REPLACE mutation-token distribution",
            "",
        ]
    )
    lines.extend(f"- {key}: {value}" for key, value in sorted(mutation_types.items()))
    generation = content.statistics
    lines.extend(
        [
            "",
            "## Eligibility and integrity",
            "",
            f"- Eligible single-split source documents: {generation['eligible_exclusive_source_documents']}",
            f"- Unreferenced/non-positive source documents excluded: {generation['excluded_unreferenced_or_nonpositive_documents']}",
            f"- Cross-qrels-split documents excluded: {generation['excluded_cross_qrels_split_documents']}",
            f"- Empty-text documents excluded: {generation['excluded_empty_text_documents']}",
            f"- Duplicate/near-duplicate candidate pairs rejected: {generation['rejected_duplicate_or_near_duplicate_pairs']}",
            f"- Validation: {'PASS' if validation.passed else 'FAIL'}",
            f"- Train/dev source overlap: {validation.statistics['source_overlap']['train_dev']}",
            f"- Train/test source overlap: {validation.statistics['source_overlap']['train_test']}",
            f"- Dev/test source overlap: {validation.statistics['source_overlap']['dev_test']}",
            f"- Train/dev query overlap: {validation.statistics['query_overlap']['train_dev']}",
            f"- Train/test query overlap: {validation.statistics['query_overlap']['train_test']}",
            f"- Dev/test query overlap: {validation.statistics['query_overlap']['dev_test']}",
            f"- Deterministic regeneration: {'PASS' if manifest['deterministic_regeneration_status'] else 'FAIL'}",
            "",
            "## Warnings and shortfalls",
            "",
        ]
    )
    warnings = list(content.warnings) + list(validation.warnings)
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None.")
    lines.extend(
        [
            "",
            "The temporal events are controlled synthetic scenarios, not historical FiQA revisions. Human-review fields remain blank pending manual audit.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--report-path", default="reports/phase1_dataset_report.md")
    parser.add_argument("--no-determinism-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_mapping, _ = load_config(args.config)
    benchmark_config = EvolutionBenchmarkConfig.from_mapping(config_mapping)
    dataset_config = config_mapping["dataset"]
    raw_dir = repository_path(dataset_config["raw_dir"])
    source_dir = Path(args.source_dir) if args.source_dir else raw_dir / "fiqa"
    output_dir = Path(args.output_dir) if args.output_dir else repository_path(dataset_config["processed_dir"])
    report_path = repository_path(args.report_path)
    dataset = load_fiqa_dataset(source_dir)
    archive_path = raw_dir / "fiqa.zip"
    manifest, content = build_benchmark(
        dataset,
        benchmark_config,
        output_dir,
        expected_fiqa_md5=str(dataset_config["expected_md5"]),
        archive_path=archive_path,
        verify_determinism=not args.no_determinism_check,
    )
    validation = validate_benchmark(
        output_dir,
        dataset,
        benchmark_config,
        archive_path=archive_path,
        expected_md5=str(dataset_config["expected_md5"]),
    )
    manifest["leakage_validation_status"] = "pass" if not any("leakage" in error for error in validation.errors) else "fail"
    manifest["integrity_validation_status"] = "pass" if validation.passed else "fail"
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(manifest, content, validation, len(dataset.documents), len(dataset.queries)),
        encoding="utf-8",
        newline="\n",
    )
    if not validation.passed:
        for error in validation.errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Benchmark build: SUCCESS ({len(content.events)} events at {output_dir})")
    print(f"Deterministic rebuild: {'PASS' if manifest['deterministic_regeneration_status'] else 'FAIL'}")
    for warning in content.warnings + validation.warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
