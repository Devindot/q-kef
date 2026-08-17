"""Build provenance-safe ingested units and three deterministic chunk corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _phase1_common import PROJECT_ROOT, load_config, repository_path
from qkef.ingestion.corpus import (
    build_phase2_corpus,
    render_phase2_report,
    validate_phase2_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phase1-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--report-path", default="reports/phase2_ingestion_chunking_report.md")
    parser.add_argument("--no-determinism-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _ = load_config(args.config)
    dataset = config["dataset"]
    phase1_dir = Path(args.phase1_dir) if args.phase1_dir else repository_path(dataset["processed_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else repository_path(dataset["chunk_processed_dir"])
    report_path = repository_path(args.report_path)
    manifest, _ = build_phase2_corpus(
        phase1_dir,
        output_dir,
        config,
        verify_determinism=not args.no_determinism_check,
    )
    validation = validate_phase2_corpus(phase1_dir, output_dir, config)
    manifest["label_leakage_audit_status"] = "pass" if not any(
        "leak" in error or "label" in error for error in validation.errors
    ) else "fail"
    manifest["source_ancestry_integrity_status"] = "pass" if not any(
        validation.statistics.get("source_overlap", {}).values()
    ) else "fail"
    manifest["integrity_validation_status"] = "pass" if validation.passed else "fail"
    (output_dir / "phase2_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_phase2_report(manifest, validation), encoding="utf-8", newline="\n"
    )
    if not validation.passed:
        for error in validation.errors:
            print(f"ERROR: {error}")
        print("Phase 2 build: FAILED validation")
        return 1
    counts = manifest["ingested_counts"]
    print(
        f"Phase 2 build: SUCCESS ({counts['T0']} T0 + {counts['T1']} T1 units)"
    )
    for strategy, statistics in manifest["chunk_statistics"].items():
        print(f"{strategy}: {statistics['chunk_count']} chunks")
    print(
        f"Deterministic rebuild: {'PASS' if manifest['deterministic_rebuild_status'] else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
