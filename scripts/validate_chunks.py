"""Validate Phase 2 ingestion, chunk coverage, provenance, and leakage controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from _phase1_common import load_config, repository_path
from qkef.ingestion.corpus import validate_phase2_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phase1-dir")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _ = load_config(args.config)
    dataset = config["dataset"]
    phase1_dir = Path(args.phase1_dir) if args.phase1_dir else repository_path(dataset["processed_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else repository_path(dataset["chunk_processed_dir"])
    result = validate_phase2_corpus(phase1_dir, output_dir, config)
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    if not result.passed:
        for error in result.errors:
            print(f"ERROR: {error}")
        print("Phase 2 validation: FAILED")
        return 1
    print(
        f"Phase 2 validation: PASS ({result.statistics['ingested_t0'] + result.statistics['ingested_t1']} units, "
        f"{result.statistics['model_content_coverage_percent']}% coverage)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
