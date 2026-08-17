"""Print observed Phase 2 sanitization, chunking, and integrity statistics."""

from __future__ import annotations

import argparse
import json
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
    manifest = json.loads((output_dir / "phase2_manifest.json").read_text(encoding="utf-8"))
    counts = manifest["ingested_counts"]
    print("Q-KEF Phase 2 Chunk Corpus")
    print("--------------------------")
    print(f"\nIngested units\nT0: {counts['T0']}\nT1: {counts['T1']}\nTotal: {counts['total']}")
    for strategy, stats in manifest["chunk_statistics"].items():
        print(
            f"\n{strategy}\nchunks: {stats['chunk_count']}\n"
            f"mean words: {stats['mean_words']}\nmedian words: {stats['median_words']}\n"
            f"multi-chunk units: {stats['multi_chunk_unit_count']}"
        )
    diagnostics = manifest["tfidf_diagnostics"]
    print("\nTF-IDF diagnostics")
    print(f"lexical boundaries: {diagnostics['lexical_boundaries']}")
    print(f"forced boundaries: {diagnostics['forced_boundaries']}")
    print(f"fallback units: {diagnostics['fallback_units']}")
    print("\nSanitization")
    print(f"units changed: {counts['model_text_changed']}")
    for operation, count in manifest["sanitization_operation_counts"].items():
        print(f"{operation}: {count}")
    overlap = result.statistics.get("source_overlap", {})
    print("\nLeakage")
    print(f"train/dev source overlap: {overlap.get('train_dev', 0)}")
    print(f"train/test source overlap: {overlap.get('train_test', 0)}")
    print(f"dev/test source overlap: {overlap.get('dev_test', 0)}")
    print(f"synthetic markers in model text: {sum(result.statistics.get('marker_occurrences', {}).values())}")
    print(f"\nValidation: {'PASS' if result.passed else 'FAIL'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
