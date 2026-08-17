# Phase 2 Ingestion and Chunking Report

This report describes deterministic preprocessing infrastructure, not model or retrieval performance.

## Dataset

- T0 units: 200
- T1 units: 300
- Total units: 500

## Sanitization

- Units with raw/model differences: 369
- `neutralize_synthetic_split_boundary`: 50
- `normalize_horizontal_whitespace`: 293
- `remove_archive_target_identifier`: 50
- `remove_synthetic_merge_header`: 50
- `remove_zero_width_characters`: 1
- `trim_leading_trailing_whitespace`: 48
- `unicode_normalization_nfkc`: 1

## Chunking

### identity

- Chunks / units: 500 / 500
- Mean / median words: 199.112 / 148.5
- P90 / P95 words: 414.1 / 511.95
- Minimum / maximum words: 4 / 1386
- Multi-chunk units: 0

### fixed_window

- Chunks / units: 945 / 500
- Mean / median words: 119.477 / 152
- P90 / P95 words: 160.0 / 160.0
- Minimum / maximum words: 4 / 160
- Multi-chunk units: 226

### tfidf_boundary

- Chunks / units: 1340 / 500
- Mean / median words: 74.296 / 68.0
- P90 / P95 words: 113.1 / 130.0
- Minimum / maximum words: 4 / 179
- Multi-chunk units: 288

## TF-IDF diagnostics

- Lexical boundaries: 808
- Paragraph boundaries: 0
- Forced-size boundaries: 32
- Fallback units: 6
- Short-tail merges: 224

## Integrity

- Validation: PASS
- Model-content coverage: 100.0%
- Duplicate chunk IDs: 0
- Unresolved parents: 0
- Synthetic marker occurrences: 0
- Train/dev ancestry overlap: 0
- Train/test ancestry overlap: 0
- Dev/test ancestry overlap: 0

## Reproducibility

- Deterministic rebuild: PASS
- Canonical content SHA-256: `f233ef182fc3424dc12bb7abb278f564b38b977917cbd6254d5bf899ab896b4a`
- Output SHA-256 hashes are stored in `phase2_manifest.json`.

## Limitations

- TF-IDF captures lexical similarity, not deep semantic equivalence.
- Regex sentence splitting is deterministic but not linguistically complete.
- Default thresholds are engineering defaults, not optimized values.
- FiQA passages may be shorter and less structured than enterprise manuals.
- Phase 2 does not establish that one chunking strategy is superior.
- Final tuning must use TRAIN/DEV only; TEST must remain untouched.
