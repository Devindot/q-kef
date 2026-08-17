# Phase 1 Dataset Report

This report describes generated benchmark infrastructure and integrity checks; it contains no model-performance results.

## Source

- Expected FiQA MD5: `17918ed23cd04fb15047f73e6c3bd9d9`
- Observed FiQA MD5: `17918ed23cd04fb15047f73e6c3bd9d9`
- Checksum verified: `True`
- Source corpus documents: 57638
- Source queries: 6648

## Generated benchmark

- Events: 300
- T0 knowledge units: 200
- T1 incoming units: 300
- Unique FiQA source documents: 450
- Human-review sample rows: 30

### Events by action

- NEW: 50
- REPLACE: 50
- MERGE: 50
- ARCHIVE: 50
- COEXIST: 50
- SPLIT: 50

### Events by split

- train: 180
- dev: 60
- test: 60

### T1 text-length statistics (characters)

- Minimum: 123
- Mean: 1152.31
- Median: 853.50
- Maximum: 7792

### REPLACE mutation-token distribution

- currency: 20
- number: 22
- percent: 4
- year: 4

## Eligibility and integrity

- Eligible single-split source documents: 17110
- Unreferenced/non-positive source documents excluded: 40528
- Cross-qrels-split documents excluded: 0
- Empty-text documents excluded: 38
- Duplicate/near-duplicate candidate pairs rejected: 0
- Validation: PASS
- Train/dev source overlap: 0
- Train/test source overlap: 0
- Dev/test source overlap: 0
- Train/dev query overlap: 0
- Train/test query overlap: 0
- Dev/test query overlap: 0
- Deterministic regeneration: PASS

## Warnings and shortfalls

- None.

The temporal events are controlled synthetic scenarios, not historical FiQA revisions. Human-review fields remain blank pending manual audit.
