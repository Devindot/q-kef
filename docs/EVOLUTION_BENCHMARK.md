# Q-KEF FiQA Evolution Benchmark

## Purpose

FiQA in BEIR is a static information-retrieval benchmark. It can measure whether
a system retrieves relevant source documents, but it does not directly label how
incoming knowledge should affect an already populated knowledge base. Phase 1
therefore constructs controlled sequential scenarios with explicit ground-truth
lifecycle actions. It creates experimental infrastructure only; it does not test
whether Q-KEF or any baseline performs well.

```text
Checksum-verified FiQA -> immutable T0 source references
                       -> deterministic rule generator
                       -> controlled T1 inputs + ground-truth events
```

Original corpus, query, and qrels files are never rewritten by the generator.

## Logical time

- `T0 = 2025-01-01T00:00:00Z`
- `T1 = 2026-01-01T00:00:00Z`

These fixed timestamps model two sequential knowledge-base states. They do **not**
assert that FiQA documents were authored, published, or revised on those dates.

## Stable action semantics

### NEW

Incoming information has no predecessor in the event's T0 state. The expected
behavior is to create one active unit without superseding an existing target.
The incoming text is copied unchanged from an eligible FiQA document.

### REPLACE

Incoming knowledge expresses the same source statement as a T0 unit but changes
one controlled numeric token. The T1 unit is defined as the newer synthetic
version: the T0 target should become superseded and the incoming unit active.
The generator records token type, old and new values, exact character span, and
mutation rule. It changes only that recorded span.

### MERGE

Incoming knowledge retains all base text from source A and appends an exact,
clearly delimited leading segment copied from source B. Both sources are relevant
to the same FiQA query. The expected target is A, and provenance from A and B must
be preserved. Phase 1 labels the desired relationship but does not execute a
merge.

### ARCHIVE

A deterministic administrative notice targets a T0 unit and states that it has
been withdrawn from active use. The notice is synthetic metadata, clearly marked
as such, and contains no new financial claim. It is not a replacement factual
answer. The expected behavior is to archive the specified target.

### COEXIST

T0 source A and unchanged incoming source B are relevant to the same FiQA query,
but neither is defined to supersede the other. Both should remain active. Exact
normalized duplicates and pairs with token Jaccard similarity at or above `0.9`
are rejected to reduce accidental near-duplicates.

### SPLIT

Two unchanged source documents from distinct FiQA queries are joined with an
explicit `KNOWLEDGE SEGMENT BOUNDARY`. Ground truth identifies two deterministic
expected child IDs and the component spans. SPLIT is experimental and may be
analyzed separately because it partly concerns semantic segmentation rather than
pure lifecycle routing.

## Provenance and source protection

Every event records stable event and knowledge IDs, FiQA source document and query
IDs, original qrels split, construction method and parameters, generator version,
seed, timestamps, expected targets or children, and an automatic rule-derived
rationale. T1 knowledge metadata labels text as one of:

- unchanged original FiQA text;
- deterministic numeric mutation;
- deterministic merge construction;
- synthetic administrative notice; or
- deterministic compound split construction.

The downloader verifies the canonical archive MD5 before extraction. Extraction
rejects absolute paths, parent traversal, and symbolic links. The manifest stores
SHA-256 hashes of the source files observed at build time; validation recomputes
them to detect later alteration.

## Split construction and leakage prevention

The default target is 50 events per action, partitioned as 30 train, 10 dev, and
10 test. FiQA train/dev/test qrels primarily supply the matching benchmark split.

A source document is eligible only when its positive qrels membership assigns it
to exactly one of those original splits. Documents referenced across multiple
qrels splits are excluded conservatively. Within each benchmark build, a source
document is used in at most one event. Validation constructs full source ancestry
sets and requires pairwise train/dev/test intersections to be empty. Query overlap
is measured and reported separately.

## Determinism

All selection and IDs derive from sorted source identifiers, the action, split,
generator version, and seed `42`. Candidate ordering uses SHA-256 rather than
process-dependent hashing or filesystem order. No external generative service is
called. Each build regenerates the research content in memory and compares its
canonical content hash before writing. Wall-clock creation time is confined to
the manifest and excluded from this research-content comparison.

## Output files

```text
data/processed/qkef_fiqa_evolution/
├── t0_knowledge.jsonl
├── t1_incoming.jsonl
├── events.jsonl
├── train_events.jsonl
├── dev_events.jsonl
├── test_events.jsonl
├── review_sample.csv
└── benchmark_manifest.json
```

The manifest records requested and actual counts, source and generated-file
hashes, configuration, warnings, leakage state, and deterministic-regeneration
status. Large raw and generated data are ignored by Git.

## Human review

`review_sample.csv` deterministically selects up to five events per action. It
includes source/target IDs, excerpts, method, and automatic rationale. The
`reviewer_action`, `reviewer_valid`, and `reviewer_notes` columns are intentionally
blank. Generation never implies human validation. Researchers should audit the
sample, document corrections, and consider broader review before experiments.

## Commands

```powershell
python scripts/download_fiqa.py
python scripts/build_evolution_benchmark.py
python scripts/validate_evolution_benchmark.py
python scripts/summarize_evolution_benchmark.py
pytest -q
```

The downloader is idempotent. `--force` is reserved for explicitly replacing an
invalid local archive or incomplete extraction. Build, validation, and summary
commands support source/output overrides for controlled tests.

## Limitations

- Temporal updates are controlled/synthetic scenarios, not historical FiQA
  revisions.
- Deterministic numeric changes simplify real enterprise policy evolution and
  contradiction.
- Labels partly follow from construction rules, which makes the benchmark useful
  for controlled tests but not fully representative of naturally occurring data.
- MERGE and COEXIST boundaries can be subjective outside these fixed rules.
- ARCHIVE inputs are artificial administrative notices rather than FiQA prose.
- SPLIT overlaps with chunking and semantic segmentation.
- Rule-based automatic checks do not replace human review.
- Success on this benchmark alone would not demonstrate general enterprise
  performance, novelty, or superiority.
