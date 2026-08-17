# Phase 2 Ingestion and Deterministic Chunking

## Purpose and boundary

Phase 2 converts the controlled Phase 1 evolution benchmark into reproducible
text units suitable for later representation, candidate matching, and retrieval
experiments. It implements no neural embedding, vector index, lifecycle
classifier, quantum-inspired encoder, or answer generation.

```text
Phase 1 KnowledgeUnit
        |
        v
Schema-aware ingestion
        |
        +---- exact raw_text (reconstruction and audit)
        |
        +---- conservative model_text (future model input)
                         |
                         v
           identity / fixed_window / tfidf_boundary
```

A knowledge unit remains the lifecycle-reasoning object. A chunk is a derived
segment belonging to exactly one knowledge unit; chunking does not change or
predict an event label.

## Raw text and model-facing text

`IngestedKnowledgeUnit` stores exact Phase 1 `raw_text` and a separate
`model_text`. SHA-256 hashes cover each form. It also records temporal state
(`T0` or `T1`), split, source IDs, event IDs, timestamp, version, lifecycle
status, transformation operations, and structured provenance.

Raw text is never overwritten. Model-facing text is the only text supplied to
the chunkers. Identifiers and provenance are internal traceability fields and
must not be naively encoded as learned features in later phases.

## Conservative normalization

Normalization is deterministic and records an operation only when it changes the
text:

1. Unicode NFKC normalization;
2. CRLF/CR line endings to LF;
3. zero-width character removal;
4. repeated horizontal whitespace to one space;
5. excessive blank-line reduction;
6. leading/trailing whitespace trimming.

It does not lowercase, stem, remove punctuation, remove numbers or currency,
discard negation, or paraphrase. A configurable removal-ratio guard rejects large
unexplained reductions.

## Metadata-driven sanitization

Sanitization checks Phase 1 `text_origin`; it does not blindly delete matching
phrases from original FiQA text.

- **MERGE:** removes the generated `Additional related information:` header while
  preserving the complete base text, copied complementary source segment, and
  paragraph separation.
- **SPLIT:** replaces the generated `--- KNOWLEDGE SEGMENT BOUNDARY ---` marker
  with a neutral paragraph boundary. It neither uses the expected SPLIT action as
  a chunking input nor treats the marker as a privileged split location.
- **ARCHIVE:** extracts the generated `Target knowledge ID: ...` value into
  structured provenance and removes the literal ID line. The withdrawal and
  “no longer active” semantics remain.
- **REPLACE:** retains the controlled new numeric value. It never restores the
  original value.
- **NEW/COEXIST:** retain original FiQA content, subject only to conservative
  normalization.

The Phase 2 audit confirms known scaffold strings are absent from model text and
chunks while raw text still preserves them.

## Chunk record and identifiers

`KnowledgeChunk` records an opaque chunk ID, internal parent knowledge/event
provenance, split, temporal state, strategy, contiguous index, exact text and
hash, word/character/sentence counts, exact model-character span where available,
source IDs, timestamp/version/status, overlap data, and strategy diagnostics.

Chunk IDs are `chk_` plus 24 hexadecimal characters from SHA-256 of strategy,
parent ID, and chunk index. Visible IDs therefore expose neither action nor split
names even though the separate parent mapping preserves full traceability.
Ground-truth fields such as `expected_action`, `class_label`, and
`expected_relation` are absent from the chunk schema.

## Word and sentence rules

“Word count” means a deterministic whitespace-delimited lexical-unit count; it
is not an LLM tokenizer count. The same rule drives windowing, statistics, and
validation.

Sentence splitting uses conservative regex boundaries after `.`, `?`, or `!`
when followed by whitespace/end-of-text, plus natural paragraph gaps. A decimal
such as `10.50` is not split at its internal period. The method is intentionally
lightweight and does not claim complete linguistic segmentation.

## Strategies

### Identity

One knowledge unit produces exactly one chunk equal to `model_text`. It is the
baseline for lifecycle classification, ablation, and debugging.

### Fixed window

Whitespace-delimited words are chunked with a default maximum of 160 and overlap
of 30. Windows preserve original character substrings and word order, never cross
parents, permit a short final window, and record exact word and character ranges.

### TF-IDF boundary

The lexical strategy operates independently inside each knowledge unit:

1. partition paragraphs and sentences;
2. construct deterministic classical TF-IDF vectors for the unit's sentences;
3. compute adjacent cosine similarities;
4. split on low similarity only after minimum-size conditions;
5. force size boundaries at at most 180 words;
6. merge a short tail into the preceding chunk when the maximum remains valid.

Defaults are minimum 50, target 120, maximum 180 words, and similarity threshold
0.15. These are Phase 2 engineering defaults, not optimized research values.
Future tuning must use TRAIN/DEV only; TEST must not guide parameter selection.
One-sentence and empty-vocabulary cases fall back safely, and oversized sentences
use deterministic word-boundary subdivision.

TF-IDF here is lexical similarity on conventional CPU computation. It is not a
neural semantic embedding.

## Provenance, coverage, and leakage validation

Every chunk resolves through its parent unit and event IDs to Phase 1 and FiQA
source IDs, including both parents for MERGE/SPLIT T1 units. Validation checks:

- unique opaque IDs and contiguous per-parent indexes;
- parent, split, temporal, event, and multi-source provenance consistency;
- counts, hashes, spans, non-empty text, and known strategy names;
- exact identity equality;
- complete ordered fixed-window word coverage accounting for overlap;
- contiguous TF-IDF character spans reconstructing all model text;
- action-specific MERGE/SPLIT/REPLACE/ARCHIVE content preservation;
- no forbidden label fields or known scaffolding;
- independently recomputed train/dev/test source ancestry intersections;
- generated-file hashes and deterministic rebuild signature.

Ground-truth events remain in the separate Phase 1 table. Future feature pipelines
must explicitly select safe fields rather than encode knowledge/event identifiers.

## Outputs and commands

```text
data/processed/qkef_fiqa_chunks/
├── ingested_t0.jsonl
├── ingested_t1.jsonl
├── chunks_identity.jsonl
├── chunks_fixed_window.jsonl
├── chunks_tfidf_boundary.jsonl
├── unit_chunk_map.jsonl
├── sanitization_audit.csv
└── phase2_manifest.json
```

```powershell
python scripts/build_chunks.py
python scripts/validate_chunks.py
python scripts/summarize_chunks.py
pytest -q
```

## Limitations

- TF-IDF measures lexical overlap rather than deep semantic equivalence.
- Regex sentence boundaries are not linguistically perfect.
- Default thresholds have not been optimized for downstream retrieval.
- FiQA passages can be shorter and less structured than enterprise manuals.
- Chunking is infrastructure, not a claimed research contribution or proof that
  one strategy is superior.
- Final strategy selection requires downstream TRAIN/DEV evidence while keeping
  TEST untouched.
