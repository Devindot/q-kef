# Q-KEF: Quantum-Inspired Knowledge Evolution Framework

Q-KEF is a university research project investigating whether managing the
lifecycle of incoming enterprise knowledge **before final indexing** can improve
retrieval and question answering in knowledge bases that change over time.

## Research motivation

Append-only retrieval-augmented generation (RAG) systems can leave outdated,
duplicated, superseded, or contradictory chunks equally available to retrieval.
Q-KEF proposes a pre-indexing knowledge evolution layer that can represent
lifecycle actions such as `NEW`, `REPLACE`, `MERGE`, `ARCHIVE`, `COEXIST`, and
potentially `SPLIT`. The project asks:

> Can pre-indexing knowledge lifecycle management improve retrieval and answer
> correctness in temporally evolving and contradictory knowledge bases compared
> with append-only RAG?

This is a research hypothesis, not a claim of demonstrated superiority or
novelty. The planned experiments compare append-only RAG, conventional evolution
logic, and the proposed Q-KEF system.

## High-level architecture

```text
Incoming documents -> Ingest -> Chunk -> Embed -> Compare with candidates
                                               -> Context features
Candidates + context -> Quantum-inspired encoder -> Evolution engine
                     -> Evolution-aware vector index + semantic graph
                     -> Hybrid retrieval -> LLM / QA
```

The central contribution under investigation is the pre-indexing knowledge
evolution layer. See [System Architecture](docs/SYSTEM_ARCHITECTURE.md) for the
full conceptual flow and comparison with standard RAG.

## Scientific caveat

“Quantum-inspired” means a **classical simulation** using mathematical ideas
such as normalized state vectors, superposition-style multidimensional
representations, inner products, and fidelity-like similarity. Q-KEF is not a
quantum-computing or quantum-hardware project, and no quantum advantage is
claimed. Whether this representation adds measurable value is explicitly tested
by an ablation experiment.

## Current implementation status

Phase 2 extends the reproducible foundation with model-facing ingestion and
deterministic chunking:

- Python package boundaries and a validated `KnowledgeUnit` schema
- lifecycle and evolution-action enums
- configuration placeholders with project seed `42`
- research scope, hypotheses, dataset plan, architecture, terminology, and a
  three-system experiment plan
- offline environment verification and unit tests
- safe, checksum-verifying and idempotent canonical FiQA acquisition
- a dependency-free BEIR FiQA loader with reference validation
- deterministic ground-truth generation for all six lifecycle actions
- train/dev/test ancestry-leakage controls, provenance and source-file hashes
- benchmark validation, summary, manifest, report, and blank human-review sample
- exact raw/model-text separation with traced conservative normalization
- metadata-driven MERGE, SPLIT, and ARCHIVE scaffold sanitization
- identity, overlapping fixed-window, and lexical TF-IDF boundary chunkers
- opaque chunk IDs, full provenance, coverage/hash checks, and leakage audits

The canonical BEIR archive was acquired with verified MD5
`17918ed23cd04fb15047f73e6c3bd9d9` (57,638 documents; 6,648 queries). The default
build produced all 300 requested events and passed deterministic and leakage
validation. Exact observed statistics are in the
[Phase 1 dataset report](reports/phase1_dataset_report.md).

The Phase 2 real-data build processed all 500 benchmark knowledge units and
produced 500 identity, 945 fixed-window, and 1,340 TF-IDF-boundary chunks with
100% model-content coverage and zero synthetic-marker or cross-split ancestry
leakage. See the [Phase 2 report](reports/phase2_ingestion_chunking_report.md).

No neural semantic model, evolution predictor, vector store, graph integration,
dashboard, LLM pipeline, or performance experiment is implemented yet. Phase 1
rules construct labels; they do not predict them.

## Planned phases

1. **Phase 0 — Foundation (complete):** contracts, reproducibility, research and
   experimental design.
2. **Phase 1 — Controlled data pipeline (complete):** acquire canonical FiQA and
   create separately stored, provenance-preserving temporal scenarios.
3. **Phase 2 — Ingestion and chunking (current):** separate raw/model-facing text
   and create deterministic label-safe chunk corpora.
4. **Phase 3 — Baselines:** implement later authorized representations and/or
   conventional comparison infrastructure.
5. **Phase 4 — Proposed representation:** implement the classical
   quantum-inspired state representation and Q-KEF decision mechanism.
6. **Phase 5 — Evaluation:** run retrieval, answer, evolution, system, and
   ablation experiments.
7. **Phase 6 — Demonstration and reporting:** package results and, if useful,
   build a local interface and optional graph integration.

Later phases remain contingent on experimental design and available time.

## Local setup

Python 3.11 or newer is required. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the offline checks:

```powershell
python scripts/verify_environment.py
pytest -q
python -m compileall -q src scripts tests
```

Acquire and build the Phase 1 benchmark:

```powershell
python scripts/download_fiqa.py
python scripts/build_evolution_benchmark.py
python scripts/validate_evolution_benchmark.py
python scripts/summarize_evolution_benchmark.py
```

See [Evolution Benchmark](docs/EVOLUTION_BENCHMARK.md) for construction rules,
provenance, leakage prevention, and limitations.

Build the Phase 2 chunk corpus:

```powershell
python scripts/build_chunks.py
python scripts/validate_chunks.py
python scripts/summarize_chunks.py
```

See [Ingestion and Chunking](docs/INGESTION_AND_CHUNKING.md) for normalization,
sanitization, chunking, provenance, and label-leakage controls.

## Repository structure

```text
qkef/
├── configs/default.yaml       # Future parameters; unknown thresholds are null
├── data/                      # Raw, interim, processed, and provenance metadata
├── docs/                      # Research and experimental specifications
├── models/                    # Future local model artifacts
├── reports/                   # Future experimental reports
├── scripts/verify_environment.py
├── src/qkef/                  # Source-layout Python package
├── tests/                     # Offline import and schema tests
├── pyproject.toml
└── requirements.txt
```

## Reproducibility

The project seed is `42`. Future generated data and experiments must preserve
source IDs, the effective random seed, timestamps and version metadata,
transformation type, configuration, and generation method. Controlled updates
must be stored separately from unmodified benchmark data.
