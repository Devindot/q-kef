# Q-KEF

> **Q-KEF v2 final release:** the `qkef-v2-retrieval-safe-evolution` branch adds a retrieval-safe plan–simulate–validate–commit architecture while preserving the v1 scientific release. The ancestry-isolated 2,400-event confirmatory experiment is complete. Start with [V2_PROJECT_STATUS.md](V2_PROJECT_STATUS.md) and run `streamlit run app_v2.py`. Results do not establish patentability, production readiness, or quantum advantage.

Quantum-Inspired Knowledge Evolution Framework for Dynamic Enterprise Semantic Graphs

**Project status: Final Academic Release**

## Overview

Q-KEF is a reproducible academic prototype that places a Knowledge Evolution layer before final retrieval indexing. It identifies whether incoming knowledge should be added, replace existing knowledge, merge, archive, coexist, or split, then preserves lifecycle and provenance in a local vector knowledge base and semantic graph.

## Problem

Append-only retrieval-augmented systems can expose outdated, superseded, duplicate, or contradictory records with equal status. Q-KEF studies whether pre-indexing lifecycle management reduces obsolete retrieval while retaining auditable history.

## Research Question

Can a six-action pre-indexing lifecycle layer improve retrieval over append-only storage, and does a classical quantum-inspired state augmentation add measurable value beyond conventional evolution features?

## Architecture

`Incoming text → sanitize → MiniLM → candidate search → conventional + optional Q-state features → lifecycle classifier → evolution-aware vector KB + graph → evidence retrieval`

The Streamlit application loads frozen artifacts. It does not train or tune models at startup.

## Dataset

The source is canonical BEIR FiQA: 57,638 documents and 6,648 queries. A controlled temporal benchmark contains 300 events—50 per action—with 180 TRAIN, 60 DEV, and 60 held-out TEST events. Temporal changes are controlled/synthetic, not historical enterprise updates.

## Evolution Actions

- `NEW`: add active knowledge.
- `REPLACE`: supersede a predecessor.
- `MERGE`: create a deterministic representation with both provenance branches.
- `ARCHIVE`: remove a target from active retrieval while retaining audit history.
- `COEXIST`: keep both records active and linked.
- `SPLIT`: create independently active children when segmentation is usable.

## Quantum-Inspired Component

MiniLM embeddings are transformed by a TRAIN-only 16-dimensional PCA basis and L2-normalized. Fidelity-like similarity, squared-amplitude entropy, coherence-like descriptors, and Hellinger distance are classical engineered features. No quantum hardware, simulator, circuit, qubit, or quantum advantage is involved.

## Experimental Systems

- A: append-only retrieval baseline.
- B: conventional evolution using safe content and candidate features.
- C: Q-KEF using the same candidates/classifier family plus quantum-inspired features.
- Oracle: descriptive lifecycle upper-bound state.

## Results

Candidate Recall@5 is 0.9600. On held-out TEST, B achieved 0.8455 macro F1 and C achieved 0.8933, a descriptive +0.0479 absolute difference. Obsolete retrieval@5 was 0.3393 for append-only and 0.1180 for Q-KEF. See [the final experiment report](reports/FINAL_EXPERIMENT_REPORT.md) for uncertainty and limitations.

### Q-KEF v2 confirmatory results

The 2,400-event confirmatory benchmark used 1,200 TRAIN, 360 DEV, 360 CALIBRATION, and one 480-event TEST evaluation. Dense Recall@5 was 0.8958. B0 achieved 0.8791 macro-F1 and Q-full achieved 0.8708; Q-full did not beat B0. Safe auto-commit precision was 0.9466 on 281/480 transitions, graph/index consistency was 1.0, and all seven failure stages preserved the prior epoch. Conformal coverage was only 0.7792 versus the nominal 0.90 target, a retained negative result. See [the confirmatory report](reports/v2/confirmatory/CONFIRMATORY_EXPERIMENT_REPORT.md).

## Installation

Verified with Python 3.12.13.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the App

```powershell
python scripts/final_release_check.py
streamlit run app.py
```

After dependencies and the MiniLM cache are present, no internet, paid API, Neo4j, Pinecone, or LLM is required.

## Reproducing Experiments

The final app consumes frozen Phase 3 models and reports. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for source acquisition, Phase 1–3 rebuild commands, and the non-training Phase 4 analysis.

## Testing

```powershell
python -m compileall -q src scripts tests app.py
pytest -q
pip check
python scripts/v2_release_check.py
```

## Repository Structure

- `src/qkef/`: ingestion, evolution, runtime, retrieval, graph, QA, and analysis packages.
- `data/demo/`: disclosed benchmark-backed demo cases.
- `models/phase3/`: trusted frozen model artifacts generated locally.
- `reports/phase3/`: frozen scientific outputs.
- `reports/final/`: statistics, supplementary analyses, figures, tables, results, and release manifest.
- `reports/v2/confirmatory/`: locked pre/post-TEST receipts, predictions, statistics, technical effects, and final figures.
- `docs/`: architecture, claims, demo, viva, reproduction, responsible-use, and handoff guides.
- `app.py`: final Streamlit entry point.

## Limitations

The benchmark is modest, controlled, finance-domain, and based on synthetic temporal transformations. MiniLM is generic; COEXIST remains difficult; SPLIT overlaps with segmentation; retrieval relevance is provenance-dependent; wider enterprise and human evaluation are required.

## Responsible Use

Predictions require human review before lifecycle changes. Preserve provenance and audit history; never treat ARCHIVE as authorization for irreversible physical deletion. The evidence demo is not financial advice and is not production validated.

## Quantum Disclaimer

All quantum-inspired calculations are classical linear algebra. Q-KEF does not demonstrate or claim quantum advantage.

## Project Status

Phases 0–4 and the Q-KEF v2 confirmatory release are complete. Independent human sign-off on the prepared v2 audit sample remains external to the software release.
