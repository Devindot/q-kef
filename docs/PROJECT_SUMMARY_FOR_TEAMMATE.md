# Project Summary for Teammate

## Phases

- Phase 0: schemas, configuration, research framing.
- Phase 1: deterministic 300-event FiQA evolution benchmark.
- Phase 2: provenance-safe ingestion and three chunk strategies.
- Phase 3: MiniLM, candidates, B/C models, Q-state, KB/graph simulation, frozen evaluation.
- Phase 4: trusted runtime, Streamlit, deterministic QA, statistics, supplementary metrics, figures, docs, release checks.

## Architecture and I/O

Incoming sanitized text is embedded with cached MiniLM, compared against T0 candidates, converted into safe conventional and optional Q-state features, and classified into one of six actions. Execution updates an active vector index and lineage graph. QA extracts sentences from active evidence. Interactive outputs never contain benchmark ground truth.

## Important paths

- `app.py`: final application
- `src/qkef/runtime`: trusted loader/service/demo/results
- `src/qkef/qa`: extractive QA
- `models/phase3`: frozen trusted models
- `reports/phase3`: frozen experiment
- `reports/final`: final analysis, figures, tables, manifest
- `data/demo`: six disclosed demo cases
- `docs`: demo, viva, reproducibility, claims, architecture, limits

## Dataset, models, results

FiQA has 57,638 documents and 6,648 queries; the controlled benchmark has 300 events. B is balanced logistic regression C=0.1; Q-KEF C uses C=1.0 and a TRAIN-only 16D PCA state. TEST macro F1 is 0.8455 vs 0.8933. Append-only vs Q-KEF obsolete@5 is 0.3393 vs 0.1180.

## Run and demo

Install requirements, run `python scripts/final_release_check.py`, then `streamlit run app.py`. Follow `FINAL_DEMO_GUIDE.md` and start with the stored REPLACE case.

## Rebuild

The app requires no retraining. `run_final_analysis.py` regenerates statistics/figures from saved predictions. Full earlier-phase rebuild commands are in `REPRODUCIBILITY.md`.

## Do not claim

Do not claim actual quantum computation, quantum advantage, universal improvement, hallucination elimination, production readiness, historical temporal labels, financial advice, or patent status.

## Known limitations

Controlled/synthetic updates, 60 TEST events, generic MiniLM, weak COEXIST, three SPLIT fallbacks per system, and provenance-dependent retrieval relevance.
