# Final Project Status

- Phase 0 ✅ Research/software foundation
- Phase 1 ✅ Reproducible FiQA evolution benchmark
- Phase 2 ✅ Provenance-safe ingestion and chunking
- Phase 3 ✅ Complete frozen research backend and experiment
- Phase 4 ✅ Final application, runtime, QA, statistics, figures, documentation, and release validation

## Architecture

MiniLM candidate discovery feeds fair conventional/Q-KEF lifecycle models, an evolution-aware exact vector KB, and a NetworkX lineage graph. Streamlit loads trusted frozen artifacts and provides demo, comparison, retrieval, graph, Q-state, QA, result, error, and methodology views.

## Dataset and models

FiQA: 57,638 documents/6,648 queries. Controlled benchmark: 300 events, 180/60/60. B: balanced logistic regression C=0.1. C: same family C=1.0 plus TRAIN-only PCA-16 state features.

## Metrics

Candidate R@5 0.9600. TEST macro F1 B/C 0.8455/0.8933. Obsolete@5 append-only/Q-KEF 0.3393/0.1180. Exact post-hoc uncertainty is in `reports/final/STATISTICAL_ANALYSIS.md`.

## Application and reproducibility

```powershell
pip install -r requirements.txt
python scripts/final_release_check.py
streamlit run app.py
pytest -q
```

No LLM, paid API, cloud vector store, graph server, quantum hardware, or quantum simulator is required. Controlled updates and limitations are disclosed.

## Tests and warnings

Final suite: 90 passed. Release checker: PASS. Dependency check: PASS. Streamlit returned HTTP 200 and was terminated cleanly. Known non-blocking warnings: three safe SPLIT fallbacks per learned/oracle simulation and external Joblib/NumPy deprecations.

## Final status

Final Academic Release: ready for academic demonstration, report preparation, viva, and reproducible inspection within documented limitations.
