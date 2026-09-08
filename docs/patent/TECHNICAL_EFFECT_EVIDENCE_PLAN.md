# Technical-Effect Evidence Plan

| Mechanism | Metric | Experiment | Artifact |
|---|---|---|---|
| Lifecycle retirement | obsolete exposure, active index size | append-only vs v1 vs v2 | `reports/v2/v2_results.json` |
| Candidate-state witness retrieval | OE/CM per transition | direct vs counterfactual | `execution_decisions.csv` and future witness traces |
| Hybrid candidates | Recall@1/3/5/10, MRR | dense vs lexical vs RRF | `v2_results.json` |
| Uncertainty gate | coverage, set size, quarantine, precision | CAL-frozen conformal TEST | `v2_results.json` |
| Epoch publication | consistency, mixed exposures | success + seven injected failures | tests and `TECHNICAL_EFFECT_RESULTS.md` |
| Rollback delta | exact hash equality, recovery time | commit then rollback | tests; future timing trace |
| Delta index update | churn, embeddings regenerated | per-transition instrumentation | `v2_results.json` |
| Retrieval/index effects | p50/p95 latency, memory | local controlled workload | `v2_results.json` |
| Feature representation | macro-F1/COEXIST, paired CI | B0/B1/B2/Q variants | `STATISTICAL_ANALYSIS.md` |

The future confirmatory experiment must retain raw per-query/per-transition observations so uncertainty intervals can be computed for every technical effect.
