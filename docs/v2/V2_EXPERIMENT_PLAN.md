# Q-KEF v2 Experiment Plan

## Confirmatory design

Target 2,400 balanced events: 1,200 TRAIN, 360 DEV, 360 CAL, and 480 TEST. Source ancestry and source-query sets must be disjoint across all four partitions. If those controls cannot support the target, the maximum defensible balanced size will be reported without duplication.

## Comparisons

- candidates: dense, lexical, deterministic RRF hybrid;
- models: flat conventional/Q and hierarchical conventional/Q;
- matched features: B0, PCA-16, matched 10-dimensional non-Q PCA controls, fidelity, entropy, coherence, Q-full;
- execution: direct, conformal-gated, counterfactual retrieval-safe;
- state: append-only, v1 evolution, v2 epoch-coherent evolution;
- failures: graph, index, certificate, and publication stages.

Metrics include candidate Recall@1/3/5/10 and MRR; lifecycle accuracy, balanced accuracy, macro/weighted F1 and per-class scores; conformal coverage/set size/quarantine; stale exposure/current miss; p50/p95 latency; index size/churn; memory; mixed epochs; and rollback equality. Paired bootstrap, exact McNemar, Wilson intervals, and Holm correction are used where appropriate.

The implemented retrospective pilot reserves 60 TRAIN-derived CAL rows, uses 120 TRAIN and 60 DEV rows for fitting/selection, and reports the untouched existing 60 TEST rows only as exploratory v2 evidence.
