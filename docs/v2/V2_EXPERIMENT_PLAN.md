# Q-KEF v2 Experiment Plan

## Confirmatory design

Target 2,400 balanced events: 1,200 TRAIN, 360 DEV, 360 CAL, and 480 TEST. Source ancestry and source-query sets must be disjoint across all four partitions. If those controls cannot support the target, the maximum defensible balanced size will be reported without duplication.

Execution status: the full target was deterministically generated with 400 events per action and zero pairwise query/document ancestry overlap. Models and all configuration choices were locked from TRAIN/DEV/CAL before TEST access. The 480-event TEST partition was then evaluated exactly once; the post-test lock receipt and complete predictions are preserved under `reports/v2/confirmatory/`. Human review of the controlled synthetic scenarios remains pending.

## Comparisons

- candidates: dense, lexical, deterministic RRF hybrid;
- models: flat conventional/Q and hierarchical conventional/Q;
- matched features: B0, PCA-16, matched 10-dimensional non-Q PCA controls, fidelity, entropy, coherence, Q-full;
- execution: direct, conformal-gated, counterfactual retrieval-safe;
- state: append-only, v1 evolution, v2 epoch-coherent evolution;
- failures: graph, index, certificate, and publication stages.

Metrics include candidate Recall@1/3/5/10 and MRR; lifecycle accuracy, balanced accuracy, macro/weighted F1 and per-class scores; conformal coverage/set size/quarantine; stale exposure/current miss; p50/p95 latency; index size/churn; memory; mixed epochs; and rollback equality. Paired bootstrap, exact McNemar, Wilson intervals, and Holm correction are used where appropriate.

The earlier retrospective pilot remains frozen as exploratory evidence. Its 60-row TEST result is not pooled with or substituted for the ancestry-isolated confirmatory result.
