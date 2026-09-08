# Q-KEF v2 Hypotheses

- **H1:** Hybrid candidate retrieval improves predecessor Recall@k over dense-only retrieval.
- **H2:** Hierarchical lifecycle modelling improves macro-F1 and/or COEXIST performance over a flat baseline.
- **H3:** Q-specific descriptors may add value beyond matched PCA/non-Q controls; this is not assumed.
- **H4:** Counterfactual validation reduces unsafe execution relative to direct classifier execution.
- **H5:** Epoch-synchronized publication prevents persistent graph/index inconsistency under injected failures.
- **H6:** Lifecycle-aware active retrieval reduces obsolete exposure and active search space relative to append-only retrieval.
- **H7:** Validation imposes measurable latency, memory, and quarantine costs.

Each null outcome is reportable. DEV selects configurations, CAL estimates conformal thresholds, and TEST is evaluated only after locking. The current 60-row retrospective pilot is underpowered and cannot serve as final confirmatory evidence.
