# Q-KEF v2 Hypotheses

- **H1:** Hybrid candidate retrieval improves predecessor Recall@k over dense-only retrieval.
- **H2:** Hierarchical lifecycle modelling improves macro-F1 and/or COEXIST performance over a flat baseline.
- **H3:** Q-specific descriptors may add value beyond matched PCA/non-Q controls; this is not assumed.
- **H4:** Counterfactual validation reduces unsafe execution relative to direct classifier execution.
- **H5:** Epoch-synchronized publication prevents persistent graph/index inconsistency under injected failures.
- **H6:** Lifecycle-aware active retrieval reduces obsolete exposure and active search space relative to append-only retrieval.
- **H7:** Validation imposes measurable latency, memory, and quarantine costs.

Each null outcome is reportable. DEV selects configurations, CAL estimates conformal thresholds, and TEST is evaluated only after locking. The current 60-row retrospective pilot is underpowered and cannot serve as final confirmatory evidence.

## Confirmatory outcomes

- H1: not supported; dense Recall@5 0.8958 exceeded locked hybrid 0.8875.
- H2: not supported; hierarchical Q macro-F1 0.8121 was below flat Q-full 0.8708.
- H3: not established; Q-full exceeded matched non-Q by 0.0191 macro-F1, but paired McNemar p=0.1496 and Q-full remained below B0.
- H4: descriptively supported with selectivity; safe auto-commit precision 0.9466 exceeded direct Q-full accuracy 0.8708 at 58.5% coverage.
- H5: supported for the local implementation; all seven failure stages preserved the prior epoch and graph/index consistency.
- H6: supported on the controlled workload; active index size was 455 versus append-only 800.
- H7: supported; validation cost was measurable, with local p50/p95 update latency about 2.98/4.22 seconds and quarantine rate 0.4146.

The conformal procedure failed its nominal marginal-coverage target (0.7792 observed versus 0.90 nominal), so calibration adequacy is a negative confirmatory result.
