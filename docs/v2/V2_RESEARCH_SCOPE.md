# Q-KEF v2 Research Scope

Q-KEF v2 investigates a retrieval-safe lifecycle-transition architecture. Its question is whether an uncertain lifecycle prediction can be prevented from corrupting live retrieval state by inserting a deterministic plan–simulate–validate–commit protocol.

## Research questions

1. Does hybrid predecessor resolution improve Recall@k over dense-only retrieval?
2. Does cardinality-aware modelling improve difficult classes such as COEXIST?
3. Do Q-inspired descriptors add information beyond matched PCA and non-Q controls?
4. Does counterfactual validation reduce obsolete exposure or current-evidence miss?
5. Does epoch publication prevent mixed graph/index state under injected failure?
6. What search-space reduction, latency, memory, churn, and quarantine costs result?

The v1 300-event results remain a frozen scientific baseline. Q-KEF v2 completed a separate leakage-isolated 2,400-event confirmatory benchmark with 480 TEST events and a one-time locked evaluation. Dense retrieval and flat B0 remained the strongest tested retrieval/model configurations; the optional Q-full representation did not outperform B0.

The primary architectural contribution under investigation is retrieval-safe knowledge evolution. The quantum-inspired representation is an optional feature representation evaluated through matched ablation. No quantum advantage, production generality, novelty, or patentability is assumed.
