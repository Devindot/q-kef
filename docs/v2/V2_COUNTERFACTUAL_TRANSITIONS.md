# Counterfactual Lifecycle Transitions

`TransitionPlanner` maps NEW (0→1), one-to-one REPLACE/ARCHIVE/COEXIST, MERGE (m→1), and SPLIT (1→r) to deterministic record, lineage, and index deltas. SPLIT requires at least two caller-supplied meaningful segments and never invents child text.

Each `TransitionPlan` contains a content-derived transition ID, predecessor IDs, record additions/updates, graph edges, index additions/removals, pre/postconditions, affected lineages, estimated churn, and configuration hash. `ShadowStateView` applies these as a read-only overlay. Materialization creates a separate candidate state; it does not mutate the committed base.

Witness queries default to `STRICT_CONTENT_DERIVED`: deterministic high-frequency terms from incoming/candidate content. Runtime input rejects benchmark-only action, target, mutation, relation, and split metadata. The evaluator measures obsolete exposure, current-evidence miss, index churn, cross-view, lineage, temporal, and authority violations. Consistency, lineage, temporal, authority, and plan preconditions are hard rejection conditions.
