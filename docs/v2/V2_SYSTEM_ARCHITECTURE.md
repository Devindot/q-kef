# Q-KEF v2 System Architecture

```text
incoming knowledge + provenance
        ↓
hybrid predecessor candidates
        ↓
flat/Q-optional probabilities → conformal action set
        ↓
cardinality-aware TransitionPlan per plausible action
        ↓
ShadowStateView overlays on S_t=(G_t,I_t,E_t)
        ↓
content-derived witness retrieval + risk + hard invariants
        ↓
unique safe action? ── no → QUARANTINE
        │ yes
        ↓
prepare graph/index/certificate → atomic epoch publication
        ↓
certificate + compact rollback delta
```

`qkef.v2` is separate from v1. `KnowledgeState` stores explicit records, lineage edges, searchable index entries, epoch ID, and transition ID. A shadow view overlays additions, updates, removals, and edges without modifying the live object. `EpochTransactionManager` prepares a complete next state and exposes it only through one pointer publication after graph, index, invariants, and certificate succeed.

Normal retrieval uses `RetrievalState.ACTIVE`; historical retrieval can include `HISTORICAL_ONLY` subject to valid/system time. QUARANTINED and BLOCKED records are excluded. Quantum-inspired features are optional inputs to probability estimation; all safety mechanisms work without them.
