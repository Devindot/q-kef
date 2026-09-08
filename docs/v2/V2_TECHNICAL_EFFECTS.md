# Technical Effects Under Evaluation

The software measures stale exposure, current-evidence miss, active/historical index size, search-space reduction, update/retrieval latency, index churn, embeddings regenerated, traced memory, graph/index consistency, mixed-epoch exposure, quarantine, auto-commit precision, query availability under failure, and rollback equality.

In the local retrospective pilot, 34/60 transitions auto-committed, consistency remained 100%, mixed-epoch exposures were zero, and active index size was 30 versus an append-only comparator size of 100. The measured 70% search-space reduction and latency values are workload- and laptop-specific. Witness obsolete exposure was zero, while current-evidence miss was high (about 0.91), exposing a weakness in the simple content-derived witness/lineage relevance test that requires improvement rather than concealment.

Failure-injection tests establish local transaction semantics, not distributed availability or production durability.
