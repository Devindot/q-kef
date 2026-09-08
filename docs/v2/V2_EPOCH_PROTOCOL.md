# Epoch Publication and Rollback Protocol

For committed state `S_t=(G_t,I_t,E_t)`, a safe plan targets `E_t+1`.

1. Prepare graph records and lineage edges in a non-live view.
2. Prepare the retrieval-index delta in that same view.
3. Validate `ACTIVE_G(v,e) ⇔ SEARCHABLE_I(v,e)`, lineage, temporal rules, risk, and hashes.
4. Build a deterministic certificate payload.
5. Publish the fully prepared state as one in-memory reference update.
6. Retain a compact rollback delta containing prior changed records/index entries and added/removed edges.

Failures at `BEFORE_GRAPH_PREPARE`, `AFTER_GRAPH_PREPARE`, `BEFORE_INDEX_PREPARE`, `AFTER_INDEX_PREPARE`, `BEFORE_CERTIFICATE`, `BEFORE_EPOCH_COMMIT`, or `AFTER_EPOCH_COMMIT_METADATA` restore the old reference and hash. Tests assert no mixed epoch is readable. Certificate files use temporary write, flush/fsync, and `os.replace`.

This local academic transaction model is real and testable but not a substitute for a distributed production commit protocol.
