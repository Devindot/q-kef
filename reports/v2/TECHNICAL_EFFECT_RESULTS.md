# Q-KEF v2 Technical-Effect Results

All measurements are local academic retrospective-pilot results.

| Metric | Result |
|---|---:|
| Committed / evaluated transitions | 34 / 60 |
| Quarantine rate | 0.4333 |
| Auto-commit precision | 0.9412 |
| Initial active index | 40 |
| Final active index | 30 |
| Append-only comparator | 100 |
| Search-space reduction vs append-only | 0.7000 |
| Historical records | 48 |
| Embeddings/index additions produced | 26 |
| Mean index churn | 0.0585 |
| Graph/index consistency | 1.0000 |
| Mixed-epoch exposures | 0 |
| Exact rollback recovery | approximately 0.063 ms on recorded micro-run |
| Mean witness obsolete exposure | 0.0000 |
| Mean witness current-evidence miss | 0.9096 |
| Update latency p50 / p95 | approximately 236 / 300 ms on recorded run |
| Retrieval latency p50 / p95 | approximately 0.53 / 0.55 ms on recorded run |
| Peak traced Python memory | approximately 2.51 MB |

Seven failure stages were tested. Every injected failure retained epoch 12, preserved graph/index equivalence, and restored the exact original state hash. A committed REPLACE advanced epoch 12→13; its compact rollback delta restored epoch 12 and the exact hash. These tests concern the local in-memory transaction implementation, not distributed durability.

The high current-evidence miss is a negative result: the first deterministic witness generator often retrieves related but not affected-lineage evidence. It should be improved on TRAIN/DEV or real historical workloads before confirmatory TEST evaluation.
