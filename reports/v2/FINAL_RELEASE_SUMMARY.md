# Q-KEF v2 Final Release Summary

Q-KEF v2 delivers an end-to-end research prototype for retrieval-safe knowledge evolution: typed lifecycle transitions, ancestry-isolated data partitions, bitemporal and authority-aware records, conventional and Q-inspired ablations, conformal gating, counterfactual risk selection, atomic graph/index publication, rollback certificates, and failure-injection validation.

## Confirmatory evidence

The locked experiment used a balanced 2,400-event synthetic benchmark with 1,200 TRAIN, 360 DEV, 360 CALIBRATION, and a single 480-event TEST evaluation. Source ancestry and query/document identities are disjoint across partitions. Model binaries, configuration, benchmark, and split hashes were frozen before TEST access.

- Dense Recall@5 was 0.8958 versus 0.8875 for the locked hybrid.
- Conventional B0 macro-F1 was 0.8791, Q-full was 0.8708, matched non-Q was 0.8517, and hierarchical Q-full was 0.8121.
- Q-full minus B0 was -0.0083 with paired-bootstrap 95% CI [-0.0250, 0.0078] and Holm-adjusted p=1.0000.
- Q-full minus matched non-Q was +0.0191 with paired-bootstrap 95% CI [-0.0031, 0.0423] and exact McNemar p=0.1496; Q-specific predictive value was not established.
- Conformal marginal coverage was 0.7792 against a nominal 0.90 target; ARCHIVE coverage was 0.0.
- Counterfactual selective execution committed 281/480 transitions with precision 0.9466 and quarantine rate 0.4146.
- Graph/index consistency was 1.0000, mixed-epoch exposure was zero, and all seven injected failure stages restored the prior state.
- The active index finished at 455 records versus an 800-record append-only comparator.

## Conclusion

The confirmatory evidence supports the transactional safety and bounded-index mechanisms under the controlled benchmark. It does not support hybrid retrieval superiority, hierarchical classifier superiority, or an advantage of Q-full over the strongest conventional baseline. The conformal component failed its coverage objective and requires redesign before any production claim.

The machine-verifiable release is complete. Independent human review remains explicitly pending: a deterministic 120-row stratified audit sample, protocol, and status receipt are included, but reviewer identity, judgment, date, and signature must come from a real reviewer.

This release does not claim quantum computational advantage, production readiness, or legal patentability.
