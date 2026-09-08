# Q-KEF v2 Error Analysis

## Classification boundaries

- **NEW vs COEXIST:** B0 made three COEXIST→NEW and one NEW→COEXIST errors. Q-full made four COEXIST→NEW and two NEW→COEXIST errors. Opaque examples `sample_5616eb9e5f709ae465e2` and `sample_0182c77dc1e3bed42d9d` were high-confidence singleton NEW decisions for COEXIST rows and were therefore unsafe auto-commits rather than quarantines.
- **COEXIST vs MERGE:** no direct B0/Q-full COEXIST↔MERGE TEST confusion occurred, but the hierarchy did not improve COEXIST; structural routing introduced COEXIST→SPLIT and MERGE→SPLIT errors in the conventional hierarchy.
- **REPLACE vs COEXIST:** B0/Q-full had no direct TEST confusion; B2 produced a MERGE→REPLACE error. This small result does not demonstrate general separation.
- **Incorrect cardinality:** hierarchical conventional macro-F1 (0.8991) was below flat B0 (0.9327); hierarchical Q was 0.8756. Explicit structure did not compensate for limited training rows.

## Candidate and uncertainty failures

- Equal-weight RRF hybrid missed three of 30 TEST predecessors at rank five and one at rank ten, while dense retrieved all 30 at rank five. Fusion harmed this benchmark.
- Conformal coverage was 0.95; COEXIST coverage was lowest at 0.80. Twenty-six rows were quarantined because the action set was not a singleton.
- Counterfactual risk did not correct two singleton classifier errors because only NEW was plausible. This shows why calibration coverage and classifier quality remain safety dependencies.

## Risk and policy failures

- Mean current-evidence miss was 0.9096. Deterministic term probes were too broad relative to affected-lineage IDs; this is the largest technical weakness observed.
- Synthetic lower-authority REPLACE is hard-rejected and quarantined. FiQA contains no natural authority evidence.
- Reversed temporal intervals are rejected at record construction; as-of tests verify half-open validity. No natural FiQA temporal conflict is claimed.

## Recovery

Graph, index, certificate, and epoch-stage failures did not publish mixed state. Exact rollback restoration passed. These controlled failures validate code semantics only.
