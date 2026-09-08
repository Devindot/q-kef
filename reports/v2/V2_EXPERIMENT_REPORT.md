# Q-KEF v2 Retrospective Pilot Report

## Status and integrity

This is a Q-KEF v2 retrospective architecture pilot on the frozen 300-event v1 benchmark. It does not replace the Q-KEF v1 frozen experiment and is not the planned 2,400-event confirmatory v2 benchmark. The split is 120 TRAIN, 60 DEV, 60 TRAIN-derived CAL, and the existing 60 TEST. DEV selected C; CAL fixed class-conditional conformal quantiles; TEST was not used for tuning.

## Candidate retrieval

On 30 eligible TEST events, dense/lexical/equal-weight RRF hybrid Recall@5 were 1.000/0.733/0.900 and MRR 0.903/0.744/0.825. Equal-weight RRF therefore failed H1 on this pilot and must not be TEST-tuned. A future hybrid must be selected on DEV and locked.

## Lifecycle models and matched ablation

| Model | TEST accuracy | TEST macro-F1 | COEXIST F1 |
|---|---:|---:|---:|
| B0 conventional | 0.9333 | 0.9327 | 0.7778 |
| B1 PCA-16 | 0.8167 | 0.8198 | 0.5263 |
| B2 matched non-Q | 0.8833 | 0.8860 | 0.7273 |
| Q fidelity | 0.9333 | 0.9327 | 0.7778 |
| Q entropy | 0.9333 | 0.9327 | 0.7778 |
| Q coherence | 0.9333 | 0.9327 | 0.7778 |
| Q full | 0.9000 | 0.8990 | 0.6667 |
| Hierarchical conventional | 0.9000 | 0.8991 | 0.7778 |
| Hierarchical Q | 0.8833 | 0.8756 | 0.5333 |

Q-full minus B0 macro-F1 was −0.0337; its paired bootstrap 95% interval was [−0.0876, 0.0000], exact McNemar p=0.5, Holm-adjusted p=1.0. Q-full exceeded matched non-Q by 0.0130 but this does not establish Q-specific predictive value. H2 and H3 were not supported.

## Calibration and execution

Conformal marginal coverage was 0.95, average set size 1.417, and 43.3% of TEST rows were non-singleton. Direct Q-full covered all rows at 0.900 accuracy. Both conformal-gated direct execution and the current counterfactual selector covered 56.7% at 0.941 precision; the risk layer did not reject any additional singleton case in this pilot. Counterfactual execution auto-committed 34/60 and quarantined 26/60. Graph/index consistency was 100%, mixed-epoch exposures were zero, and failure tests restored exact pre-state hashes.

## Interpretation

The architecture implements the intended safety separation, but the pilot identifies important weaknesses: hybrid fusion and hierarchy did not improve their baselines; Q-full underperformed B0; and mean current-evidence miss in simple witness probes was 0.910. These results motivate witness design and a new confirmatory benchmark, not post-hoc TEST tuning.
