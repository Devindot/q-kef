# Q-KEF v2 Confirmatory Experiment Report

## Protocol

The 2,400-event ancestry-isolated benchmark was used exactly as locked: 1,200 TRAIN, 360 DEV, 360 CALIBRATION, and one 480-row TEST evaluation. DEV selected configurations, CAL fitted Mondrian thresholds, and model hashes were verified before TEST access.

## Candidate retrieval

Dense Recall@5 was 0.8958; the locked hybrid achieved 0.8875. H1 was not supported.

## Lifecycle models

B0 macro-F1 was 0.8791; Q-full was 0.8708; matched non-Q was 0.8517; hierarchical Q-full was 0.8121. Q-full minus B0 was -0.0083, bootstrap 95% CI [-0.0250, +0.0078], Holm-adjusted p=1.0000. The planned Q-full versus matched non-Q contrast was +0.0191, bootstrap 95% CI [-0.0031, +0.0423], exact McNemar p=0.1496; this did not establish a Q-specific benefit.

## Calibration and safe execution

Conformal marginal coverage was 0.7792, mean set size 1.0354, and singleton rate 0.5854. Counterfactual execution committed 281/480 transitions at precision 0.9466; direct top-1 accuracy was 0.8708.

## Technical effects

Graph/index consistency was 1.0000; all 7/7 failure stages preserved the prior epoch. Final active index size was 455 versus append-only 800. Update latency p50/p95 was 2984.16/4220.28 ms.

## Interpretation

All null and negative outcomes are retained. The benchmark is controlled and synthetic; this experiment does not establish production readiness, legal patentability, or quantum computational advantage. Human review of the audit sample remains a separate signed process.
