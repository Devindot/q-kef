# Q-KEF v2 Final Project Status

Q-KEF v2 is complete as a reproducible academic software release on branch `qkef-v2-retrieval-safe-evolution`. Q-KEF v1 remains frozen at commit `52bd9f3c703b002c0d03acc0b4ef1a689d86ad9c`.

## Completed scope

- Retrieval-safe plan–simulate–validate–commit runtime with six lifecycle operators, non-live overlays, hard invariants, conformal gating, quarantine, epoch publication, certificates, bitemporal retrieval, authority policy, exact rollback, and seven failure stages.
- Deterministic 2,400-event benchmark: 400 events per action; 1,200 TRAIN, 360 DEV, 360 CALIBRATION, and 480 TEST; 3,600 unique source documents; zero pairwise source/query ancestry overlap.
- Locked confirmatory protocol: DEV configuration selection, CAL-only conformal fitting, hashed pre-TEST models, and one guarded TEST execution.
- Full ablation, paired bootstrap, exact McNemar tests, Holm correction, Wilson intervals, technical-effect replay, final figures, interactive app, and release validation.

## Confirmatory results

- Dense / locked-hybrid TEST Recall@5: 0.8958 / 0.8875; H1 not supported.
- B0 / Q-full / matched non-Q / hierarchical Q TEST macro-F1: 0.8791 / 0.8708 / 0.8517 / 0.8121.
- Q-full minus B0: -0.0083, bootstrap 95% CI [-0.0250, 0.0078], Holm-adjusted p=1.0000.
- Q-full minus matched non-Q: +0.0191, exact McNemar p=0.1496; Q-specific predictive value is not established.
- Conformal marginal coverage: 0.7792 versus nominal 0.90; ARCHIVE coverage was 0.0. This is a material negative result.
- Counterfactual execution: 281/480 auto-commits, precision 0.9466, quarantine 0.4146, mean current-evidence miss 0.0.
- Graph/index consistency 1.0; mixed epochs 0; all 7/7 injected failure stages preserved the prior epoch.
- Final active index 455 versus append-only 800; local update latency p50/p95 approximately 2.98/4.22 seconds.

## Release boundary

All machine-executable project work is complete. The 120-row stratified human-audit package is prepared, but independent reviewer identity, judgments, dates, and signature remain `AWAITING_HUMAN_SIGNOFF`; software cannot truthfully fabricate them. The system remains an academic prototype, not production infrastructure, legal advice, financial advice, a patentability opinion, or evidence of quantum computational advantage.

## Reproduction

```powershell
.phase3-venv\Scripts\python.exe scripts\build_v2_confirmatory_benchmark.py
.phase3-venv\Scripts\python.exe scripts\prepare_v2_confirmatory_experiment.py
# The next command is guarded and must not be rerun for the same experiment version.
.phase3-venv\Scripts\python.exe scripts\evaluate_v2_confirmatory_test.py
.phase3-venv\Scripts\python.exe scripts\complete_v2_confirmatory_reporting.py
.phase3-venv\Scripts\python.exe scripts\generate_v2_confirmatory_figures.py
.phase3-venv\Scripts\python.exe scripts\build_v2_human_audit_pack.py
# After committing all release inputs:
.phase3-venv\Scripts\python.exe scripts\build_v2_final_release_manifest.py
.phase3-venv\Scripts\python.exe scripts\v2_release_check.py
.phase3-venv\Scripts\streamlit.exe run app_v2.py
```

The frozen TEST receipt already exists for `qkef-v2-confirmatory-1`; reproduction of TEST requires a new explicitly versioned experiment rather than overwriting the release.
