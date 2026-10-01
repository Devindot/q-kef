# Q-KEF v2 Project Status

Q-KEF v2 implements the retrieval-safe transition architecture on branch `qkef-v2-retrieval-safe-evolution` while preserving v1 at commit `52bd9f3c703b002c0d03acc0b4ef1a689d86ad9c`.

Implemented: explicit graph/index/epoch state; separate lifecycle/retrieval status; six cardinality-aware operators; non-live delta overlays; deterministic witnesses; interpretable risk and hard invariants; Mondrian conformal gating; quarantine; epoch publication; seven failure stages; canonical certificates; exact rollback deltas; bitemporal retrieval; optional authority policy; dense/lexical/RRF retrieval; fitted hierarchical models; matched Q/non-Q ablations; local adapters; app_v2; release checker; patent-oriented technical docs; and 12 figures.

The completed run is a retrospective pilot on v1 rows, not the planned 2,400-event confirmatory benchmark. Results are mixed: dense retrieval beat equal-weight hybrid, flat B0 beat Q-full and hierarchical variants, 34/60 transitions auto-committed, and current-evidence miss was high. See `reports/v2/V2_EXPERIMENT_REPORT.md`.

Post-pilot development now includes a leakage-guarded DEV-only retrieval selector, weighted RRF configurations, incoming-anchored contrastive witnesses, IDF-weighted shadow retrieval, complete SPLIT affected-lineage accounting, and action-aware current-evidence miss semantics. Dense retrieval was locked from 30 eligible DEV queries (Recall@5 1.000; MRR 0.903), and revised probes had zero current-evidence misses across the controlled 60-event DEV audit. These changes do not alter the frozen pilot results or constitute confirmatory evidence. See `docs/v2/V2_DEVELOPMENT_PROTOCOL.md` and `reports/v2/development/`.

```powershell
.phase3-venv\Scripts\python.exe scripts\run_v2_benchmark.py
.phase3-venv\Scripts\python.exe scripts\run_v2_statistics.py
.phase3-venv\Scripts\python.exe scripts\v2_power_analysis.py
.phase3-venv\Scripts\python.exe scripts\generate_v2_figures.py
.phase3-venv\Scripts\python.exe scripts\v2_release_check.py
.phase3-venv\Scripts\python.exe scripts\select_v2_dev_retrieval.py
.phase3-venv\Scripts\python.exe scripts\audit_v2_dev_witnesses.py
.phase3-venv\Scripts\streamlit.exe run app_v2.py
```

Patentability is not guaranteed. Prior-art characterizations and any proposed differentiation require professional verification.
