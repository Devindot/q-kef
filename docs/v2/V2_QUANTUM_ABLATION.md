# Quantum-Inspired Matched Ablation

The primary architectural contribution under investigation is retrieval-safe knowledge evolution. The quantum-inspired representation is an optional feature representation evaluated through matched ablation.

The v2 pilot compares B0 conventional features; B1 plus signed PCA-16 state components; B2 plus ten deterministic non-Q PCA summaries; fidelity-only; entropy-only; coherence-only; and Q-full. Rows, split discipline, classifier family, and C-grid budget are matched. PCA remains TRAIN-only from v1.

On the retrospective 60-row TEST pilot, B0 macro-F1 was 0.9327, Q-full 0.8990, PCA-16 0.8198, and matched non-Q 0.8860. Q-full exceeded the matched non-Q control by 0.0130 but underperformed B0 by 0.0337; the paired interval crosses zero and McNemar p=0.5. This does not establish a Q-specific advantage.
