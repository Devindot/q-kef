# Phase 3 Core Research Results

Neural backend: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).

Events: 180 TRAIN / 60 DEV / 60 TEST. TEST was not used for tuning.

Candidate Recall@5: 0.9600; MRR: 0.8978. ARCHIVE explicit references and NEW/SPLIT no-target cases are excluded.

Conventional TEST macro F1: 0.8455. Q-KEF TEST macro F1: 0.8933. Ablation difference: +0.0479.

All computations are classical. Results concern a modest controlled synthetic benchmark and do not establish general superiority or statistical significance.
