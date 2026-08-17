# Final Viva Guide

## Core questions

**What is Q-KEF?** A pre-indexing lifecycle layer that predicts and executes six knowledge-evolution actions before retrieval.

**What problem is solved? Why isn't ordinary RAG enough?** Append-only retrieval can preserve obsolete or conflicting records without status. Q-KEF explicitly models lifecycle and audit history.

**Main contribution?** A reproducible controlled temporal FiQA benchmark, six-action engine, fair A/B/C evaluation, classical quantum-inspired augmentation, obsolete-retrieval metric, and working prototype.

**Why FiQA and synthetic temporal evolution?** FiQA is public and financial-domain with queries/provenance. It lacks historical lifecycle labels, so controlled transformations make ground truth reproducible; they are not claimed as real history.

**Why six actions?** They cover addition, supersession, composition, removal from active retrieval, valid alternatives, and segmentation.

**Why MiniLM?** CPU-compatible semantic embeddings with practical 384-dimensional inference.

**Why logistic regression?** Only 180 TRAIN events; a transparent, regularized model reduces overfitting risk and supports coefficient inspection.

**Why PCA and dimension 16?** PCA learns a compact TRAIN-only contextual basis. Dimension 16 and C=1.0 were selected on DEV, never TEST.

**What is the superposition-style representation?** A normalized real vector expressed in a learned basis. Squared components form a probability-like distribution; this is analogy, not physical state preparation.

**What is fidelity-like similarity?** The squared dot product of normalized real states, bounded to [0,1].

**Actual quantum computing?** No. No qubits, circuits, simulator, hardware, or advantage claim.

**Did Q features improve?** Descriptively, TEST macro F1 rose from 0.8455 to 0.8933 (+0.0479). Statistical interpretation must follow the final bootstrap and McNemar report.

**Why is COEXIST weak?** It has the lowest F1 (B 0.5333, C 0.6250); its semantic boundary with NEW/MERGE can be ambiguous. Stored confusion data supports the error discussion.

**Why is Oracle Recall lower?** Lifecycle consolidation reduces independently retrievable records under source-record recall. Oracle obsolete@5 is zero; lineage-normalized hit rate is reported separately.

**What is obsolete retrieval?** The top-k fraction whose identifiers are archived or superseded under oracle lifecycle state.

**Why NetworkX instead of Neo4j?** The experiment is small; local deterministic serialization improves reproducibility and removes server infrastructure.

**Why no LLM?** The research question concerns lifecycle and retrieval. Deterministic extractive QA avoids cost, hallucination, and API dependencies.

**Does Q-KEF eliminate hallucinations?** No. It only reduces observed obsolete retrieval; generation was not evaluated.

**Production ready?** No. Human review, broader enterprise data, calibration, access control, governance, and deployment testing are needed.

**Main limitations/future work?** Small controlled dataset, synthetic updates, generic embeddings, COEXIST ambiguity, segmentation overlap, and provenance-specific relevance. Future work should use real longitudinal enterprise corpora and human judgments.

**Patentability proven?** No; no legal novelty or patent status is claimed.

**How was leakage prevented?** Ancestry-disjoint splits, Phase 2 sanitization, opaque IDs, TRAIN-only PCA, safe numeric feature matrices, explicit audits, and a locked TEST protocol.
