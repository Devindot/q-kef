# Final Experiment Report

## 1. Research question

Can pre-indexing lifecycle management reduce obsolete retrieval, and does a classical quantum-inspired feature augmentation add measurable held-out value beyond conventional evolution features?

## 2. Dataset and controlled temporal benchmark

Canonical BEIR FiQA contains 57,638 documents and 6,648 queries. The reproducible Q-KEF benchmark contains 300 controlled/synthetic temporal events—50 per action—with 180 TRAIN, 60 DEV, and 60 held-out TEST events. These updates are not historical enterprise events.

## 3. Ingestion, chunking, and leakage controls

Phase 2 preserves raw text/provenance and provides sanitized `model_text`, identity representations for event classification, and TF-IDF boundary chunks for retrieval/SPLIT. Source ancestry overlap is zero. Action, relation, mutation, target, split, and provenance identifiers are absent from model X. PCA uses TRAIN ancestry only; TEST was not used for selection.

## 4. Systems

- **A append-only:** retains all T0/T1 knowledge.
- **B conventional:** MiniLM candidates plus safe conventional features and balanced logistic regression (C=0.1).
- **C Q-KEF:** same candidates/classifier family plus a TRAIN-only PCA-16 normalized state and fidelity-like, entropy, coherence-like, and Hellinger features (C=1.0).
- **Oracle:** executes benchmark lifecycle actions for descriptive comparison.

All quantum-inspired calculations are classical. No LLM, quantum hardware, or simulator is involved.

## 5. Candidate retrieval

For 150 eligible REPLACE/MERGE/COEXIST events: R@1=0.8533, R@3=0.9467, R@5=0.9600, MRR=0.8978. NEW/SPLIT have no target; ARCHIVE is an explicit-reference administrative action.

## 6. Held-out lifecycle classification

| System | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Conventional B | 0.8500 | 0.8500 | 0.8455 | 0.8455 |
| Q-KEF C | 0.9000 | 0.9000 | 0.8933 | 0.8933 |

The quantum-inspired ablation difference is **+0.0479 absolute macro F1** on this controlled held-out benchmark.

## 7. Per-class and error analysis

Q-KEF F1 is NEW 0.7826, REPLACE 1.0000, MERGE 1.0000, ARCHIVE 1.0000, COEXIST 0.6250, SPLIT 0.9524. COEXIST is weakest: Conventional classified six of ten COEXIST cases as NEW; Q-KEF classified four as NEW and one as SPLIT. This supports boundary ambiguity, not a causal explanation.

## 8. Frozen retrieval results

| System | Recall@5 | MRR | nDCG@5 | Obsolete@5 |
|---|---:|---:|---:|---:|
| Append-only A | 0.6717 | 0.6503 | 0.6109 | 0.3393 |
| Conventional B | 0.7218 | 0.6494 | 0.6612 | 0.1173 |
| Q-KEF C | 0.7225 | 0.6506 | 0.6613 | 0.1180 |
| Oracle | 0.5893 | 0.5422 | 0.5466 | 0.0000 |

Oracle's lower source-record recall follows consolidation under the fixed relevance definition; relevance was not changed post hoc.

## 9. Supplementary lifecycle-aware retrieval

Denominators are 300 evaluable queries and 50 REPLACE queries. Q-KEF active-valid hit rate is 0.8133@1/0.8820@5; obsolete-free query rate is 0.8133@1/0.5467@5; current-version hit is 0.5000@1/0.6400@5. The lineage-normalized metric gives binary query credit when any active retrieved record represents source lineage: Q-KEF 0.5733@1/0.7667@5 and oracle 0.4833@1/0.6333@5. These are descriptive additions, not replacements for frozen metrics.

## 10. Statistical uncertainty

Using 5,000 paired bootstrap resamples (seed 42), B accuracy CI is [0.7500, 0.9333], C accuracy [0.8167, 0.9667], difference [-0.0167, 0.1333]. B macro-F1 CI is [0.7491, 0.9241], C [0.8057, 0.9606], difference [-0.0376, 0.1414]. Exact McNemar discordance is B-only=1, C-only=4, p=0.3750. The paired accuracy difference is not statistically significant at 0.05.

## 11. Limitations

The controlled benchmark is modest; transformations are synthetic; MiniLM is generic; COEXIST is difficult; SPLIT overlaps segmentation and has three safe fallbacks per learned/oracle system; retrieval depends on FiQA provenance; target resolution differs from classification; broader longitudinal and human evaluation is required.

## 12. Conclusion

On the controlled temporal FiQA benchmark, evolution-aware management substantially reduced observed obsolete retrieval relative to append-only storage. Q-KEF achieved higher descriptive held-out macro F1 than B, but its paired confidence interval includes zero and McNemar p=0.3750. The result therefore supports further investigation—not quantum advantage, statistical superiority, or production generalizability.
