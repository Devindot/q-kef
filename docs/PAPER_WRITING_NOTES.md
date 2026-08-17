# Paper Writing Notes

## Candidate title

“Q-KEF: A Classical Quantum-Inspired Knowledge Evolution Layer for Temporally Changing Retrieval Corpora”

## Abstract structure

Problem with append-only retrieval → six-action pre-indexing method → controlled FiQA benchmark → A/B/C design → candidate/classification/retrieval results → uncertainty → limitations.

## Gap and contributions

- Explicit lifecycle management before final indexing.
- Reproducible 300-event temporal FiQA benchmark.
- Six-action deterministic evolution engine with vector and graph state.
- Classical normalized PCA-state augmentation and controlled B/C ablation.
- Obsolete and lineage-aware retrieval analysis.
- Offline interactive research prototype.

## Methodology and hypotheses

Describe ancestry-disjoint splits, Phase 2 text sanitization, MiniLM, same-split top-5 candidates, safe conventional features, TRAIN-only PCA, DEV lock, one TEST report, and A/B/C/oracle simulations. H1: lifecycle management lowers obsolete retrieval vs append-only. H2: Q features alter held-out macro F1 beyond conventional features.

## Experimental setup and results

Use `reports/final/tables` and figures 1–8. Report R@5 0.9600; B/C macro F1 0.8455/0.8933; obsolete@5 A/C 0.3393/0.1180; then exact bootstrap/McNemar outputs.

## Limitations and conclusion wording

Disclose controlled mutations, small TEST, generic embeddings, COEXIST weakness, segmentation overlap, and provenance relevance. Conclude that evolution-aware management reduced observed obsolete retrieval and Q features had a positive descriptive result here, without claiming quantum advantage or broad generalization.

Do not fabricate literature citations; add independently verified sources during paper preparation.
