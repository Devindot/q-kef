# Knowledge Evolution Engine

Phase 3 implements three learned/evaluation conditions plus an oracle simulator over the same initial T0 knowledge base:

- A: append-only, which always inserts incoming T1 knowledge.
- B: conventional evolution, using lexical, numeric, length, and MiniLM similarity features.
- C: Q-KEF, using all B features plus the documented quantum-inspired state features.
- Oracle: executes benchmark actions and targets to establish an upper-bound state for retrieval analysis.

The execution layer supports NEW, REPLACE, MERGE, ARCHIVE, COEXIST, and SPLIT. REPLACE supersedes a target; MERGE creates a deterministic derived record with source and parent provenance; ARCHIVE removes a target from active retrieval; COEXIST retains both and writes bidirectional lineage; SPLIT creates children from Phase 2 chunk boundaries, falling back safely with a warning when segmentation is unavailable. Every condition owns independent records, vector status, and a serializable NetworkX lineage graph.

Active-only search is the default. Retrieval evaluation compares semantic-plus-lexical ranking across A/B/C/oracle and reports Recall@5, MRR, nDCG@5, and obsolete-result rate. Candidate-target metrics explicitly exclude NEW and SPLIT (no target) and ARCHIVE (explicit-reference action), preventing invalid denominator inflation.

Labels, targets, actions, benchmark relations, split names, and provenance identifiers are never model inputs. Feature tables contain opaque sample keys; labels are written separately for evaluation.
