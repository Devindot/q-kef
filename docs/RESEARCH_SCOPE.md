# Research Scope

## Problem

Standard append-only knowledge and retrieval-augmented generation systems add
new documents or chunks without determining how they relate to existing
knowledge. As a knowledge base evolves, this may retain outdated, duplicated,
superseded, or contradictory information as equally active retrieval candidates.
The result can be retrieval of obsolete evidence and answers supported by the
wrong temporal version.

Q-KEF investigates a pre-indexing lifecycle step for incoming knowledge. It will
eventually compare incoming units with candidate existing units and represent an
action such as `NEW`, `REPLACE`, `MERGE`, `ARCHIVE`, `COEXIST`, or—if supported
by later experiments—`SPLIT` before the final vector and graph indexes are
updated.

## Primary research question

> Can pre-indexing knowledge lifecycle management improve retrieval and answer
> correctness in temporally evolving and contradictory knowledge bases compared
> with append-only RAG?

## Secondary research questions

1. Does evolution-aware indexing reduce the rate at which obsolete knowledge is
   retrieved after controlled updates?
2. How effectively does lifecycle management handle contradictions and explicit
   supersession relationships?
3. Does the proposed layer improve retrieval precision while retaining relevant
   knowledge recall?
4. Does it improve answer correctness and reduce answers based on obsolete
   evidence?
5. How accurately can the system classify lifecycle actions across action
   classes, including minority classes?
6. Does the proposed classical quantum-inspired representation add measurable
   benefit beyond a conventional semantic-similarity and metadata mechanism?
7. What latency, ingestion-time, and knowledge-base-size costs accompany any
   measured benefits?

## Scope boundaries

### In scope for the overall research

- controlled, provenance-preserving temporal evolution of a reproducible FiQA
  subset;
- semantic units with explicit versions, timestamps, statuses, and relationships;
- append-only, conventional-evolution, and Q-KEF experimental systems;
- classical vector-based quantum-inspired representations;
- lifecycle, retrieval, answer, and system metrics;
- an ablation that isolates the quantum-inspired component.

### Out of scope for Phase 0

- downloading or modifying FiQA data;
- training or selecting a final embedding or language model;
- implementing semantic chunking, candidate selection, evolution decisions,
  retrieval, answer generation, dashboards, or experiments;
- production graph/vector storage or Neo4j integration;
- quantum hardware, Qiskit, or claims of quantum advantage.

## Scientific positioning

Q-KEF is a **proposed** framework and research hypothesis. Its possible novelty,
effectiveness, costs, and the contribution of its quantum-inspired component are
to be experimentally evaluated. Phase 0 establishes falsifiable questions and
reproducible interfaces without assuming favorable results.

## Reproducibility requirements

The default seed is `42`. Future runs must record source IDs, random seeds,
document versions and timestamps, transformation types, generation methods,
software/configuration versions, and experiment identifiers. Original benchmark
data must remain unaltered, with controlled variants stored separately.
