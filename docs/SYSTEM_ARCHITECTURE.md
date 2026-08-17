# System Architecture

## Research architecture

```text
Incoming Documents
        |
        v
Semantic Ingestion
        |
        v
Semantic Chunking
        |
        v
Embedding Encoder
        |
        +--------------------------+
        |                          |
        v                          v
Candidate Existing Knowledge   Context Features
        |                          |
        +-------------+------------+
                      |
                      v
          Quantum-Inspired Encoder
                      |
                      v
            Knowledge Evolution
                  Engine
                      |
        +-------------+------------------+
        |       |       |       |        |
       NEW   REPLACE   MERGE  ARCHIVE  COEXIST
                      |                  (+ SPLIT if supported)
                      v
             Evolution-Aware KB
              /              \
             v                v
        Vector Index      Semantic Graph
             \                /
              +-------+------+
                      |
                      v
               Hybrid Retrieval
                      |
                      v
                 LLM / QA
```

The central research contribution under investigation is the **pre-indexing
knowledge evolution layer**: comparison and lifecycle action occur before the
incoming unit is finalized in the retrieval indexes.

## Comparison architecture

```text
Standard RAG:
Ingest -> Chunk -> Embed -> Append -> Retrieve -> Generate

Q-KEF:
Ingest -> Chunk -> Compare -> Evolve -> Index -> Retrieve -> Generate
```

The experimental design also includes a conventional evolution system with the
same overall Q-KEF flow but without the quantum-inspired encoder. This is needed
to distinguish the effect of lifecycle management from the effect of the proposed
representation.

## Planned component responsibilities

1. **Semantic ingestion (implemented in Phase 2)** preserves exact raw text,
   source identity, version, timestamps, and provenance while producing a
   separate conservatively normalized and scaffold-sanitized model text.
2. **Deterministic chunking (implemented in Phase 2)** derives identity,
   fixed-window, and lexical TF-IDF-boundary chunks from one knowledge unit at a
   time without discarding source relationships or using lifecycle labels.
3. **Embedding encoder** supplies conventional semantic vectors used for
   candidate discovery and retrieval.
4. **Candidate selection** finds a limited set of potentially related active or
   historical knowledge units.
5. **Context features** represent version, time, provenance, and other non-textual
   evidence relevant to lifecycle decisions.
6. **Quantum-inspired encoder** may combine semantic and contextual dimensions in
   a normalized classical state representation.
7. **Knowledge Evolution Engine** will eventually predict or apply a lifecycle
   action. It is not implemented in Phase 0.
8. **Evolution-aware knowledge base** preserves active, archived, and superseded
   units plus explicit relationships.
9. **Vector index and semantic graph** provide complementary semantic and
   relational access paths. NetworkX is planned for initial prototyping; optional
   Neo4j integration is deferred.
10. **Hybrid retrieval and LLM/QA** consume evolution-aware evidence. They are
    downstream evaluators, not Phase 0 components.

## Classical quantum-inspired formulation

A future contextual knowledge state may be represented conceptually as

```text
|psi> = alpha_1|c_1> + alpha_2|c_2> + ... + alpha_n|c_n>
```

subject to

```text
sum_i |alpha_i|^2 = 1
```

where basis states may represent learned semantic or contextual dimensions. A
possible state comparison is a fidelity-like quantity:

```text
F(psi_a, psi_b) = |<psi_a | psi_b>|^2
```

These would be ordinary classical numerical computations inspired by quantum
mathematical notation. The final basis, amplitudes, feature mapping, dimension,
and comparison rule are undecided. No scientific superiority or quantum
advantage is assumed; System C versus Baseline B must test the contribution.

## Phase 0 package boundaries

The `src/qkef/` modules reserve boundaries for ingestion, chunking, embeddings,
quantum-inspired representation, evolution, graph, retrieval, evaluation, and
schemas. Only implementation-neutral schemas are active in Phase 0. Empty
component packages do not imply implemented functionality.
