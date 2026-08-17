# Terminology

The following definitions constrain project documentation, code, and annotation
guidelines.

## Document

A source-level artifact with a stable source identity and provenance, such as one
FiQA corpus entry or one controlled temporal version of it. A document may yield
one or more chunks.

## Chunk

A contiguous or semantically selected segment produced from a document for
encoding or retrieval. Chunk boundaries are processing choices and do not by
themselves establish a lifecycle relationship.

## Knowledge unit

The smallest independently represented semantic item in the Q-KEF pipeline. It
contains text plus source, version, temporal, lifecycle, relationship, and
optional index identifiers. A chunk may map to one or more knowledge units,
depending on later design decisions.

## Knowledge node

The graph representation of a knowledge unit (or, where explicitly modelled, a
related entity). A node has an identifier and may participate in supersession,
provenance, or semantic relationships.

## Lifecycle status

The persisted current state of a knowledge unit, such as `active`, `archived`, or
`superseded`. Status is distinct from the action that caused a transition.

## Evolution action

A labelled, predicted, or applied relationship decision for incoming knowledge:
`NEW`, `REPLACE`, `MERGE`, `ARCHIVE`, `COEXIST`, or potentially `SPLIT`. An
action can update statuses and relationships but is not itself a lifecycle
status.

## Supersession

An explicit temporal relationship in which a newer or authoritative knowledge
unit replaces the applicability of an older unit for a defined scope. The older
unit is preserved for history but should not be treated as equally current.

## Contradiction

Two claims that cannot both be true under the same scope, assumptions, and time.
Apparent textual conflict is not necessarily contradiction if scope or effective
time differs. Contradiction does not automatically establish which claim
supersedes the other.

## Semantic similarity

A numerical estimate of meaning-relatedness, commonly derived from vector
embeddings. High similarity does not alone prove equivalence, contradiction, or
supersession.

## Quantum-inspired state

A normalized vector or related classical mathematical object whose design draws
inspiration from quantum state notation. In Q-KEF it is computed on conventional
hardware and is not a physical quantum state.

## Superposition-style representation

A classical multidimensional combination written conceptually as
`|psi> = sum_i alpha_i|c_i>`, with normalized coefficients. The term describes a
mathematical modelling analogy; it does not imply physical superposition or
quantum computation.

## Fidelity-like similarity

A classical similarity inspired by quantum fidelity, potentially calculated as
`|<psi_a|psi_b>|^2` for normalized states. “Fidelity-like” is used because the
project has not established a physical quantum interpretation or final metric.

## Vector embedding

A dense numerical representation generated from text or other features so that
geometric operations can support semantic comparison and retrieval.

## Knowledge graph

A graph of nodes and typed edges representing knowledge units, provenance,
lifecycle, and semantic relationships. NetworkX is planned for early prototypes;
the term does not require Neo4j.

## Append-only RAG

A retrieval-augmented generation design in which incoming chunks are added to an
index without first revising the lifecycle of related existing chunks.

## Evolution-aware RAG

A RAG design in which incoming knowledge is compared with existing knowledge and
an explicit lifecycle action can affect what is active, related, or preferred in
retrieval before final indexing.

## Usage constraint

Documentation must not describe Q-KEF's classical calculations as genuine quantum
computation, claim a quantum advantage, or assert that the proposed approach is
proven superior. Such questions remain subject to experiment.
