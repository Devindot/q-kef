# Research Hypotheses

All hypotheses are prospective and must be evaluated against predefined metrics.
Failure to reject a null hypothesis is a valid research outcome.

## H1 — Obsolete retrieval

An evolution-aware knowledge base will retrieve less obsolete information than
an append-only baseline after knowledge updates are introduced.

- **Primary comparison:** Baseline B and System C versus Baseline A.
- **Primary outcome:** obsolete retrieval rate at fixed values of `k`.
- **Supporting outcomes:** Precision@k, nDCG@k, and contradiction retrieval rate.
- **Null hypothesis:** evolution-aware systems do not reduce obsolete retrieval
  relative to append-only RAG.

## H2 — Answer correctness

Evolution-aware preprocessing will improve answer correctness for questions
affected by updated or contradictory knowledge.

- **Primary comparison:** Baseline B and System C versus Baseline A.
- **Primary outcomes:** answer correctness and obsolete-answer rate.
- **Supporting outcomes:** token F1, Exact Match where applicable, and
  supported-answer rate.
- **Null hypothesis:** evolution-aware preprocessing does not improve answer
  correctness or reduce obsolete answers.

## H3 — Quantum-inspired component ablation

The proposed quantum-inspired state representation will provide measurable
benefit over a conventional similarity-only evolution decision mechanism.

- **Required comparison:** System C versus Baseline B under otherwise matched
  data, candidate sets, downstream retrieval, and evaluation settings.
- **Primary outcomes:** lifecycle macro F1 and obsolete retrieval rate.
- **Supporting outcomes:** per-class F1, answer correctness, retrieval nDCG@k,
  ingestion time, and retrieval latency.
- **Null hypothesis:** the proposed representation provides no measurable
  benefit over the conventional mechanism.

H3 must not be inferred from an improvement of System C over append-only RAG
alone. Baseline B is required to isolate the effect of knowledge evolution from
the effect of the quantum-inspired representation.

## H4 — Cost and benefit

Any retrieval or answer-quality improvement from lifecycle management will have
measurable computational and storage costs that can be characterized on a normal
student laptop.

- **Outcomes:** ingestion time, retrieval latency, active knowledge-base size,
  and counts of archived or superseded nodes.
- **Purpose:** report trade-offs rather than presuppose that an accuracy gain is
  operationally worthwhile.

## Testing expectations

Before experiments, the project will define evaluation splits, fixed seeds,
values of `k`, aggregation rules, and appropriate uncertainty or significance
analysis. Results will be reported for all compared systems, including negative
and class-specific results. Thresholds currently set to `null` must be selected
without using the held-out test set.
