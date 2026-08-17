# Experiment Plan

## Objective

Evaluate separately whether pre-indexing knowledge evolution helps and whether
the proposed classical quantum-inspired representation adds value beyond a
conventional mechanism. This document defines the planned comparison; it does
not implement or prejudge the experiments.

## Mandatory systems

### Baseline A — Append-Only RAG

Incoming documents or chunks are embedded and added to the vector store. Older
units remain active and no lifecycle action is inferred before retrieval.

### Baseline B — Conventional Evolution Engine

Incoming units are compared with existing candidates using conventional semantic
similarity plus available metadata, document versions, and timestamps. It has no
quantum-inspired state representation. This baseline measures the value of
knowledge evolution itself relative to Baseline A.

### Proposed System C — Q-KEF

Incoming units use the proposed classical quantum-inspired contextual state
representation together with the Knowledge Evolution Engine. This system is
compared directly with Baseline B to test whether the representation contributes
anything beyond conventional lifecycle handling.

## Controlled comparison

All systems should use the same source subset, temporal mutations, train/dev/test
partition, query set, downstream embedding model, retrieval budget, answer
component, random seeds, and hardware. Baselines B and C should receive the same
candidate pool wherever their designs permit. Configuration and software
versions must be recorded for every run.

Thresholds must be estimated on training/development data, never on the held-out
test set. Repeated seeded runs or appropriate resampling should be used where
stochastic components are present. The default project seed is `42`.

## Planned metrics

### Evolution metrics

- lifecycle action accuracy;
- macro F1;
- per-class precision, recall, and F1;
- confusion matrix.

Macro F1 is important because action classes are likely to be imbalanced.

### Retrieval metrics

- Recall@k;
- Precision@k;
- Mean Reciprocal Rank (MRR);
- nDCG@k;
- obsolete retrieval rate;
- contradiction retrieval rate.

An obsolete retrieval must be operationally defined using ground-truth temporal
status. Contradiction retrieval should distinguish useful retrieval of a
contradiction for comparison from presenting contradictory evidence as equally
current.

### Answer metrics

- Exact Match where applicable;
- token F1;
- answer correctness;
- supported-answer rate;
- obsolete-answer rate.

Human or model-assisted correctness judging, if later used, must have a documented
rubric, blinded system labels where practical, and inter-rater or calibration
checks. Paid APIs are not required.

### System metrics

- ingestion time;
- retrieval latency;
- knowledge-base size;
- number of archived nodes;
- number of superseded nodes.

Report the laptop hardware and distinguish one-time indexing from per-query cost.

## Planned analysis

1. **Evolution effect:** compare Baseline B with Baseline A.
2. **Full proposed-system effect:** compare System C with Baseline A.
3. **Required ablation:** compare System C with Baseline B to isolate the
   quantum-inspired component (H3).
4. **Class-level analysis:** inspect action confusion, especially `MERGE`,
   `COEXIST`, and any retained `SPLIT` cases.
5. **Temporal/error analysis:** group outcomes by update type, contradiction,
   document version, and temporal distance.
6. **Trade-off analysis:** relate quality changes to ingestion, latency, and
   storage costs.

Exact sample sizes, values of `k`, confidence intervals, and statistical tests
will be preregistered in a later phase after inspecting dataset characteristics
without using final test outcomes.

## Leakage and validity controls

- Preserve original documents and controlled mutations separately.
- Keep parent-related variants in the same data split.
- Do not tune thresholds or prompts on the test set.
- Version generated mutations and evaluation labels.
- Manually audit a documented sample of mutations and labels.
- Report failed and ambiguous examples rather than silently removing them.
- Treat `SPLIT` as optional for experiments even though the schema represents it.

## Phase 0 exclusions

No benchmark data, metric computation, model training, evolution logic, retrieval
system, or answer generator is implemented in this phase.
