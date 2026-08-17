# Final System Architecture

## Data and preprocessing

```mermaid
flowchart LR
  A["BEIR FiQA"] --> B["Controlled T0/T1 benchmark"]
  B --> C["Provenance-safe ingestion"]
  C --> D["Identity + boundary chunks"]
  D --> E["Sanitized model_text"]
```

## Training and frozen evaluation

```mermaid
flowchart TD
  T["TRAIN ancestry"] --> P["Fit PCA basis"]
  T --> B["Fit classifier pipelines"]
  P --> Q["Quantum-inspired features"]
  D["DEV only"] --> S["Select C and state dimension"]
  S --> L["Lock configuration"]
  L --> X["Held-out TEST once"]
  X --> A["Frozen Phase 3 artifacts"]
```

DEV and TEST are transformed by the frozen TRAIN PCA. Ground truth is never a runtime feature.

## Runtime and application

```mermaid
flowchart TD
  I["Interactive text"] --> N["Normalization"]
  N --> M["Cached CPU MiniLM"]
  M --> C["Same top-5 candidates"]
  C --> F["Conventional features"]
  M --> P["Frozen TRAIN PCA transform"]
  P --> Q["State features"]
  F --> B["Conventional B"]
  F --> K["Q-KEF C"]
  Q --> K
  B --> E["Effect preview / demo execution"]
  K --> E
  E --> V["Active vector index"]
  E --> G["Semantic graph"]
  V --> R["Evidence retrieval"]
  G --> R
  R --> A["Extractive answer"]
```

`FinalArtifactLoader` verifies repository-trusted artifacts. `run_final_analysis.py` reads predictions without training. Streamlit caches immutable resources while mutable demo state remains per session.
