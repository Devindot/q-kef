# Dataset Plan

## Planned source benchmark

The planned public benchmark is **FiQA from BEIR**. Q-KEF requires a controlled
knowledge-evolution benchmark rather than only the original static corpus. Phase
0 documents the strategy but does not download FiQA.

## Proposed construction

1. Select an appropriately sized, reproducible FiQA subset using documented
   inclusion rules and source IDs.
2. Treat untouched source documents as Time `T0`.
3. Generate controlled `T1` variants for selected documents.
4. Assign ground-truth parent relationships and expected evolution actions.
5. Store original and generated data separately and audit the resulting labels.

Sampling size and class balance will be determined in a later phase based on
feasibility and power/coverage considerations. Parent and mutation variants must
not cross evaluation splits.

## Evolution examples

### Replace

- **Original:** “The annual management fee is 1.5%.”
- **Updated:** “Effective 2026, the annual management fee is 1.2%.”
- **Expected action:** `REPLACE`; the new fact supersedes the old fact.

### Merge

- **Original:** “The account includes fraud protection.”
- **New:** “The account additionally provides international transaction alerts.”
- **Expected action:** `MERGE`, or `COEXIST` if the final chunk representation
  treats these as independently retrievable complementary facts. The annotation
  rule must resolve this choice consistently before evaluation.

### Contradiction or supersession

- **Original:** “Withdrawals before age 60 incur a penalty.”
- **Updated:** “Under the revised policy, withdrawals after age 55 do not incur
  the previous penalty.”
- **Expected relationship:** explicit supersession rather than retaining both as
  equally active facts. The final label depends on a documented temporal and
  semantic annotation policy.

### Coexist

Two documents discuss related topics, but neither invalidates or subsumes the
other. Both remain active with an expected action of `COEXIST`.

### New

A document covers genuinely new information with no appropriate existing
candidate to replace or merge. The expected action is `NEW`.

### Archive and Split

Controlled archive examples may represent explicit withdrawal without a direct
replacement. `SPLIT` remains representable but may be excluded from experiments
if a reliable operational definition and sufficient cases cannot be established.

## Provenance record

Every controlled update must preserve at least:

| Field | Purpose |
|---|---|
| mutation ID | Stable identity for the generated example |
| parent document ID | Link to the untouched source unit |
| mutation type | Controlled transformation category |
| original text | Auditable T0 content |
| updated text | Auditable T1 content |
| expected evolution action | Ground-truth experimental label |
| generation method | Template, rule, human edit, or documented model |
| random seed | Reproduce stochastic choices; default `42` |
| source/version timestamps | Establish temporal ordering |
| generator/config version | Recreate the transformation process |
| reviewer/audit status | Track label quality checks |

Additional fields may include ambiguity notes, supersedes/superseded-by IDs, and
the experiment split.

## Storage and integrity

- Never silently alter benchmark data.
- Store FiQA source data under `data/raw/` and controlled variants under a
  separate interim/processed path in a future phase.
- Store provenance manifests under `data/metadata/`.
- Prefer checksums or immutable copies for downloaded source artifacts.
- Keep data out of Git when licensing, size, or reproducibility practice requires
  scripted acquisition; document the exact acquisition method instead.
- Record timestamps, versions, transformations, seed, and configuration for every
  generated dataset release.

## Phase 0 boundary

FiQA is neither downloaded nor transformed in Phase 0. No examples in this
document are silently inserted into benchmark data.
