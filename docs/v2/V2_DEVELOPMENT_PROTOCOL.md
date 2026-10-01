# Q-KEF v2 Development Protocol

## Post-pilot rule

The frozen retrospective TEST results are diagnostic evidence only. Changes motivated by those results begin a new development cycle and must not overwrite the pilot artifacts. Retrieval configuration is selected from DEV observations with `scripts/select_v2_dev_retrieval.py`; its API rejects non-DEV observations.

## Witness semantics

Content-derived witnesses are anchored to salient incoming terms and contrastive predecessor terms. Current-evidence miss is evaluated against affected records that should remain actively retrievable after the proposed transition. SPLIT children are affected lineage members. ARCHIVE transitions with no active successor have no current-evidence obligation and therefore are not assigned an artificial miss.

The shadow lexical probe uses IDF-weighted query-term coverage rather than whole-document Jaccard similarity, avoiding a length penalty against long but exact evidence. On the 60 DEV events, the revised probes produced zero current-evidence misses under this revised evaluator. This is a development diagnostic, not an out-of-sample claim, and the frozen pilot figure is not rewritten.

## Promotion gate

Before the confirmatory benchmark is generated, the development configuration must be locked with its input hashes and deterministic selection rule. The 2,400-event benchmark remains unexecuted. No new TEST evaluation is authorized by this development artifact.
