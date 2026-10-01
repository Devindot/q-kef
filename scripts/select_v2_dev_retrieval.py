"""Lock a candidate-retrieval configuration using DEV observations only.

This script deliberately has no TEST evaluation path. Its artifact is a
development decision for the future confirmatory benchmark, not a new result on
the frozen v1 TEST partition.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from _phase1_common import PROJECT_ROOT
from qkef.datasets.evolution import load_jsonl_models
from qkef.runtime.artifacts import FinalArtifactLoader
from qkef.schemas import EvolutionEvent
from qkef.v2.canonical import sha256_file, sha256_payload
from qkef.v2.retrieval import CandidateQuery, select_dev_configuration


ELIGIBLE_ACTIONS = {"REPLACE", "MERGE", "COEXIST"}


def opaque(event_id: str) -> str:
    return "sample_" + hashlib.sha256(event_id.encode()).hexdigest()[:20]


def main() -> None:
    root = PROJECT_ROOT
    artifacts = FinalArtifactLoader(root).load()
    labels_path = root / "data/processed/qkef_phase3/evaluation_labels.csv"
    events_path = root / "data/processed/qkef_fiqa_evolution/events.jsonl"
    labels = pd.read_csv(labels_path)
    events = load_jsonl_models(events_path, EvolutionEvent)
    events_by_sample = {opaque(event.event_id): event for event in events}
    units = {unit.knowledge_id: unit for unit in artifacts.units}

    development_units = [
        unit
        for unit in artifacts.units
        if unit.temporal_state.value == "T0" and unit.benchmark_split.value == "dev"
    ]
    documents = {unit.knowledge_id: unit.model_text for unit in development_units}
    vectors = {
        unit.knowledge_id: tuple(artifacts.embedding_vectors[artifacts.embedding_index[unit.knowledge_id]])
        for unit in development_units
    }
    eligible = labels[(labels.split == "dev") & labels.target.isin(ELIGIBLE_ACTIONS)].sort_values("sample_id")
    queries = []
    for row in eligible.itertuples():
        event = events_by_sample[row.sample_id]
        incoming_id = event.incoming_knowledge_ids[0]
        incoming = units[incoming_id]
        queries.append(
            CandidateQuery(
                query_id=row.sample_id,
                query_text=incoming.model_text,
                query_vector=tuple(artifacts.embedding_vectors[artifacts.embedding_index[incoming_id]]),
                target_ids=frozenset(event.expected_target_ids),
                split="dev",
            )
        )

    selection = select_dev_configuration(documents, vectors, queries)
    payload = {
        "artifact_version": "v2-dev-retrieval-selection-1",
        "purpose": "configuration selection for a future confirmatory experiment",
        "confirmatory_result": False,
        "test_observations_used": 0,
        "eligible_dev_query_count": len(queries),
        "query_ids_sha256": sha256_payload([query.query_id for query in queries]),
        "input_hashes": {
            "evaluation_labels.csv": sha256_file(labels_path),
            "events.jsonl": sha256_file(events_path),
        },
        "selection_rule": ["maximize recall@5", "maximize MRR", "maximize recall@1", "prefer simpler mode", "stable identifier"],
        **selection.as_dict(),
    }
    output_dir = root / "reports/v2/development"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dev_retrieval_selection.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "output": str(output_path), "selected": selection.selected.identifier, "metrics": selection.selected_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
