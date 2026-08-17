"""Build deterministic demo cases from successful stored TEST examples; never trains."""

from __future__ import annotations

import json
from pathlib import Path

from _phase1_common import PROJECT_ROOT
from qkef.datasets.evolution import load_jsonl_models
from qkef.datasets.fiqa import load_fiqa_dataset
from qkef.schemas import EvolutionEvent, IngestedKnowledgeUnit


SELECTED = {
    "NEW": "qkef-test-new-0010",
    "REPLACE": "qkef-test-replace-0003",
    "MERGE": "qkef-test-merge-0004",
    "ARCHIVE": "qkef-test-archive-0001",
    "COEXIST": "qkef-test-coexist-0007",
    "SPLIT": "qkef-test-split-0009",
}


def main() -> None:
    events = {item.event_id: item for item in load_jsonl_models(PROJECT_ROOT / "data/processed/qkef_fiqa_evolution/events.jsonl", EvolutionEvent)}
    units = load_jsonl_models(PROJECT_ROOT / "data/processed/qkef_fiqa_chunks/ingested_t0.jsonl", IngestedKnowledgeUnit)
    units += load_jsonl_models(PROJECT_ROOT / "data/processed/qkef_fiqa_chunks/ingested_t1.jsonl", IngestedKnowledgeUnit)
    lookup = {item.knowledge_id: item for item in units}
    fiqa = load_fiqa_dataset(PROJECT_ROOT / "data/raw/beir/fiqa")
    output = []
    for action, event_id in SELECTED.items():
        event = events[event_id]
        incoming = lookup[event.incoming_knowledge_ids[0]]
        target = lookup[event.expected_target_ids[0]] if event.expected_target_ids else None
        output.append({
            "name": f"Stored {action.title()} benchmark example",
            "action_metadata_for_benchmark_display_only": action,
            "event_id_for_provenance_only": event.event_id,
            "source_query_ids_for_provenance_only": event.source_query_ids,
            "qa_question": fiqa.queries[event.source_query_ids[0]].text if event.source_query_ids else "What evidence is available?",
            "incoming": {
                "identifier": incoming.knowledge_id, "text": incoming.model_text,
                "source_document_ids": incoming.source_document_ids,
                "text_origin": "synthetic administrative notice" if action == "ARCHIVE" else ("compound controlled example" if action in {"MERGE", "SPLIT"} else "controlled mutation"),
            },
            "existing": None if target is None else {
                "identifier": target.knowledge_id, "text": target.model_text,
                "source_document_ids": target.source_document_ids, "status": "ACTIVE", "text_origin": "original FiQA",
            },
            "disclosure": "This is a controlled temporal benchmark example, not a historical enterprise update.",
        })
    destination = PROJECT_ROOT / "data/demo/demo_examples.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(output)} deterministic examples to {destination}")


if __name__ == "__main__":
    main()
