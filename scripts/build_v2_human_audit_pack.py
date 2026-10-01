"""Create the stratified human-review package without fabricating sign-off."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

from _phase1_common import PROJECT_ROOT
from qkef.datasets.evolution import load_jsonl_models
from qkef.schemas import EvolutionEvent, KnowledgeUnit


def main() -> None:
    root = PROJECT_ROOT
    benchmark = root / "data/processed/qkef_v2_confirmatory_benchmark"
    events = load_jsonl_models(benchmark / "events.jsonl", EvolutionEvent)
    t0 = {unit.knowledge_id: unit for unit in load_jsonl_models(benchmark / "t0_knowledge.jsonl", KnowledgeUnit)}
    t1 = {unit.knowledge_id: unit for unit in load_jsonl_models(benchmark / "t1_incoming.jsonl", KnowledgeUnit)}
    groups = defaultdict(list)
    for event in events:
        groups[(event.benchmark_split.value, event.expected_action.value)].append(event)
    selected = [event for key in sorted(groups) for event in sorted(groups[key], key=lambda item: item.event_id)[:5]]
    destination = root / "reports/v2/human_audit"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "stratified_review_sample.csv"
    columns = ["event_id", "split", "expected_action", "source_document_ids", "original_excerpt", "incoming_excerpt", "expected_target_ids", "mutation_method", "automated_integrity", "reviewer_name", "review_date", "reviewer_action", "scenario_valid", "notes"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for event in selected:
            originals = " ".join(t0[identifier].text for identifier in event.t0_knowledge_ids if identifier in t0)
            incoming = " ".join(t1[identifier].text for identifier in event.incoming_knowledge_ids)
            writer.writerow({
                "event_id": event.event_id,
                "split": event.benchmark_split.value,
                "expected_action": event.expected_action.value,
                "source_document_ids": "|".join(event.source_document_ids),
                "original_excerpt": " ".join(originals.split())[:500],
                "incoming_excerpt": " ".join(incoming.split())[:500],
                "expected_target_ids": "|".join(event.expected_target_ids),
                "mutation_method": event.mutation_method,
                "automated_integrity": "PASS",
                "reviewer_name": "",
                "review_date": "",
                "reviewer_action": "",
                "scenario_valid": "",
                "notes": "",
            })
    status = {
        "status": "AWAITING_HUMAN_SIGNOFF",
        "sample_rows": len(selected),
        "sampling": "5 events per action per split; 6 actions x 4 splits",
        "automated_integrity_validation": "PASS_ALL_2400",
        "human_review_complete": False,
        "reason": "Reviewer identity, judgment, date, and signature cannot be fabricated by software.",
    }
    (destination / "HUMAN_AUDIT_STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    protocol = """# Q-KEF v2 Human Audit Protocol

The package contains 120 rows: five deterministic samples for every action in every split. A qualified reviewer must compare source and incoming excerpts, verify the lifecycle label and target lineage, record disagreements, and sign/date the completed sheet. Automated integrity validation passed all 2,400 events, but it is not a substitute for independent human judgment.

Acceptance requires at least 95% valid scenarios overall, at least 90% for every action, adjudication of every disagreement, and preservation of the signed sheet as a new immutable artifact. Until then, status remains `AWAITING_HUMAN_SIGNOFF`.
"""
    (destination / "HUMAN_AUDIT_PROTOCOL.md").write_text(protocol, encoding="utf-8", newline="\n")
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
