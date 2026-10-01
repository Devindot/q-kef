"""Compare legacy and revised witness semantics on DEV only."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict

from _phase1_common import PROJECT_ROOT
from qkef.datasets.evolution import SPLIT_DELIMITER, load_jsonl_models
from qkef.runtime.artifacts import FinalArtifactLoader
from qkef.schemas import EvolutionAction, EvolutionEvent
from qkef.v2.canonical import sha256_file, sha256_payload
from qkef.v2.planning import TransitionPlanner
from qkef.v2.retrieval import HybridCandidateResolver
from qkef.v2.risk import CounterfactualEvaluator, witness_queries
from qkef.v2.state import KnowledgeState, V2KnowledgeRecord


def legacy_witness_queries(incoming_text: str, candidate_texts: list[str]) -> tuple[str, ...]:
    text = " ".join([incoming_text, *candidate_texts])
    tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]
    counts = Counter(tokens)
    ranked = sorted(counts, key=lambda token: (-counts[token], token))
    probes = [" ".join(ranked[index:index + 4]) for index in range(0, min(12, len(ranked)), 4)]
    return tuple(probe for probe in probes if probe) or (incoming_text[:160],)


def make_record(unit, artifacts) -> V2KnowledgeRecord:
    return V2KnowledgeRecord(
        unit.knowledge_id,
        unit.model_text,
        tuple(artifacts.embedding_vectors[artifacts.embedding_index[unit.knowledge_id]]),
        tuple(unit.source_document_ids),
        valid_from=unit.timestamp.isoformat(),
        system_from=unit.timestamp.isoformat(),
    )


def main() -> None:
    root = PROJECT_ROOT
    artifacts = FinalArtifactLoader(root).load()
    events_path = root / "data/processed/qkef_fiqa_evolution/events.jsonl"
    selection_path = root / "reports/v2/development/dev_retrieval_selection.json"
    events = [event for event in load_jsonl_models(events_path, EvolutionEvent) if event.benchmark_split.value == "dev"]
    units = {unit.knowledge_id: unit for unit in artifacts.units}
    t0_units = [unit for unit in artifacts.units if unit.temporal_state.value == "T0" and unit.benchmark_split.value == "dev"]
    base = KnowledgeState.from_records(make_record(unit, artifacts) for unit in t0_units)
    documents = {unit.knowledge_id: unit.model_text for unit in t0_units}
    vectors = {unit.knowledge_id: tuple(artifacts.embedding_vectors[artifacts.embedding_index[unit.knowledge_id]]) for unit in t0_units}
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    configuration = selection["selected"]
    resolver = HybridCandidateResolver(
        documents,
        vectors,
        rrf_k=configuration["rrf_k"],
        dense_weight=configuration["dense_weight"],
        lexical_weight=configuration["lexical_weight"],
    )
    planner = TransitionPlanner()
    evaluator = CounterfactualEvaluator()
    results = []
    for event in sorted(events, key=lambda item: item.event_id):
        incoming_unit = units[event.incoming_knowledge_ids[0]]
        incoming = make_record(incoming_unit, artifacts)
        ranked = [
            item.identifier
            for item in resolver.resolve(incoming.text, incoming.vector, 10, mode=configuration["mode"])
        ]
        if event.expected_action is EvolutionAction.NEW:
            predecessors = []
        elif event.expected_action is EvolutionAction.MERGE:
            predecessors = ranked[:2]
        elif event.expected_action is EvolutionAction.SPLIT:
            predecessors = []
        else:
            predecessors = ranked[:1]
        children = []
        if event.expected_action is EvolutionAction.SPLIT:
            segments = [segment.strip() for segment in incoming.text.split(SPLIT_DELIMITER) if segment.strip()]
            children = [
                V2KnowledgeRecord(identifier, text, incoming.vector, incoming.source_ids)
                for identifier, text in zip(event.expected_child_ids, segments)
            ]
        plan = planner.plan(base, event.expected_action, incoming, predecessors, split_children=children)
        candidate_texts = [base.records[identifier].text for identifier in predecessors]
        legacy = evaluator.evaluate(base, plan, incoming, 1.0, legacy_witness_queries(incoming.text, candidate_texts))
        revised = evaluator.evaluate(base, plan, incoming, 1.0, witness_queries(incoming.text, candidate_texts))
        results.append(
            {
                "action": event.expected_action.value,
                "legacy_probe_current_evidence_miss": legacy.current_evidence_miss,
                "revised_probe_current_evidence_miss": revised.current_evidence_miss,
            }
        )

    by_action = defaultdict(list)
    for result in results:
        by_action[result["action"]].append(result)

    def aggregate(rows: list[dict[str, object]]) -> dict[str, float | int]:
        return {
            "count": len(rows),
            "legacy_probe_mean_current_evidence_miss": statistics.fmean(row["legacy_probe_current_evidence_miss"] for row in rows),
            "revised_probe_mean_current_evidence_miss": statistics.fmean(row["revised_probe_current_evidence_miss"] for row in rows),
        }

    payload = {
        "artifact_version": "v2-dev-witness-audit-1",
        "purpose": "post-pilot witness development audit",
        "selection_split": "dev",
        "confirmatory_result": False,
        "test_observations_used": 0,
        "event_count": len(results),
        "event_ids_sha256": sha256_payload([event.event_id for event in sorted(events, key=lambda item: item.event_id)]),
        "input_hashes": {
            "events.jsonl": sha256_file(events_path),
            "dev_retrieval_selection.json": sha256_file(selection_path),
        },
        "shared_evaluator": "IDF-weighted query-term coverage over the post-transition active index",
        "comparison_scope": "probe construction comparison under revised DEV evaluator; not a reproduction of the frozen pilot stack",
        "overall": aggregate(results),
        "by_action": {action: aggregate(rows) for action, rows in sorted(by_action.items())},
        "interpretation": "DEV-only diagnostic; zero revised-probe miss on this controlled split does not establish out-of-sample performance.",
    }
    output_path = root / "reports/v2/development/dev_witness_audit.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "output": str(output_path), "overall": payload["overall"]}, sort_keys=True))


if __name__ == "__main__":
    main()
