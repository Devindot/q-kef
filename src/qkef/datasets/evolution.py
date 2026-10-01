"""Deterministic construction and validation of the Q-KEF FiQA benchmark."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from qkef.datasets.fiqa import FiqaDataset, FiqaDocument
from qkef.schemas import (
    BenchmarkSplit,
    EvolutionAction,
    EvolutionEvent,
    EvolutionRelationType,
    KnowledgeUnit,
    LifecycleStatus,
)


MERGE_DELIMITER = "\n\nAdditional related information:\n"
SPLIT_DELIMITER = "\n\n--- KNOWLEDGE SEGMENT BOUNDARY ---\n\n"
ARCHIVE_TEMPLATE_VERSION = "1.0"
COMMON_BENCHMARK_FILES = (
    "t0_knowledge.jsonl",
    "t1_incoming.jsonl",
    "events.jsonl",
    "review_sample.csv",
    "benchmark_manifest.json",
)


def parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("benchmark timestamps must include a timezone")
    return result.astimezone(timezone.utc)


@dataclass(frozen=True)
class EvolutionBenchmarkConfig:
    seed: int
    generator_version: str
    action_definitions_version: str
    events_per_action: int
    split_targets: dict[str, int]
    t0_timestamp: datetime
    t1_timestamp: datetime
    actions: tuple[EvolutionAction, ...]
    benchmark_name: str = "qkef-fiqa-evolution"
    benchmark_version: str = "1.0"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "EvolutionBenchmarkConfig":
        dataset = config["dataset"]
        evolution = config["evolution_benchmark"]
        split_targets = {
            "train": int(evolution["train_events_per_action"]),
            "dev": int(evolution["dev_events_per_action"]),
        }
        if "calibration_events_per_action" in evolution:
            split_targets["calibration"] = int(evolution["calibration_events_per_action"])
        split_targets["test"] = int(evolution["test_events_per_action"])
        events_per_action = int(evolution["events_per_action"])
        if sum(split_targets.values()) != events_per_action:
            raise ValueError("per-split event targets must sum to events_per_action")
        actions = tuple(EvolutionAction(item) for item in evolution["actions"])
        if len(actions) != len(set(actions)):
            raise ValueError("evolution benchmark actions must be unique")
        t0_timestamp = parse_timestamp(evolution["t0_timestamp"])
        t1_timestamp = parse_timestamp(evolution["t1_timestamp"])
        if t1_timestamp <= t0_timestamp:
            raise ValueError("T1 must be later than T0")
        return cls(
            seed=int(evolution["seed"]),
            generator_version=str(evolution["generator_version"]),
            action_definitions_version=str(evolution.get("action_definitions_version", "1.0")),
            events_per_action=events_per_action,
            split_targets=split_targets,
            t0_timestamp=t0_timestamp,
            t1_timestamp=t1_timestamp,
            actions=actions,
            benchmark_name=str(dataset["temporal_variant"]),
            benchmark_version=str(evolution.get("benchmark_version", "1.0")),
        )


@dataclass
class BenchmarkContent:
    t0_units: list[KnowledgeUnit]
    t1_units: list[KnowledgeUnit]
    events: list[EvolutionEvent]
    review_csv: str
    statistics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    split_names: tuple[str, ...] = ("train", "dev", "test")


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class NumericMutation:
    text: str
    token_type: str
    old_value: str
    new_value: str
    start: int
    end: int
    rule: str


_NUMERIC_PATTERN = re.compile(
    r"(?P<currency>[$€£]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?P<percent>\b\d+(?:\.\d+)?%)"
    r"|(?P<year>\b(?:19|20)\d{2}\b)"
    r"|(?P<number>\b\d+(?:\.\d+)?\b)"
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def stable_order(values: Iterable[Any], seed: int, namespace: str, key: Callable[[Any], str]) -> list[Any]:
    """Order values with a stable hash instead of process-dependent randomness."""

    def digest(value: Any) -> str:
        payload = f"{seed}|{namespace}|{key(value)}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return sorted(values, key=lambda value: (digest(value), key(value)))


def normalize_for_comparison(text: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall(text.lower()))


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_PATTERN.findall(left.lower()))
    right_tokens = set(_TOKEN_PATTERN.findall(right.lower()))
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def mutate_numeric_token(text: str) -> NumericMutation | None:
    """Apply one conservative, auditable numeric-token mutation."""

    match = _NUMERIC_PATTERN.search(text)
    if match is None:
        return None
    token_type = match.lastgroup or "number"
    old_value = match.group(0)
    if token_type == "currency":
        prefix_match = re.match(r"([$€£]\s?)(.*)", old_value)
        if prefix_match is None:
            return None
        prefix, numeric = prefix_match.groups()
        cleaned = numeric.replace(",", "")
        decimals = len(cleaned.partition(".")[2]) if "." in cleaned else 0
        changed = float(cleaned) + max(1.0, round(float(cleaned) * 0.1, decimals))
        rendered = f"{changed:,.{decimals}f}" if decimals else f"{int(round(changed)):,}"
        new_value = prefix + rendered
        rule = "increase_currency_by_maximum_of_ten_percent_or_one"
    elif token_type == "percent":
        numeric = old_value[:-1]
        decimals = len(numeric.partition(".")[2]) if "." in numeric else 0
        changed = float(numeric) + 1.0
        rendered = f"{changed:.{decimals}f}" if decimals else str(int(changed))
        new_value = rendered + "%"
        rule = "increase_percentage_by_one_point"
    elif token_type == "year":
        new_value = str(int(old_value) + 1)
        rule = "advance_year_by_one"
    else:
        decimals = len(old_value.partition(".")[2]) if "." in old_value else 0
        changed = float(old_value) + 1.0
        new_value = f"{changed:.{decimals}f}" if decimals else str(int(changed))
        rule = "increase_numeric_value_by_one"
    mutated = text[: match.start()] + new_value + text[match.end() :]
    if mutated == text:
        return None
    return NumericMutation(
        text=mutated,
        token_type=token_type,
        old_value=old_value,
        new_value=new_value,
        start=match.start(),
        end=match.end(),
        rule=rule,
    )


def select_source_segment(text: str, maximum_length: int = 500) -> tuple[str, int, int] | None:
    """Copy a deterministic leading sentence/segment without paraphrasing it."""

    if not text:
        return None
    limit = min(len(text), maximum_length)
    candidates = [match.end() for match in re.finditer(r"[.!?](?:\s|$)", text[:limit])]
    end = next((candidate for candidate in candidates if candidate >= 40), limit)
    segment = text[:end]
    if not segment.strip():
        return None
    return segment, 0, end


def _unit_text_metadata(document: FiqaDocument) -> dict[str, Any]:
    return {"fiqa_document_id": document.document_id, "fiqa_title": document.title}


def _make_t0_unit(event_id: str, document: FiqaDocument, split: str, config: EvolutionBenchmarkConfig) -> KnowledgeUnit:
    return KnowledgeUnit(
        knowledge_id=f"{event_id}-t0",
        source_document_id=document.document_id,
        text=document.text,
        source="BEIR FiQA (original T0 source)",
        document_version="T0-v1",
        timestamp=config.t0_timestamp,
        ingestion_timestamp=config.t0_timestamp,
        metadata={
            **_unit_text_metadata(document),
            "benchmark_split": split,
            "text_origin": "original_fiqa",
            "synthetic_timestamp": True,
        },
        lifecycle_status=LifecycleStatus.ACTIVE,
    )


def _make_t1_unit(
    event_id: str,
    source_document_id: str,
    text: str,
    source: str,
    split: str,
    action: EvolutionAction,
    metadata: dict[str, Any],
    config: EvolutionBenchmarkConfig,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        knowledge_id=f"{event_id}-t1",
        source_document_id=source_document_id,
        text=text,
        source=source,
        document_version="T1-v1",
        timestamp=config.t1_timestamp,
        ingestion_timestamp=config.t1_timestamp,
        metadata={
            **metadata,
            "benchmark_split": split,
            "expected_action": action.value,
            "synthetic_timestamp": True,
            "generator_version": config.generator_version,
        },
        lifecycle_status=LifecycleStatus.ACTIVE,
        evolution_action=action,
    )


def _event(
    event_id: str,
    action: EvolutionAction,
    split: str,
    t0_ids: list[str],
    incoming_ids: list[str],
    target_ids: list[str],
    child_ids: list[str],
    source_document_ids: list[str],
    source_query_ids: list[str],
    relation: EvolutionRelationType,
    method: str,
    parameters: dict[str, Any],
    rationale: str,
    config: EvolutionBenchmarkConfig,
) -> EvolutionEvent:
    return EvolutionEvent(
        event_id=event_id,
        expected_action=action,
        benchmark_split=BenchmarkSplit(split),
        source_qrels_split=split,
        t0_knowledge_ids=t0_ids,
        incoming_knowledge_ids=incoming_ids,
        expected_target_ids=target_ids,
        expected_child_ids=child_ids,
        source_document_ids=source_document_ids,
        source_query_ids=source_query_ids,
        relation_type=relation,
        mutation_method=method,
        mutation_parameters=parameters,
        generator_version=config.generator_version,
        seed=config.seed,
        t0_timestamp=config.t0_timestamp,
        t1_timestamp=config.t1_timestamp,
        requires_human_review=True,
        automatic_rationale=rationale,
        metadata={"construction_is_synthetic": True},
    )


def _source_assignments(dataset: FiqaDataset, allowed_splits: set[str]) -> tuple[dict[str, str], set[str]]:
    memberships: dict[str, set[str]] = defaultdict(set)
    for qrel in dataset.qrels:
        if qrel.relevance > 0 and qrel.source_split in allowed_splits:
            memberships[qrel.document_id].add(qrel.source_split)
    assigned = {document_id: next(iter(splits)) for document_id, splits in memberships.items() if len(splits) == 1}
    conflicts = {document_id for document_id, splits in memberships.items() if len(splits) > 1}
    return assigned, conflicts


def _query_pools(dataset: FiqaDataset, assignments: Mapping[str, str], allowed_splits: set[str]) -> dict[str, dict[str, list[str]]]:
    pools: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for qrel in dataset.qrels:
        if qrel.relevance <= 0 or assignments.get(qrel.document_id) != qrel.source_split:
            continue
        if qrel.source_split not in allowed_splits:
            continue
        if not dataset.documents[qrel.document_id].text.strip():
            continue
        pools[qrel.source_split][qrel.query_id].add(qrel.document_id)
    return {
        split: {query: sorted(document_ids) for query, document_ids in sorted(queries.items())}
        for split, queries in pools.items()
    }


def _document_queries(query_pool: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for query_id, document_ids in query_pool.items():
        for document_id in document_ids:
            result[document_id].append(query_id)
    return {key: sorted(value) for key, value in result.items()}


def _same_query_pairs(
    query_pool: Mapping[str, Sequence[str]], seed: int, namespace: str
) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for query_id, document_ids in query_pool.items():
        for left_index, left in enumerate(document_ids):
            for right in document_ids[left_index + 1 :]:
                pairs.append((query_id, left, right))
    return stable_order(pairs, seed, namespace, lambda item: "|".join(item))


def _all_documents(query_pool: Mapping[str, Sequence[str]]) -> list[str]:
    return sorted({document_id for values in query_pool.values() for document_id in values})


def _build_review_csv(
    events: Sequence[EvolutionEvent],
    t0_by_id: Mapping[str, KnowledgeUnit],
    t1_by_id: Mapping[str, KnowledgeUnit],
) -> str:
    columns = [
        "event_id",
        "benchmark_split",
        "expected_action",
        "source_document_ids",
        "original_text_excerpt",
        "incoming_text_excerpt",
        "expected_target_ids",
        "mutation_method",
        "automatic_rationale",
        "reviewer_action",
        "reviewer_valid",
        "reviewer_notes",
    ]
    selected: list[EvolutionEvent] = []
    for action in EvolutionAction:
        candidates = sorted((event for event in events if event.expected_action is action), key=lambda event: event.event_id)
        selected.extend(candidates[:5])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for event in selected:
        originals = [t0_by_id[item].text for item in event.t0_knowledge_ids if item in t0_by_id]
        incoming = [t1_by_id[item].text for item in event.incoming_knowledge_ids]
        writer.writerow(
            {
                "event_id": event.event_id,
                "benchmark_split": event.benchmark_split.value,
                "expected_action": event.expected_action.value,
                "source_document_ids": "|".join(event.source_document_ids),
                "original_text_excerpt": " ".join(" ".join(originals).split())[:240],
                "incoming_text_excerpt": " ".join(" ".join(incoming).split())[:240],
                "expected_target_ids": "|".join(event.expected_target_ids),
                "mutation_method": event.mutation_method,
                "automatic_rationale": event.automatic_rationale,
                "reviewer_action": "",
                "reviewer_valid": "",
                "reviewer_notes": "",
            }
        )
    return stream.getvalue()


def generate_benchmark_content(dataset: FiqaDataset, config: EvolutionBenchmarkConfig) -> BenchmarkContent:
    """Generate ground-truth content without writing or modifying FiQA sources."""

    allowed_splits = set(config.split_targets)
    assignments, conflicting_documents = _source_assignments(dataset, allowed_splits)
    pools = _query_pools(dataset, assignments, allowed_splits)
    t0_units: list[KnowledgeUnit] = []
    t1_units: list[KnowledgeUnit] = []
    events: list[EvolutionEvent] = []
    warnings: list[str] = []
    rejected_near_duplicates = 0
    eligible_count = len(assignments)

    for split in config.split_targets:
        target = config.split_targets.get(split, 0)
        if target == 0:
            continue
        query_pool = pools.get(split, {})
        document_queries = _document_queries(query_pool)
        used_documents: set[str] = set()
        action_counts: Counter[EvolutionAction] = Counter()

        def next_event_id(action: EvolutionAction) -> str:
            return f"qkef-{split}-{action.value.lower()}-{action_counts[action] + 1:04d}"

        # Pair-dependent classes run first so single-document classes cannot exhaust them.
        for action in (EvolutionAction.MERGE, EvolutionAction.COEXIST):
            if action not in config.actions:
                continue
            pairs = _same_query_pairs(query_pool, config.seed, f"{split}-{action.value}")
            for query_id, left_id, right_id in pairs:
                if action_counts[action] >= target:
                    break
                if left_id in used_documents or right_id in used_documents:
                    continue
                left = dataset.documents[left_id]
                right = dataset.documents[right_id]
                if normalize_for_comparison(left.text) == normalize_for_comparison(right.text):
                    rejected_near_duplicates += 1
                    continue
                overlap = token_jaccard(left.text, right.text)
                if overlap >= 0.9:
                    rejected_near_duplicates += 1
                    continue
                event_id = next_event_id(action)
                t0 = _make_t0_unit(event_id, left, split, config)
                if action is EvolutionAction.MERGE:
                    selected = select_source_segment(right.text)
                    if selected is None:
                        continue
                    segment, segment_start, segment_end = selected
                    incoming_text = left.text + MERGE_DELIMITER + segment
                    method = "copy_base_and_append_source_segment_v1"
                    relation = EvolutionRelationType.COMPLEMENTARY
                    parameters = {
                        "base_source_document_id": left_id,
                        "additional_source_document_id": right_id,
                        "additional_span_start": segment_start,
                        "additional_span_end": segment_end,
                        "delimiter": MERGE_DELIMITER,
                        "lexical_jaccard": overlap,
                    }
                    rationale = (
                        f"Incoming unit retains base content from {left_id} and adds source material "
                        f"from {right_id} associated with FiQA query {query_id}."
                    )
                    origin = "deterministic_merge_construction"
                else:
                    incoming_text = right.text
                    method = "same_query_original_document_v1"
                    relation = EvolutionRelationType.COEXISTENCE
                    parameters = {
                        "existing_source_document_id": left_id,
                        "incoming_source_document_id": right_id,
                        "shared_query_id": query_id,
                        "lexical_jaccard": overlap,
                        "near_duplicate_threshold": 0.9,
                    }
                    rationale = (
                        f"Original FiQA documents {left_id} and {right_id} are relevant to query "
                        f"{query_id}, are lexically distinct, and are defined to remain active."
                    )
                    origin = "original_fiqa"
                t1 = _make_t1_unit(
                    event_id,
                    right_id if action is EvolutionAction.COEXIST else left_id,
                    incoming_text,
                    "BEIR FiQA controlled Phase 1 construction",
                    split,
                    action,
                    {
                        "text_origin": origin,
                        "source_document_ids": [left_id, right_id],
                        "source_query_ids": [query_id],
                    },
                    config,
                )
                event = _event(
                    event_id,
                    action,
                    split,
                    [t0.knowledge_id],
                    [t1.knowledge_id],
                    [t0.knowledge_id],
                    [],
                    [left_id, right_id],
                    [query_id],
                    relation,
                    method,
                    parameters,
                    rationale,
                    config,
                )
                t0_units.append(t0)
                t1_units.append(t1)
                events.append(event)
                used_documents.update((left_id, right_id))
                action_counts[action] += 1

        if EvolutionAction.SPLIT in config.actions:
            entries = [
                (document_id, document_queries[document_id][0])
                for document_id in _all_documents(query_pool)
                if document_queries.get(document_id)
            ]
            entries = stable_order(entries, config.seed, f"{split}-SPLIT", lambda item: "|".join(item))
            while action_counts[EvolutionAction.SPLIT] < target:
                chosen: tuple[tuple[str, str], tuple[str, str]] | None = None
                for left_index, left in enumerate(entries):
                    if left[0] in used_documents:
                        continue
                    for right in entries[left_index + 1 :]:
                        if right[0] in used_documents or left[1] == right[1]:
                            continue
                        chosen = (left, right)
                        break
                    if chosen:
                        break
                if chosen is None:
                    break
                (left_id, left_query), (right_id, right_query) = chosen
                left = dataset.documents[left_id]
                right = dataset.documents[right_id]
                event_id = next_event_id(EvolutionAction.SPLIT)
                incoming_text = left.text + SPLIT_DELIMITER + right.text
                t1 = _make_t1_unit(
                    event_id,
                    event_id,
                    incoming_text,
                    "Q-KEF deterministic compound construction",
                    split,
                    EvolutionAction.SPLIT,
                    {
                        "text_origin": "deterministic_compound_split",
                        "source_document_ids": [left_id, right_id],
                        "source_query_ids": [left_query, right_query],
                    },
                    config,
                )
                child_ids = [f"{event_id}-child-01", f"{event_id}-child-02"]
                parameters = {
                    "component_source_document_ids": [left_id, right_id],
                    "component_query_ids": [left_query, right_query],
                    "delimiter": SPLIT_DELIMITER,
                    "expected_children": child_ids,
                    "component_spans": [
                        [0, len(left.text)],
                        [len(left.text) + len(SPLIT_DELIMITER), len(incoming_text)],
                    ],
                }
                event = _event(
                    event_id,
                    EvolutionAction.SPLIT,
                    split,
                    [],
                    [t1.knowledge_id],
                    [],
                    child_ids,
                    [left_id, right_id],
                    [left_query, right_query],
                    EvolutionRelationType.COMPOUND_SPLIT,
                    "join_distinct_query_documents_with_boundary_v1",
                    parameters,
                    f"Incoming compound copies documents {left_id} and {right_id} from distinct FiQA queries and defines two expected components.",
                    config,
                )
                t1_units.append(t1)
                events.append(event)
                used_documents.update((left_id, right_id))
                action_counts[EvolutionAction.SPLIT] += 1

        documents = stable_order(
            _all_documents(query_pool), config.seed, f"{split}-single-documents", lambda item: item
        )
        for action in (EvolutionAction.REPLACE, EvolutionAction.ARCHIVE, EvolutionAction.NEW):
            if action not in config.actions:
                continue
            for document_id in documents:
                if action_counts[action] >= target:
                    break
                if document_id in used_documents:
                    continue
                document = dataset.documents[document_id]
                event_id = next_event_id(action)
                query_ids = document_queries.get(document_id, [])
                if action is EvolutionAction.REPLACE:
                    mutation = mutate_numeric_token(document.text)
                    if mutation is None:
                        continue
                    t0 = _make_t0_unit(event_id, document, split, config)
                    parameters = {
                        "parent_source_document_id": document_id,
                        "token_type": mutation.token_type,
                        "old_value": mutation.old_value,
                        "new_value": mutation.new_value,
                        "span_start": mutation.start,
                        "span_end": mutation.end,
                        "mutation_rule": mutation.rule,
                    }
                    t1 = _make_t1_unit(
                        event_id,
                        document_id,
                        mutation.text,
                        "BEIR FiQA deterministic numeric mutation",
                        split,
                        action,
                        {
                            "text_origin": "deterministic_numeric_mutation",
                            "source_document_ids": [document_id],
                            "mutation": parameters,
                        },
                        config,
                    )
                    event = _event(
                        event_id,
                        action,
                        split,
                        [t0.knowledge_id],
                        [t1.knowledge_id],
                        [t0.knowledge_id],
                        [],
                        [document_id],
                        query_ids,
                        EvolutionRelationType.SUPERSEDES,
                        "single_numeric_token_mutation_v1",
                        parameters,
                        f"Incoming unit is a controlled numeric mutation of source document {document_id} and is defined as its newer synthetic version.",
                        config,
                    )
                    t0_units.append(t0)
                elif action is EvolutionAction.ARCHIVE:
                    t0 = _make_t0_unit(event_id, document, split, config)
                    notice = (
                        "Administrative knowledge-base notice.\n\n"
                        f"Target knowledge ID: {t0.knowledge_id}\n\n"
                        "The referenced knowledge item has been withdrawn and should no longer be treated as active."
                    )
                    parameters = {
                        "target_knowledge_id": t0.knowledge_id,
                        "source_document_id": document_id,
                        "notice_template_version": ARCHIVE_TEMPLATE_VERSION,
                    }
                    t1 = _make_t1_unit(
                        event_id,
                        event_id,
                        notice,
                        "Q-KEF synthetic administrative notice",
                        split,
                        action,
                        {
                            "text_origin": "synthetic_administrative_notice",
                            "source_document_ids": [document_id],
                            "notice_template_version": ARCHIVE_TEMPLATE_VERSION,
                        },
                        config,
                    )
                    event = _event(
                        event_id,
                        action,
                        split,
                        [t0.knowledge_id],
                        [t1.knowledge_id],
                        [t0.knowledge_id],
                        [],
                        [document_id],
                        query_ids,
                        EvolutionRelationType.RETRACTION,
                        "administrative_retraction_notice_v1",
                        parameters,
                        f"Incoming unit is an administrative retraction notice targeting knowledge unit {t0.knowledge_id}.",
                        config,
                    )
                    t0_units.append(t0)
                else:
                    t1 = _make_t1_unit(
                        event_id,
                        document_id,
                        document.text,
                        "BEIR FiQA original source introduced at T1",
                        split,
                        action,
                        {
                            **_unit_text_metadata(document),
                            "text_origin": "original_fiqa",
                            "source_document_ids": [document_id],
                        },
                        config,
                    )
                    event = _event(
                        event_id,
                        action,
                        split,
                        [],
                        [t1.knowledge_id],
                        [],
                        [],
                        [document_id],
                        query_ids,
                        EvolutionRelationType.NEW,
                        "introduce_original_fiqa_document_v1",
                        {"source_document_id": document_id, "text_preserved": True},
                        f"Original FiQA document {document_id} is introduced at T1 with no predecessor in the event's T0 state.",
                        config,
                    )
                t1_units.append(t1)
                events.append(event)
                used_documents.add(document_id)
                action_counts[action] += 1

        for action in config.actions:
            actual = action_counts[action]
            if actual < target:
                warnings.append(
                    f"{split}/{action.value}: generated {actual} of {target} requested events because eligible sources were exhausted"
                )

    t0_units.sort(key=lambda unit: unit.knowledge_id)
    t1_units.sort(key=lambda unit: unit.knowledge_id)
    events.sort(key=lambda event: event.event_id)
    t0_by_id = {unit.knowledge_id: unit for unit in t0_units}
    t1_by_id = {unit.knowledge_id: unit for unit in t1_units}
    review_csv = _build_review_csv(events, t0_by_id, t1_by_id)
    counts_by_action = Counter(event.expected_action.value for event in events)
    counts_by_split = Counter(event.benchmark_split.value for event in events)
    statistics = {
        "eligible_exclusive_source_documents": eligible_count,
        "excluded_unreferenced_or_nonpositive_documents": len(dataset.documents)
        - eligible_count
        - len(conflicting_documents),
        "excluded_cross_qrels_split_documents": len(conflicting_documents),
        "excluded_empty_text_documents": sum(1 for doc in dataset.documents.values() if not doc.text.strip()),
        "rejected_duplicate_or_near_duplicate_pairs": rejected_near_duplicates,
        "count_per_action": dict(sorted(counts_by_action.items())),
        "count_per_split": dict(sorted(counts_by_split.items())),
        "unique_source_document_count": len({item for event in events for item in event.source_document_ids}),
        "review_sample_count": sum(1 for line in review_csv.splitlines()[1:] if line),
    }
    return BenchmarkContent(t0_units, t1_units, events, review_csv, statistics, warnings, tuple(config.split_targets))


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonl(values: Iterable[Any]) -> str:
    return "".join(canonical_json(value) + "\n" for value in values)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_hashes(dataset: FiqaDataset) -> dict[str, str]:
    paths = [dataset.root / "corpus.jsonl", dataset.root / "queries.jsonl"]
    paths.extend(sorted((dataset.root / "qrels").glob("*.tsv"), key=lambda item: item.name))
    return {path.relative_to(dataset.root).as_posix(): sha256_file(path) for path in paths}


def render_content_files(content: BenchmarkContent) -> dict[str, str]:
    files = {
        "t0_knowledge.jsonl": _jsonl(content.t0_units),
        "t1_incoming.jsonl": _jsonl(content.t1_units),
        "events.jsonl": _jsonl(content.events),
        "review_sample.csv": content.review_csv,
    }
    for split in content.split_names:
        files[f"{split}_events.jsonl"] = _jsonl(
            event for event in content.events if event.benchmark_split.value == split
        )
    return files


def content_signature(content: BenchmarkContent) -> str:
    files = render_content_files(content)
    payload = "".join(f"{name}\0{files[name]}\0" for name in sorted(files))
    return sha256_bytes(payload.encode("utf-8"))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def build_benchmark(
    dataset: FiqaDataset,
    config: EvolutionBenchmarkConfig,
    output_dir: Path,
    *,
    expected_fiqa_md5: str,
    archive_path: Path | None = None,
    verify_determinism: bool = True,
) -> tuple[dict[str, Any], BenchmarkContent]:
    """Generate, determinism-check, and atomically write benchmark files."""

    content = generate_benchmark_content(dataset, config)
    deterministic_status: bool | None = None
    if verify_determinism:
        rebuilt = generate_benchmark_content(dataset, config)
        deterministic_status = content_signature(content) == content_signature(rebuilt)
        if not deterministic_status:
            raise RuntimeError("deterministic rebuild content hash mismatch")
    rendered = render_content_files(content)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        _atomic_write(output_dir / name, text)
    archive_md5 = md5_file(archive_path) if archive_path and archive_path.exists() else None
    file_hashes = {
        name: sha256_bytes(text.encode("utf-8")) for name, text in sorted(rendered.items())
    }
    action_counts = Counter(event.expected_action.value for event in content.events)
    split_counts = Counter(event.benchmark_split.value for event in content.events)
    manifest = {
        "benchmark_name": config.benchmark_name,
        "benchmark_version": config.benchmark_version,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fiqa_expected_md5": expected_fiqa_md5,
        "fiqa_observed_md5": archive_md5,
        "fiqa_checksum_verified": archive_md5 == expected_fiqa_md5 if archive_md5 else None,
        "generator_version": config.generator_version,
        "action_definitions_version": config.action_definitions_version,
        "seed": config.seed,
        "requested_number_of_events": config.events_per_action * len(config.actions),
        "actual_number_of_events": len(content.events),
        "count_per_action": {action.value: action_counts[action.value] for action in config.actions},
        "count_per_benchmark_split": {split: split_counts[split] for split in content.split_names},
        "source_qrels_split_mapping": {split: split for split in content.split_names},
        "t0_timestamp": config.t0_timestamp.isoformat().replace("+00:00", "Z"),
        "t1_timestamp": config.t1_timestamp.isoformat().replace("+00:00", "Z"),
        "t0_knowledge_unit_count": len(content.t0_units),
        "generated_t1_count": len(content.t1_units),
        "unique_source_document_count": content.statistics["unique_source_document_count"],
        "leakage_validation_status": "pending_validation",
        "deterministic_regeneration_status": deterministic_status,
        "deterministic_content_sha256": content_signature(content),
        "generated_file_sha256": file_hashes,
        "source_file_sha256": source_file_hashes(dataset),
        "warnings": content.warnings,
        "generation_statistics": content.statistics,
        "generation_configuration": {
            "events_per_action": config.events_per_action,
            "split_targets_per_action": config.split_targets,
            "actions": [action.value for action in config.actions],
        },
    }
    _atomic_write(output_dir / "benchmark_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest, content


def load_jsonl_models(path: Path, model: type[Any]) -> list[Any]:
    values: list[Any] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                values.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: schema validation failed: {exc}") from exc
    return values


def validate_benchmark(
    benchmark_dir: Path,
    dataset: FiqaDataset,
    config: EvolutionBenchmarkConfig,
    *,
    archive_path: Path | None = None,
    expected_md5: str | None = None,
) -> ValidationResult:
    """Perform reference, provenance, temporal, leakage, and source-integrity checks."""

    result = ValidationResult()
    required_files = (*COMMON_BENCHMARK_FILES, *(f"{split}_events.jsonl" for split in config.split_targets))
    missing = [name for name in required_files if not (benchmark_dir / name).exists()]
    if missing:
        result.errors.append(f"missing required files: {', '.join(missing)}")
        return result
    try:
        manifest = json.loads((benchmark_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
        t0_units = load_jsonl_models(benchmark_dir / "t0_knowledge.jsonl", KnowledgeUnit)
        t1_units = load_jsonl_models(benchmark_dir / "t1_incoming.jsonl", KnowledgeUnit)
        events = load_jsonl_models(benchmark_dir / "events.jsonl", EvolutionEvent)
    except Exception as exc:
        result.errors.append(str(exc))
        return result

    def unique_map(values: Sequence[Any], attribute: str, label: str) -> dict[str, Any]:
        mapped: dict[str, Any] = {}
        for value in values:
            identifier = getattr(value, attribute)
            if identifier in mapped:
                result.errors.append(f"duplicate {label}: {identifier}")
            mapped[identifier] = value
        return mapped

    t0_by_id = unique_map(t0_units, "knowledge_id", "T0 knowledge ID")
    t1_by_id = unique_map(t1_units, "knowledge_id", "T1 knowledge ID")
    event_by_id = unique_map(events, "event_id", "event ID")
    if set(t0_by_id) & set(t1_by_id):
        result.errors.append("T0 and T1 knowledge IDs overlap")

    ancestry: dict[str, set[str]] = defaultdict(set)
    query_ancestry: dict[str, set[str]] = defaultdict(set)
    for event in events:
        split = event.benchmark_split.value
        ancestry[split].update(event.source_document_ids)
        query_ancestry[split].update(event.source_query_ids)
        for reference in event.t0_knowledge_ids + event.expected_target_ids:
            if reference not in t0_by_id:
                result.errors.append(f"{event.event_id}: unresolved T0 reference {reference}")
        for reference in event.incoming_knowledge_ids:
            if reference not in t1_by_id:
                result.errors.append(f"{event.event_id}: unresolved T1 reference {reference}")
        unknown_sources = set(event.source_document_ids) - set(dataset.documents)
        if unknown_sources:
            result.errors.append(f"{event.event_id}: unknown FiQA source IDs {sorted(unknown_sources)}")
        if event.t1_timestamp <= event.t0_timestamp:
            result.errors.append(f"{event.event_id}: T1 is not later than T0")
        incoming = t1_by_id.get(event.incoming_knowledge_ids[0])
        if incoming is None:
            continue
        if event.expected_action is EvolutionAction.NEW:
            if event.t0_knowledge_ids or event.expected_target_ids:
                result.errors.append(f"{event.event_id}: NEW has a predecessor/target")
            source = dataset.documents[event.source_document_ids[0]]
            if incoming.text != source.text:
                result.errors.append(f"{event.event_id}: NEW did not preserve source text")
        elif event.expected_action in {EvolutionAction.REPLACE, EvolutionAction.MERGE, EvolutionAction.ARCHIVE}:
            if not event.expected_target_ids:
                result.errors.append(f"{event.event_id}: action requires a T0 target")
        elif event.expected_action is EvolutionAction.COEXIST:
            if len(event.source_document_ids) < 2:
                result.errors.append(f"{event.event_id}: COEXIST requires two source documents")
            elif incoming.text != dataset.documents[event.source_document_ids[1]].text:
                result.errors.append(f"{event.event_id}: COEXIST incoming text changed")
            if event.relation_type is not EvolutionRelationType.COEXISTENCE:
                result.errors.append(f"{event.event_id}: COEXIST relation is incorrect")

        if event.expected_action is EvolutionAction.REPLACE and event.t0_knowledge_ids:
            original = t0_by_id[event.t0_knowledge_ids[0]].text
            params = event.mutation_parameters
            required = {"old_value", "new_value", "span_start", "span_end", "mutation_rule"}
            if not required.issubset(params):
                result.errors.append(f"{event.event_id}: incomplete REPLACE mutation provenance")
            else:
                start, end = int(params["span_start"]), int(params["span_end"])
                if original[start:end] != params["old_value"]:
                    result.errors.append(f"{event.event_id}: recorded old token/span mismatch")
                reconstructed = original[:start] + str(params["new_value"]) + original[end:]
                if reconstructed != incoming.text or incoming.text == original:
                    result.errors.append(f"{event.event_id}: REPLACE text is not the recorded single mutation")
        elif event.expected_action is EvolutionAction.MERGE and event.t0_knowledge_ids:
            base = t0_by_id[event.t0_knowledge_ids[0]].text
            params = event.mutation_parameters
            additional_id = params.get("additional_source_document_id")
            if additional_id not in dataset.documents:
                result.errors.append(f"{event.event_id}: MERGE additional source is invalid")
            else:
                source_text = dataset.documents[additional_id].text
                start = int(params.get("additional_span_start", -1))
                end = int(params.get("additional_span_end", -1))
                segment = source_text[start:end] if 0 <= start <= end <= len(source_text) else None
                expected_text = base + str(params.get("delimiter", "")) + (segment or "")
                if not segment or incoming.text != expected_text:
                    result.errors.append(f"{event.event_id}: MERGE content/provenance mismatch")
        elif event.expected_action is EvolutionAction.ARCHIVE:
            target = event.expected_target_ids[0] if event.expected_target_ids else ""
            if event.mutation_parameters.get("target_knowledge_id") != target or target not in incoming.text:
                result.errors.append(f"{event.event_id}: ARCHIVE notice does not identify its target")
        elif event.expected_action is EvolutionAction.SPLIT:
            if len(event.source_document_ids) < 2 or len(event.expected_child_ids) < 2:
                result.errors.append(f"{event.event_id}: SPLIT lacks source components/children")
            else:
                left = dataset.documents[event.source_document_ids[0]].text
                right = dataset.documents[event.source_document_ids[1]].text
                if incoming.text != left + SPLIT_DELIMITER + right:
                    result.errors.append(f"{event.event_id}: SPLIT compound text mismatch")

    split_pairs = tuple(combinations(config.split_targets, 2))
    for left, right in split_pairs:
        overlap = ancestry[left] & ancestry[right]
        if overlap:
            result.errors.append(f"source ancestry leakage {left}/{right}: {len(overlap)} documents")
        query_overlap = query_ancestry[left] & query_ancestry[right]
        if query_overlap:
            result.warnings.append(f"query overlap {left}/{right}: {len(query_overlap)} queries")

    for unit in t0_units:
        source = dataset.documents.get(unit.source_document_id)
        if source is None or unit.text != source.text:
            result.errors.append(f"{unit.knowledge_id}: T0 source text integrity failure")

    actual_action_counts = Counter(event.expected_action.value for event in events)
    actual_split_counts = Counter(event.benchmark_split.value for event in events)
    for action in config.actions:
        actual = actual_action_counts[action.value]
        if actual != config.events_per_action:
            result.warnings.append(
                f"{action.value}: actual count {actual}, requested {config.events_per_action}"
            )
    for split, per_action in config.split_targets.items():
        requested = per_action * len(config.actions)
        if actual_split_counts[split] != requested:
            result.warnings.append(
                f"{split}: actual count {actual_split_counts[split]}, requested {requested}"
            )

    for name, expected_hash in manifest.get("generated_file_sha256", {}).items():
        path = benchmark_dir / name
        if not path.exists() or sha256_file(path) != expected_hash:
            result.errors.append(f"generated file hash mismatch: {name}")
    observed_source_hashes = source_file_hashes(dataset)
    if observed_source_hashes != manifest.get("source_file_sha256"):
        result.errors.append("FiQA source file hashes differ from build-time hashes")
    if archive_path and archive_path.exists() and expected_md5:
        if md5_file(archive_path) != expected_md5:
            result.errors.append("canonical FiQA archive MD5 mismatch")

    split_file_ids: set[str] = set()
    for split in config.split_targets:
        try:
            split_events = load_jsonl_models(benchmark_dir / f"{split}_events.jsonl", EvolutionEvent)
        except Exception as exc:
            result.errors.append(str(exc))
            continue
        for event in split_events:
            if event.benchmark_split.value != split:
                result.errors.append(f"{event.event_id}: stored in wrong split file")
            split_file_ids.add(event.event_id)
    if split_file_ids != set(event_by_id):
        result.errors.append("split event files do not exactly partition events.jsonl")

    result.statistics = {
        "event_count": len(events),
        "count_per_action": dict(sorted(actual_action_counts.items())),
        "count_per_split": dict(sorted(actual_split_counts.items())),
        "t0_count": len(t0_units),
        "t1_count": len(t1_units),
        "unique_source_document_count": len(set().union(*ancestry.values())) if ancestry else 0,
        "source_overlap": {f"{left}_{right}": len(ancestry[left] & ancestry[right]) for left, right in split_pairs},
        "query_overlap": {f"{left}_{right}": len(query_ancestry[left] & query_ancestry[right]) for left, right in split_pairs},
    }
    return result
