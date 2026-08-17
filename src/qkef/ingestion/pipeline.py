"""Reusable schema-aware ingestion of Phase 1 benchmark units."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from qkef.datasets.evolution import load_jsonl_models
from qkef.ingestion.normalize import normalize_model_text
from qkef.ingestion.sanitize import sanitize_model_text
from qkef.schemas import (
    EvolutionEvent,
    IngestedKnowledgeUnit,
    KnowledgeUnit,
    TemporalState,
)
from qkef.schemas.ingestion import text_sha256


def ingest_knowledge_unit(
    unit: KnowledgeUnit,
    temporal_state: TemporalState,
    events: Sequence[EvolutionEvent],
    config: Mapping[str, Any],
) -> IngestedKnowledgeUnit:
    """Preserve raw text and derive a deterministic label-safe model text."""

    if not events:
        raise ValueError(f"{unit.knowledge_id}: no associated Phase 1 event")
    splits = {event.benchmark_split for event in events}
    if len(splits) != 1:
        raise ValueError(f"{unit.knowledge_id}: associated events cross benchmark splits")
    normalization_config = config["normalization"]
    sanitization_config = config["sanitization"]
    model_text, operations = normalize_model_text(unit.text, normalization_config)
    construction_origin = unit.metadata.get("text_origin")
    sanitized = sanitize_model_text(
        model_text,
        str(construction_origin) if construction_origin else None,
        sanitization_config,
    )
    model_text = sanitized.text
    operations.extend(sanitized.operations)
    if not model_text.strip():
        raise ValueError(f"{unit.knowledge_id}: sanitization produced empty model text")
    removed_fraction = max(0, len(unit.text) - len(model_text)) / len(unit.text)
    approved_removal = bool(operations)
    threshold = float(sanitization_config.get("max_unapproved_removal_fraction", 0.25))
    if removed_fraction > threshold and not approved_removal:
        raise ValueError(
            f"{unit.knowledge_id}: unexplained sanitization removed {removed_fraction:.1%} of text"
        )
    if temporal_state is TemporalState.T0:
        source_document_ids = [unit.source_document_id]
    else:
        source_document_ids = sorted(
            {source_id for event in events for source_id in event.source_document_ids}
        )
    provenance = {
        "phase1_event_ids": sorted(event.event_id for event in events),
        "phase1_source_document_ids": source_document_ids,
        "text_origin": construction_origin,
        "structured_relationships": {
            "t0_knowledge_ids": sorted(
                {item for event in events for item in event.t0_knowledge_ids}
            ),
            "expected_target_ids": sorted(
                {item for event in events for item in event.expected_target_ids}
            ),
        },
        **sanitized.extracted_metadata,
    }
    safe_metadata = {
        key: value
        for key, value in unit.metadata.items()
        if key not in {"expected_action", "mutation"}
    }
    return IngestedKnowledgeUnit(
        knowledge_id=unit.knowledge_id,
        benchmark_split=next(iter(splits)),
        temporal_state=temporal_state,
        raw_text=unit.text,
        model_text=model_text,
        source_document_id=unit.source_document_id,
        source_document_ids=source_document_ids,
        source=unit.source,
        document_version=unit.document_version,
        timestamp=unit.timestamp,
        lifecycle_status=unit.lifecycle_status,
        event_ids=sorted(event.event_id for event in events),
        provenance=provenance,
        sanitization_operations=operations,
        raw_sha256=text_sha256(unit.text),
        model_text_sha256=text_sha256(model_text),
        metadata=safe_metadata,
    )


def ingest_benchmark(
    benchmark_dir: Path, ingestion_config: Mapping[str, Any]
) -> tuple[list[IngestedKnowledgeUnit], list[IngestedKnowledgeUnit]]:
    """Load Phase 1 files, associate events, and ingest all T0/T1 units."""

    t0_units = load_jsonl_models(benchmark_dir / "t0_knowledge.jsonl", KnowledgeUnit)
    t1_units = load_jsonl_models(benchmark_dir / "t1_incoming.jsonl", KnowledgeUnit)
    events = load_jsonl_models(benchmark_dir / "events.jsonl", EvolutionEvent)
    t0_events: dict[str, list[EvolutionEvent]] = defaultdict(list)
    t1_events: dict[str, list[EvolutionEvent]] = defaultdict(list)
    for event in events:
        for knowledge_id in event.t0_knowledge_ids:
            t0_events[knowledge_id].append(event)
        for knowledge_id in event.incoming_knowledge_ids:
            t1_events[knowledge_id].append(event)
    ingested_t0 = [
        ingest_knowledge_unit(unit, TemporalState.T0, t0_events[unit.knowledge_id], ingestion_config)
        for unit in sorted(t0_units, key=lambda item: item.knowledge_id)
    ]
    ingested_t1 = [
        ingest_knowledge_unit(unit, TemporalState.T1, t1_events[unit.knowledge_id], ingestion_config)
        for unit in sorted(t1_units, key=lambda item: item.knowledge_id)
    ]
    return ingested_t0, ingested_t1
