"""Phase 2 corpus construction, deterministic serialization, and validation."""

from __future__ import annotations

import csv
import io
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from qkef.chunking import ChunkingConfig, chunk_unit, count_words
from qkef.chunking.base import word_spans
from qkef.datasets.evolution import canonical_json, load_jsonl_models, sha256_file
from qkef.ingestion.pipeline import ingest_benchmark
from qkef.ingestion.normalize import normalize_model_text
from qkef.ingestion.sanitize import ARCHIVE_TARGET_PATTERN, known_synthetic_markers
from qkef.schemas import (
    ChunkStrategy,
    EvolutionAction,
    EvolutionEvent,
    IngestedKnowledgeUnit,
    KnowledgeChunk,
    TemporalState,
    UnitChunkMap,
)
from qkef.schemas.ingestion import FORBIDDEN_LABEL_FIELDS, chunk_schema_label_fields


OUTPUT_NAMES = {
    "ingested_t0": "ingested_t0.jsonl",
    "ingested_t1": "ingested_t1.jsonl",
    "identity": "chunks_identity.jsonl",
    "fixed_window": "chunks_fixed_window.jsonl",
    "tfidf_boundary": "chunks_tfidf_boundary.jsonl",
    "maps": "unit_chunk_map.jsonl",
    "audit": "sanitization_audit.csv",
}


@dataclass
class Phase2Content:
    ingested_t0: list[IngestedKnowledgeUnit]
    ingested_t1: list[IngestedKnowledgeUnit]
    chunks: dict[ChunkStrategy, list[KnowledgeChunk]]
    mappings: list[UnitChunkMap]
    sanitization_audit_csv: str


@dataclass
class ChunkValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors


def _jsonl(values: Sequence[Any]) -> str:
    return "".join(canonical_json(value) + "\n" for value in values)


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _audit_csv(units: Sequence[IngestedKnowledgeUnit]) -> str:
    stream = io.StringIO(newline="")
    fields = [
        "knowledge_id",
        "temporal_state",
        "benchmark_split",
        "raw_char_count",
        "model_char_count",
        "text_changed",
        "sanitization_operations",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for unit in sorted(units, key=lambda item: item.knowledge_id):
        writer.writerow(
            {
                "knowledge_id": unit.knowledge_id,
                "temporal_state": unit.temporal_state.value,
                "benchmark_split": unit.benchmark_split.value,
                "raw_char_count": len(unit.raw_text),
                "model_char_count": len(unit.model_text),
                "text_changed": str(unit.raw_text != unit.model_text).lower(),
                "sanitization_operations": "|".join(unit.sanitization_operations),
            }
        )
    return stream.getvalue()


def generate_phase2_content(
    phase1_dir: Path, config: Mapping[str, Any]
) -> Phase2Content:
    ingested_t0, ingested_t1 = ingest_benchmark(phase1_dir, config["ingestion"])
    expected_t0 = int(config["ingestion"].get("expected_t0_units", 200))
    expected_t1 = int(config["ingestion"].get("expected_t1_units", 300))
    if len(ingested_t0) != expected_t0 or len(ingested_t1) != expected_t1:
        raise ValueError(
            f"unexpected Phase 1 counts: T0={len(ingested_t0)}, T1={len(ingested_t1)}; "
            f"expected {expected_t0}/{expected_t1}"
        )
    chunk_config = ChunkingConfig.from_mapping(config["chunking"])
    all_units = sorted(ingested_t0 + ingested_t1, key=lambda item: item.knowledge_id)
    chunks: dict[ChunkStrategy, list[KnowledgeChunk]] = {}
    mappings: list[UnitChunkMap] = []
    for strategy in chunk_config.enabled_strategies:
        strategy_chunks: list[KnowledgeChunk] = []
        for unit in all_units:
            unit_chunks = chunk_unit(unit, strategy, chunk_config)
            strategy_chunks.extend(unit_chunks)
            mappings.append(
                UnitChunkMap(
                    knowledge_id=unit.knowledge_id,
                    temporal_state=unit.temporal_state,
                    benchmark_split=unit.benchmark_split,
                    strategy=strategy,
                    chunk_ids=[chunk.chunk_id for chunk in unit_chunks],
                    chunk_count=len(unit_chunks),
                )
            )
        chunks[strategy] = sorted(strategy_chunks, key=lambda item: item.chunk_id)
    mappings.sort(key=lambda item: (item.strategy.value, item.knowledge_id))
    return Phase2Content(
        ingested_t0,
        ingested_t1,
        chunks,
        mappings,
        _audit_csv(all_units),
    )


def render_phase2_files(content: Phase2Content) -> dict[str, str]:
    return {
        OUTPUT_NAMES["ingested_t0"]: _jsonl(content.ingested_t0),
        OUTPUT_NAMES["ingested_t1"]: _jsonl(content.ingested_t1),
        OUTPUT_NAMES["identity"]: _jsonl(content.chunks.get(ChunkStrategy.IDENTITY, [])),
        OUTPUT_NAMES["fixed_window"]: _jsonl(content.chunks.get(ChunkStrategy.FIXED_WINDOW, [])),
        OUTPUT_NAMES["tfidf_boundary"]: _jsonl(content.chunks.get(ChunkStrategy.TFIDF_BOUNDARY, [])),
        OUTPUT_NAMES["maps"]: _jsonl(content.mappings),
        OUTPUT_NAMES["audit"]: content.sanitization_audit_csv,
    }


def phase2_content_signature(content: Phase2Content) -> str:
    files = render_phase2_files(content)
    payload = "".join(f"{name}\0{files[name]}\0" for name in sorted(files))
    return _sha256_text(payload)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def strategy_statistics(chunks: Sequence[KnowledgeChunk]) -> dict[str, Any]:
    words = [chunk.word_count for chunk in chunks]
    per_unit = Counter(chunk.knowledge_id for chunk in chunks)
    return {
        "chunk_count": len(chunks),
        "unit_count": len(per_unit),
        "mean_words": round(statistics.fmean(words), 3) if words else 0,
        "median_words": round(statistics.median(words), 3) if words else 0,
        "p90_words": round(_percentile(words, 0.90), 3),
        "p95_words": round(_percentile(words, 0.95), 3),
        "min_words": min(words) if words else 0,
        "max_words": max(words) if words else 0,
        "multi_chunk_unit_count": sum(count > 1 for count in per_unit.values()),
        "count_by_split": dict(
            sorted(Counter(chunk.benchmark_split.value for chunk in chunks).items())
        ),
    }


def _tfidf_diagnostics(chunks: Sequence[KnowledgeChunk]) -> dict[str, int]:
    return {
        "lexical_boundaries": sum(
            chunk.chunking_metadata.get("boundary_reason") == "lexical_similarity"
            for chunk in chunks
        ),
        "paragraph_boundaries": sum(
            chunk.chunking_metadata.get("boundary_reason") == "paragraph_structure"
            for chunk in chunks
        ),
        "forced_boundaries": sum(
            chunk.chunking_metadata.get("boundary_reason") == "forced_size"
            for chunk in chunks
        )
        + sum(
            int(chunk.chunking_metadata.get("oversized_sentence_forced_boundaries", 0))
            for chunk in chunks
            if chunk.chunk_index == 0
        ),
        "fallback_units": len(
            {
                chunk.knowledge_id
                for chunk in chunks
                if chunk.chunking_metadata.get("fallback_reason")
            }
        ),
        "short_tail_merges": len(
            {
                chunk.knowledge_id
                for chunk in chunks
                if chunk.chunking_metadata.get("short_tail_merged")
            }
        ),
    }


def build_phase2_corpus(
    phase1_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    verify_determinism: bool = True,
) -> tuple[dict[str, Any], Phase2Content]:
    content = generate_phase2_content(phase1_dir, config)
    deterministic = None
    if verify_determinism:
        rebuilt = generate_phase2_content(phase1_dir, config)
        deterministic = phase2_content_signature(content) == phase2_content_signature(rebuilt)
        if not deterministic:
            raise RuntimeError("Phase 2 deterministic rebuild content hash mismatch")
    rendered = render_phase2_files(content)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        _atomic_write(output_dir / name, text)
    phase1_manifest = json.loads(
        (phase1_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    all_units = content.ingested_t0 + content.ingested_t1
    operation_counts = Counter(
        operation for unit in all_units for operation in unit.sanitization_operations
    )
    marker_counts = {
        marker: sum(marker in unit.model_text for unit in all_units)
        for marker in known_synthetic_markers()
    }
    chunk_stats = {
        strategy.value: strategy_statistics(chunks)
        for strategy, chunks in content.chunks.items()
    }
    manifest = {
        "phase_name": "Phase 2 — Semantic Ingestion and Deterministic Chunking",
        "phase_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_phase1_benchmark_sha256": phase1_manifest["deterministic_content_sha256"],
        "source_fiqa_md5": phase1_manifest["fiqa_observed_md5"],
        "seed": int(config["project"]["seed"]),
        "ingestion_config": config["ingestion"],
        "chunking_config": config["chunking"],
        "normalization_version": config["ingestion"]["normalization_version"],
        "sanitizer_version": config["ingestion"]["sanitizer_version"],
        "chunker_versions": {
            strategy.value: "1.0" for strategy in content.chunks
        },
        "ingested_counts": {
            "T0": len(content.ingested_t0),
            "T1": len(content.ingested_t1),
            "total": len(all_units),
            "model_text_changed": sum(unit.raw_text != unit.model_text for unit in all_units),
        },
        "sanitization_operation_counts": dict(sorted(operation_counts.items())),
        "synthetic_marker_occurrences_in_model_text": marker_counts,
        "archive_target_identifier_occurrences_in_model_text": sum(
            bool(ARCHIVE_TARGET_PATTERN.search(unit.model_text)) for unit in all_units
        ),
        "chunk_statistics": chunk_stats,
        "tfidf_diagnostics": _tfidf_diagnostics(
            content.chunks.get(ChunkStrategy.TFIDF_BOUNDARY, [])
        ),
        "deterministic_rebuild_status": deterministic,
        "deterministic_content_sha256": phase2_content_signature(content),
        "generated_file_sha256": {
            name: _sha256_text(text) for name, text in sorted(rendered.items())
        },
        "label_leakage_audit_status": "pending_validation",
        "source_ancestry_integrity_status": "pending_validation",
        "warnings": [],
    }
    _atomic_write(
        output_dir / "phase2_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest, content


def _load_outputs(output_dir: Path) -> tuple[
    list[IngestedKnowledgeUnit],
    list[IngestedKnowledgeUnit],
    dict[ChunkStrategy, list[KnowledgeChunk]],
    list[UnitChunkMap],
    dict[str, Any],
]:
    t0 = load_jsonl_models(output_dir / OUTPUT_NAMES["ingested_t0"], IngestedKnowledgeUnit)
    t1 = load_jsonl_models(output_dir / OUTPUT_NAMES["ingested_t1"], IngestedKnowledgeUnit)
    chunks = {
        ChunkStrategy.IDENTITY: load_jsonl_models(
            output_dir / OUTPUT_NAMES["identity"], KnowledgeChunk
        ),
        ChunkStrategy.FIXED_WINDOW: load_jsonl_models(
            output_dir / OUTPUT_NAMES["fixed_window"], KnowledgeChunk
        ),
        ChunkStrategy.TFIDF_BOUNDARY: load_jsonl_models(
            output_dir / OUTPUT_NAMES["tfidf_boundary"], KnowledgeChunk
        ),
    }
    mappings = load_jsonl_models(output_dir / OUTPUT_NAMES["maps"], UnitChunkMap)
    manifest = json.loads((output_dir / "phase2_manifest.json").read_text(encoding="utf-8"))
    return t0, t1, chunks, mappings, manifest


def validate_phase2_corpus(
    phase1_dir: Path, output_dir: Path, config: Mapping[str, Any]
) -> ChunkValidationResult:
    result = ChunkValidationResult()
    required = [*render_phase2_files(Phase2Content([], [], {}, [], "")).keys(), "phase2_manifest.json"]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        result.errors.append(f"missing Phase 2 files: {', '.join(missing)}")
        return result
    try:
        t0, t1, chunks_by_strategy, mappings, manifest = _load_outputs(output_dir)
        events = load_jsonl_models(phase1_dir / "events.jsonl", EvolutionEvent)
    except Exception as exc:
        result.errors.append(str(exc))
        return result
    units = t0 + t1
    unit_by_id: dict[str, IngestedKnowledgeUnit] = {}
    for unit in units:
        if unit.knowledge_id in unit_by_id:
            result.errors.append(f"duplicate ingested knowledge ID: {unit.knowledge_id}")
        unit_by_id[unit.knowledge_id] = unit
    expected_t0 = int(config["ingestion"].get("expected_t0_units", 200))
    expected_t1 = int(config["ingestion"].get("expected_t1_units", 300))
    if len(t0) != expected_t0 or len(t1) != expected_t1:
        result.errors.append(f"unexpected ingested counts: T0={len(t0)}, T1={len(t1)}")

    all_chunk_ids: set[str] = set()
    unresolved_parents = 0
    ancestry: dict[str, set[str]] = defaultdict(set)
    coverage_failures = 0
    for strategy, chunks in chunks_by_strategy.items():
        by_parent: dict[str, list[KnowledgeChunk]] = defaultdict(list)
        for chunk in chunks:
            if chunk.chunk_id in all_chunk_ids:
                result.errors.append(f"duplicate chunk ID: {chunk.chunk_id}")
            all_chunk_ids.add(chunk.chunk_id)
            parent = unit_by_id.get(chunk.knowledge_id)
            if parent is None:
                unresolved_parents += 1
                result.errors.append(f"{chunk.chunk_id}: unresolved parent {chunk.knowledge_id}")
                continue
            by_parent[parent.knowledge_id].append(chunk)
            ancestry[chunk.benchmark_split.value].update(chunk.source_document_ids)
            if chunk.strategy is not strategy:
                result.errors.append(f"{chunk.chunk_id}: strategy/file mismatch")
            if chunk.benchmark_split != parent.benchmark_split:
                result.errors.append(f"{chunk.chunk_id}: split differs from parent")
            if chunk.temporal_state != parent.temporal_state:
                result.errors.append(f"{chunk.chunk_id}: temporal state differs from parent")
            if chunk.source_document_ids != parent.source_document_ids:
                result.errors.append(f"{chunk.chunk_id}: source provenance differs from parent")
            if chunk.event_ids != parent.event_ids:
                result.errors.append(f"{chunk.chunk_id}: event provenance differs from parent")
            if chunk.word_count != count_words(chunk.text):
                result.errors.append(f"{chunk.chunk_id}: word count mismatch")
            if chunk.model_char_start is not None:
                if parent.model_text[chunk.model_char_start : chunk.model_char_end] != chunk.text:
                    result.errors.append(f"{chunk.chunk_id}: character span mismatch")
            for marker in known_synthetic_markers():
                if marker in chunk.text:
                    result.errors.append(f"{chunk.chunk_id}: synthetic marker leakage: {marker}")
        if set(by_parent) != set(unit_by_id):
            result.errors.append(f"{strategy.value}: parent coverage is incomplete")
        for knowledge_id, parent_chunks in by_parent.items():
            parent = unit_by_id[knowledge_id]
            ordered = sorted(parent_chunks, key=lambda item: item.chunk_index)
            if [chunk.chunk_index for chunk in ordered] != list(range(len(ordered))):
                result.errors.append(f"{strategy.value}/{knowledge_id}: non-contiguous indexes")
            if strategy is ChunkStrategy.IDENTITY:
                covered = len(ordered) == 1 and ordered[0].text == parent.model_text
            elif strategy is ChunkStrategy.FIXED_WINDOW:
                parent_words = word_spans(parent.model_text)
                represented: set[int] = set()
                order_ok = True
                previous_start = -1
                for chunk in ordered:
                    start = int(chunk.overlap_metadata.get("word_start", -1))
                    end = int(chunk.overlap_metadata.get("word_end", -1))
                    order_ok &= start > previous_start and 0 <= start < end <= len(parent_words)
                    previous_start = start
                    represented.update(range(start, end))
                covered = order_ok and represented == set(range(len(parent_words)))
            else:
                covered = bool(ordered) and ordered[0].model_char_start == 0
                for left, right in zip(ordered, ordered[1:]):
                    covered &= left.model_char_end == right.model_char_start
                covered &= ordered[-1].model_char_end == len(parent.model_text)
                covered &= "".join(chunk.text for chunk in ordered) == parent.model_text
            if not covered:
                coverage_failures += 1
                result.errors.append(f"{strategy.value}/{knowledge_id}: content coverage failure")

    mapping_keys: set[tuple[str, str]] = set()
    chunk_lookup = {
        chunk.chunk_id: chunk for chunks in chunks_by_strategy.values() for chunk in chunks
    }
    for mapping in mappings:
        key = (mapping.strategy.value, mapping.knowledge_id)
        if key in mapping_keys:
            result.errors.append(f"duplicate unit/chunk map: {key}")
        mapping_keys.add(key)
        if mapping.knowledge_id not in unit_by_id:
            result.errors.append(f"mapping has unresolved parent: {mapping.knowledge_id}")
        if any(chunk_id not in chunk_lookup for chunk_id in mapping.chunk_ids):
            result.errors.append(f"mapping has unresolved chunk: {mapping.knowledge_id}")
    expected_mapping_keys = {
        (strategy.value, knowledge_id)
        for strategy in chunks_by_strategy
        for knowledge_id in unit_by_id
    }
    if mapping_keys != expected_mapping_keys:
        result.errors.append("unit/chunk mapping does not cover every unit/strategy")

    marker_occurrences = {
        marker: sum(marker in unit.model_text for unit in units)
        for marker in known_synthetic_markers()
    }
    for marker, count in marker_occurrences.items():
        if count:
            result.errors.append(f"model-text synthetic marker leakage {marker!r}: {count}")
    archive_id_occurrences = sum(bool(ARCHIVE_TARGET_PATTERN.search(unit.model_text)) for unit in units)
    if archive_id_occurrences:
        result.errors.append(f"archive target identifiers remain in model text: {archive_id_occurrences}")
    label_fields = chunk_schema_label_fields()
    if label_fields:
        result.errors.append(f"label fields present in chunk schema: {sorted(label_fields)}")
    for chunks in chunks_by_strategy.values():
        for chunk in chunks:
            if any(label in chunk.chunk_id for label in ("new", "replace", "merge", "archive", "coexist", "split", "train", "dev", "test")):
                result.errors.append(f"action/split encoded in chunk ID: {chunk.chunk_id}")
            dumped_fields = set(chunk.model_dump())
            leaked = dumped_fields & FORBIDDEN_LABEL_FIELDS
            if leaked:
                result.errors.append(f"{chunk.chunk_id}: label fields leaked: {sorted(leaked)}")

    event_by_id = {event.event_id: event for event in events}
    for unit in t1:
        event = event_by_id[unit.event_ids[0]]
        if event.expected_action is EvolutionAction.MERGE:
            parts = unit.raw_text.split("\n\nAdditional related information:\n", maxsplit=1)
            normalized_parts = [
                normalize_model_text(part, config["ingestion"]["normalization"])[0]
                for part in parts
            ]
            if len(parts) != 2 or any(part not in unit.model_text for part in normalized_parts):
                result.errors.append(f"{unit.knowledge_id}: MERGE source contribution lost")
        elif event.expected_action is EvolutionAction.SPLIT:
            parts = unit.raw_text.split("\n\n--- KNOWLEDGE SEGMENT BOUNDARY ---\n\n", maxsplit=1)
            normalized_parts = [
                normalize_model_text(part, config["ingestion"]["normalization"])[0]
                for part in parts
            ]
            if len(parts) != 2 or any(part not in unit.model_text for part in normalized_parts):
                result.errors.append(f"{unit.knowledge_id}: SPLIT source contribution lost")
            if len(unit.source_document_ids) < 2:
                result.errors.append(f"{unit.knowledge_id}: SPLIT multi-source provenance lost")
        elif event.expected_action is EvolutionAction.REPLACE:
            new_value = str(event.mutation_parameters["new_value"])
            if new_value not in unit.model_text:
                result.errors.append(f"{unit.knowledge_id}: REPLACE value was lost")
        elif event.expected_action is EvolutionAction.ARCHIVE:
            lowered = unit.model_text.lower()
            if "withdrawn" not in lowered or "no longer be treated as active" not in lowered:
                result.errors.append(f"{unit.knowledge_id}: ARCHIVE semantics were lost")
            if "archive_target_knowledge_id" not in unit.provenance:
                result.errors.append(f"{unit.knowledge_id}: ARCHIVE structured target lost")

    overlaps = {
        "train_dev": len(ancestry["train"] & ancestry["dev"]),
        "train_test": len(ancestry["train"] & ancestry["test"]),
        "dev_test": len(ancestry["dev"] & ancestry["test"]),
    }
    if any(overlaps.values()):
        result.errors.append(f"source ancestry overlap detected: {overlaps}")
    for name, expected_hash in manifest.get("generated_file_sha256", {}).items():
        path = output_dir / name
        if not path.exists() or sha256_file(path) != expected_hash:
            result.errors.append(f"output file hash mismatch: {name}")

    result.statistics = {
        "ingested_t0": len(t0),
        "ingested_t1": len(t1),
        "model_text_changed": sum(unit.raw_text != unit.model_text for unit in units),
        "sanitization_operation_counts": dict(
            sorted(Counter(op for unit in units for op in unit.sanitization_operations).items())
        ),
        "marker_occurrences": marker_occurrences,
        "archive_target_id_occurrences": archive_id_occurrences,
        "label_schema_fields": sorted(label_fields),
        "action_encoded_chunk_ids": sum(
            any(label in chunk.chunk_id for label in ("new", "replace", "merge", "archive", "coexist", "split"))
            for chunks in chunks_by_strategy.values()
            for chunk in chunks
        ),
        "duplicate_chunk_ids": sum(len(chunks) for chunks in chunks_by_strategy.values()) - len(all_chunk_ids),
        "unresolved_parents": unresolved_parents,
        "coverage_failures": coverage_failures,
        "model_content_coverage_percent": 100.0 if not coverage_failures else 0.0,
        "source_overlap": overlaps,
        "chunk_statistics": {
            strategy.value: strategy_statistics(chunks)
            for strategy, chunks in chunks_by_strategy.items()
        },
        "tfidf_diagnostics": _tfidf_diagnostics(
            chunks_by_strategy[ChunkStrategy.TFIDF_BOUNDARY]
        ),
    }
    return result


def render_phase2_report(manifest: Mapping[str, Any], validation: ChunkValidationResult) -> str:
    counts = manifest["ingested_counts"]
    lines = [
        "# Phase 2 Ingestion and Chunking Report",
        "",
        "This report describes deterministic preprocessing infrastructure, not model or retrieval performance.",
        "",
        "## Dataset",
        "",
        f"- T0 units: {counts['T0']}",
        f"- T1 units: {counts['T1']}",
        f"- Total units: {counts['total']}",
        "",
        "## Sanitization",
        "",
        f"- Units with raw/model differences: {counts['model_text_changed']}",
    ]
    lines.extend(
        f"- `{operation}`: {count}"
        for operation, count in manifest["sanitization_operation_counts"].items()
    )
    lines.extend(["", "## Chunking", ""])
    for strategy, stats in manifest["chunk_statistics"].items():
        lines.extend(
            [
                f"### {strategy}",
                "",
                f"- Chunks / units: {stats['chunk_count']} / {stats['unit_count']}",
                f"- Mean / median words: {stats['mean_words']} / {stats['median_words']}",
                f"- P90 / P95 words: {stats['p90_words']} / {stats['p95_words']}",
                f"- Minimum / maximum words: {stats['min_words']} / {stats['max_words']}",
                f"- Multi-chunk units: {stats['multi_chunk_unit_count']}",
                "",
            ]
        )
    diagnostics = manifest["tfidf_diagnostics"]
    lines.extend(
        [
            "## TF-IDF diagnostics",
            "",
            f"- Lexical boundaries: {diagnostics['lexical_boundaries']}",
            f"- Paragraph boundaries: {diagnostics['paragraph_boundaries']}",
            f"- Forced-size boundaries: {diagnostics['forced_boundaries']}",
            f"- Fallback units: {diagnostics['fallback_units']}",
            f"- Short-tail merges: {diagnostics['short_tail_merges']}",
            "",
            "## Integrity",
            "",
            f"- Validation: {'PASS' if validation.passed else 'FAIL'}",
            f"- Model-content coverage: {validation.statistics['model_content_coverage_percent']}%",
            f"- Duplicate chunk IDs: {validation.statistics['duplicate_chunk_ids']}",
            f"- Unresolved parents: {validation.statistics['unresolved_parents']}",
            f"- Synthetic marker occurrences: {sum(validation.statistics['marker_occurrences'].values())}",
            f"- Train/dev ancestry overlap: {validation.statistics['source_overlap']['train_dev']}",
            f"- Train/test ancestry overlap: {validation.statistics['source_overlap']['train_test']}",
            f"- Dev/test ancestry overlap: {validation.statistics['source_overlap']['dev_test']}",
            "",
            "## Reproducibility",
            "",
            f"- Deterministic rebuild: {'PASS' if manifest['deterministic_rebuild_status'] else 'FAIL'}",
            f"- Canonical content SHA-256: `{manifest['deterministic_content_sha256']}`",
            "- Output SHA-256 hashes are stored in `phase2_manifest.json`.",
            "",
            "## Limitations",
            "",
            "- TF-IDF captures lexical similarity, not deep semantic equivalence.",
            "- Regex sentence splitting is deterministic but not linguistically complete.",
            "- Default thresholds are engineering defaults, not optimized values.",
            "- FiQA passages may be shorter and less structured than enterprise manuals.",
            "- Phase 2 does not establish that one chunking strategy is superior.",
            "- Final tuning must use TRAIN/DEV only; TEST must remain untouched.",
            "",
        ]
    )
    return "\n".join(lines)
