"""Shared configuration, tokenization, IDs, and chunk dispatch."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from qkef.schemas import ChunkStrategy, IngestedKnowledgeUnit, KnowledgeChunk
from qkef.schemas.ingestion import text_sha256


WORD_PATTERN = re.compile(r"\S+")
SENTENCE_END_PATTERN = re.compile(r"[.!?]+(?=[ \t\r\n]+|$)[ \t]*")
PARAGRAPH_GAP_PATTERN = re.compile(r"\n[ \t]*\n+")


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ChunkingConfig:
    enabled_strategies: tuple[ChunkStrategy, ...]
    fixed_max_words: int
    fixed_overlap_words: int
    tfidf_min_words: int
    tfidf_target_words: int
    tfidf_max_words: int
    tfidf_boundary_similarity_threshold: float

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ChunkingConfig":
        try:
            strategies = tuple(ChunkStrategy(item) for item in mapping["enabled_strategies"])
        except ValueError as exc:
            raise ValueError(f"unknown chunking strategy: {exc}") from exc
        if not strategies or len(strategies) != len(set(strategies)):
            raise ValueError("enabled_strategies must be non-empty and unique")
        fixed = mapping["fixed_window"]
        tfidf = mapping["tfidf_boundary"]
        config = cls(
            enabled_strategies=strategies,
            fixed_max_words=int(fixed["max_words"]),
            fixed_overlap_words=int(fixed["overlap_words"]),
            tfidf_min_words=int(tfidf["min_words"]),
            tfidf_target_words=int(tfidf["target_words"]),
            tfidf_max_words=int(tfidf["max_words"]),
            tfidf_boundary_similarity_threshold=float(tfidf["boundary_similarity_threshold"]),
        )
        if config.fixed_max_words <= 0:
            raise ValueError("fixed_window.max_words must be positive")
        if not 0 <= config.fixed_overlap_words < config.fixed_max_words:
            raise ValueError("fixed_window.overlap_words must satisfy 0 <= overlap < max_words")
        if not 0 < config.tfidf_min_words <= config.tfidf_target_words <= config.tfidf_max_words:
            raise ValueError("tfidf word limits must satisfy 0 < min <= target <= max")
        if not 0 <= config.tfidf_boundary_similarity_threshold <= 1:
            raise ValueError("TF-IDF boundary similarity threshold must be between 0 and 1")
        return config


def word_spans(text: str) -> list[TextSpan]:
    return [TextSpan(match.start(), match.end()) for match in WORD_PATTERN.finditer(text)]


def count_words(text: str) -> int:
    """Count whitespace-delimited lexical units; this is not an LLM token count."""

    return len(WORD_PATTERN.findall(text))


def sentence_spans(text: str) -> list[TextSpan]:
    """Partition text at conservative punctuation and paragraph boundaries."""

    if not text:
        return []
    boundaries = {0, len(text)}
    boundaries.update(match.end() for match in SENTENCE_END_PATTERN.finditer(text))
    boundaries.update(match.end() for match in PARAGRAPH_GAP_PATTERN.finditer(text))
    ordered = sorted(boundaries)
    spans: list[TextSpan] = []
    for start, end in zip(ordered, ordered[1:]):
        if start == end:
            continue
        if text[start:end].strip():
            spans.append(TextSpan(start, end))
        elif spans:
            spans[-1] = TextSpan(spans[-1].start, end)
    if spans and spans[0].start != 0:
        spans[0] = TextSpan(0, spans[0].end)
    if spans and spans[-1].end != len(text):
        spans[-1] = TextSpan(spans[-1].start, len(text))
    return spans or [TextSpan(0, len(text))]


def split_sentences(text: str) -> list[str]:
    return [text[span.start : span.end].strip() for span in sentence_spans(text)]


def opaque_chunk_id(strategy: ChunkStrategy, knowledge_id: str, chunk_index: int) -> str:
    payload = f"{strategy.value}|{knowledge_id}|{chunk_index}".encode("utf-8")
    return "chk_" + hashlib.sha256(payload).hexdigest()[:24]


def create_chunk(
    unit: IngestedKnowledgeUnit,
    strategy: ChunkStrategy,
    chunk_index: int,
    text: str,
    *,
    sentence_count: int | None,
    model_char_start: int | None,
    model_char_end: int | None,
    overlap_metadata: dict[str, Any] | None = None,
    chunking_metadata: dict[str, Any] | None = None,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=opaque_chunk_id(strategy, unit.knowledge_id, chunk_index),
        knowledge_id=unit.knowledge_id,
        event_ids=unit.event_ids,
        benchmark_split=unit.benchmark_split,
        temporal_state=unit.temporal_state,
        strategy=strategy,
        chunk_index=chunk_index,
        text=text,
        word_count=count_words(text),
        char_count=len(text),
        sentence_count=sentence_count,
        model_char_start=model_char_start,
        model_char_end=model_char_end,
        source_document_ids=unit.source_document_ids,
        timestamp=unit.timestamp,
        document_version=unit.document_version,
        lifecycle_status=unit.lifecycle_status,
        overlap_metadata=overlap_metadata or {},
        chunking_metadata=chunking_metadata or {},
        chunk_sha256=text_sha256(text),
    )


def chunk_unit(
    unit: IngestedKnowledgeUnit, strategy: ChunkStrategy, config: ChunkingConfig
) -> list[KnowledgeChunk]:
    """Dispatch using only model text and non-label unit metadata."""

    if strategy is ChunkStrategy.IDENTITY:
        from qkef.chunking.identity import chunk_identity

        return chunk_identity(unit)
    if strategy is ChunkStrategy.FIXED_WINDOW:
        from qkef.chunking.fixed_window import chunk_fixed_window

        return chunk_fixed_window(unit, config.fixed_max_words, config.fixed_overlap_words)
    if strategy is ChunkStrategy.TFIDF_BOUNDARY:
        from qkef.chunking.tfidf_boundary import chunk_tfidf_boundary

        return chunk_tfidf_boundary(
            unit,
            min_words=config.tfidf_min_words,
            target_words=config.tfidf_target_words,
            max_words=config.tfidf_max_words,
            threshold=config.tfidf_boundary_similarity_threshold,
        )
    raise ValueError(f"unsupported strategy: {strategy}")
