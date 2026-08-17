"""Typed Phase 2 ingestion and chunk-corpus records."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qkef.schemas.evolution import BenchmarkSplit
from qkef.schemas.knowledge import LifecycleStatus


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TemporalState(str, Enum):
    T0 = "T0"
    T1 = "T1"


class ChunkStrategy(str, Enum):
    IDENTITY = "identity"
    FIXED_WINDOW = "fixed_window"
    TFIDF_BOUNDARY = "tfidf_boundary"


class IngestedKnowledgeUnit(BaseModel):
    """A provenance-preserving raw unit and its model-facing derivative."""

    model_config = ConfigDict(extra="forbid")

    knowledge_id: str = Field(min_length=1)
    benchmark_split: BenchmarkSplit
    temporal_state: TemporalState
    raw_text: str = Field(min_length=1)
    model_text: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_document_ids: list[str] = Field(min_length=1)
    source: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    timestamp: datetime
    lifecycle_status: LifecycleStatus
    event_ids: list[str] = Field(min_length=1)
    provenance: dict[str, Any] = Field(default_factory=dict)
    sanitization_operations: list[str] = Field(default_factory=list)
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_hashes_and_ids(self) -> "IngestedKnowledgeUnit":
        if self.raw_sha256 != text_sha256(self.raw_text):
            raise ValueError("raw_sha256 does not match raw_text")
        if self.model_text_sha256 != text_sha256(self.model_text):
            raise ValueError("model_text_sha256 does not match model_text")
        if len(self.source_document_ids) != len(set(self.source_document_ids)):
            raise ValueError("source_document_ids must be unique")
        if len(self.event_ids) != len(set(self.event_ids)):
            raise ValueError("event_ids must be unique")
        return self


class KnowledgeChunk(BaseModel):
    """One label-free model-facing segment with internal provenance."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(pattern=r"^chk_[0-9a-f]{24}$")
    knowledge_id: str = Field(min_length=1)
    event_ids: list[str] = Field(min_length=1)
    benchmark_split: BenchmarkSplit
    temporal_state: TemporalState
    strategy: ChunkStrategy
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    word_count: int = Field(ge=1)
    char_count: int = Field(ge=1)
    sentence_count: int | None = Field(default=None, ge=1)
    model_char_start: int | None = Field(default=None, ge=0)
    model_char_end: int | None = Field(default=None, ge=1)
    source_document_ids: list[str] = Field(min_length=1)
    timestamp: datetime
    document_version: str = Field(min_length=1)
    lifecycle_status: LifecycleStatus
    overlap_metadata: dict[str, Any] = Field(default_factory=dict)
    chunking_metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_content(self) -> "KnowledgeChunk":
        if self.char_count != len(self.text):
            raise ValueError("char_count does not match text")
        if self.chunk_sha256 != text_sha256(self.text):
            raise ValueError("chunk_sha256 does not match text")
        if (self.model_char_start is None) != (self.model_char_end is None):
            raise ValueError("model character offsets must both be set or both be null")
        if self.model_char_start is not None and self.model_char_end <= self.model_char_start:
            raise ValueError("model_char_end must be greater than model_char_start")
        if len(self.source_document_ids) != len(set(self.source_document_ids)):
            raise ValueError("source_document_ids must be unique")
        return self


class UnitChunkMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_id: str = Field(min_length=1)
    temporal_state: TemporalState
    benchmark_split: BenchmarkSplit
    strategy: ChunkStrategy
    chunk_ids: list[str] = Field(min_length=1)
    chunk_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_count(self) -> "UnitChunkMap":
        if self.chunk_count != len(self.chunk_ids):
            raise ValueError("chunk_count does not match chunk_ids")
        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("chunk_ids must be unique")
        return self


FORBIDDEN_LABEL_FIELDS = {
    "expected_action",
    "ground_truth_action",
    "class_label",
    "target_label",
    "expected_relation",
}


def chunk_schema_label_fields() -> set[str]:
    return set(KnowledgeChunk.model_fields) & FORBIDDEN_LABEL_FIELDS


def chunk_id_is_opaque(chunk_id: str) -> bool:
    return bool(re.fullmatch(r"chk_[0-9a-f]{24}", chunk_id))
