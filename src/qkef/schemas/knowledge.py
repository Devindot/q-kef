"""Core, implementation-neutral knowledge schemas for Q-KEF."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LifecycleStatus(str, Enum):
    """Persistence state of a knowledge unit in an evolution-aware store."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"


class EvolutionAction(str, Enum):
    """Representable outcomes for a future knowledge-evolution decision."""

    NEW = "NEW"
    REPLACE = "REPLACE"
    MERGE = "MERGE"
    ARCHIVE = "ARCHIVE"
    COEXIST = "COEXIST"
    SPLIT = "SPLIT"


class KnowledgeUnit(BaseModel):
    """A traceable semantic unit and its lifecycle metadata.

    ``evolution_action`` records a known or labelled action; this schema does not
    infer one. Relationship fields contain knowledge IDs rather than embedded
    objects so units can be serialized without recursive structures.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    knowledge_id: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    timestamp: datetime
    ingestion_timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_id: str | None = None
    graph_node_id: str | None = None
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE
    evolution_action: EvolutionAction | None = None
    supersedes: list[str] | None = None
    superseded_by: list[str] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator(
        "knowledge_id", "source_document_id", "text", "source", "document_version"
    )
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        """Reject whitespace-only identifiers and required text."""

        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("embedding_id", "graph_node_id")
    @classmethod
    def reject_blank_optional_ids(cls, value: str | None) -> str | None:
        """Treat present relationship identifiers as meaningful identifiers."""

        if value is not None and not value.strip():
            raise ValueError("identifier must not be blank")
        return value

    @field_validator("supersedes", "superseded_by")
    @classmethod
    def validate_relationship_ids(cls, value: list[str] | None) -> list[str] | None:
        """Prevent ambiguous empty IDs and duplicate graph relationships."""

        if value is None:
            return None
        if any(not item.strip() for item in value):
            raise ValueError("relationship identifiers must not be blank")
        if len(value) != len(set(value)):
            raise ValueError("relationship identifiers must be unique")
        return value
