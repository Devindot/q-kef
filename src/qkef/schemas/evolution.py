"""Ground-truth contracts for the controlled temporal evolution benchmark."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qkef.schemas.knowledge import EvolutionAction


class BenchmarkSplit(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    CALIBRATION = "calibration"
    TEST = "test"


class EvolutionRelationType(str, Enum):
    NEW = "none/new"
    SUPERSEDES = "supersedes"
    COMPLEMENTARY = "complementary"
    RETRACTION = "retraction"
    COEXISTENCE = "coexistence"
    COMPOUND_SPLIT = "compound/split"


class EvolutionEvent(BaseModel):
    """One constructed input and its expected lifecycle action."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    expected_action: EvolutionAction
    benchmark_split: BenchmarkSplit
    source_qrels_split: str = Field(min_length=1)
    t0_knowledge_ids: list[str] = Field(default_factory=list)
    incoming_knowledge_ids: list[str] = Field(min_length=1)
    expected_target_ids: list[str] = Field(default_factory=list)
    expected_child_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(min_length=1)
    source_query_ids: list[str] = Field(default_factory=list)
    relation_type: EvolutionRelationType
    mutation_method: str = Field(min_length=1)
    mutation_parameters: dict[str, Any] = Field(default_factory=dict)
    generator_version: str = Field(min_length=1)
    seed: int
    t0_timestamp: datetime
    t1_timestamp: datetime
    requires_human_review: bool = True
    automatic_rationale: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_semantics(self) -> "EvolutionEvent":
        if self.t1_timestamp <= self.t0_timestamp:
            raise ValueError("t1_timestamp must be later than t0_timestamp")
        sequence_fields = (
            self.t0_knowledge_ids,
            self.incoming_knowledge_ids,
            self.expected_target_ids,
            self.expected_child_ids,
            self.source_document_ids,
            self.source_query_ids,
        )
        if any(len(values) != len(set(values)) for values in sequence_fields):
            raise ValueError("event identifier lists must not contain duplicates")
        if self.expected_action is EvolutionAction.NEW and self.expected_target_ids:
            raise ValueError("NEW events cannot have expected targets")
        if self.expected_action in {
            EvolutionAction.REPLACE,
            EvolutionAction.MERGE,
            EvolutionAction.ARCHIVE,
            EvolutionAction.COEXIST,
        } and not self.expected_target_ids:
            raise ValueError(f"{self.expected_action.value} events require a target")
        if self.expected_action is EvolutionAction.SPLIT and len(self.expected_child_ids) < 2:
            raise ValueError("SPLIT events require at least two expected child IDs")
        if self.expected_action in {EvolutionAction.MERGE, EvolutionAction.COEXIST} and len(
            self.source_document_ids
        ) < 2:
            raise ValueError(f"{self.expected_action.value} events require two source documents")
        return self
