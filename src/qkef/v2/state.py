"""Explicit graph/index/epoch state with bitemporal retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

from qkef.v2.canonical import sha256_payload
from qkef.v2.types import ACTIVE_LIFECYCLE_STATES, LifecycleState, RetrievalState


def _instant(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class AuthorityMetadata:
    source_authority: str | None = None
    approval_level: str | None = None
    department: str | None = None
    document_type: str | None = None
    jurisdiction: str | None = None
    source_priority: int | None = None


@dataclass(frozen=True)
class V2KnowledgeRecord:
    identifier: str
    text: str
    vector: tuple[float, ...]
    source_ids: tuple[str, ...]
    lifecycle_state: LifecycleState = LifecycleState.CURRENT
    retrieval_state: RetrievalState = RetrievalState.ACTIVE
    valid_from: str | None = None
    valid_to: str | None = None
    system_from: str | None = None
    system_to: str | None = None
    authority: AuthorityMetadata = field(default_factory=AuthorityMetadata)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.text.strip() or not self.vector:
            raise ValueError("record identifier, text, and vector are required")
        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("record vector must be finite")
        if np.linalg.norm(self.vector) == 0:
            raise ValueError("record vector must be non-zero")
        if self.valid_from and self.valid_to and _instant(self.valid_from) > _instant(self.valid_to):
            raise ValueError("valid_from must not be after valid_to")
        if self.system_from and self.system_to and _instant(self.system_from) > _instant(self.system_to):
            raise ValueError("system_from must not be after system_to")

    def normalized_vector(self) -> tuple[float, ...]:
        vector = np.asarray(self.vector, dtype=np.float64)
        return tuple((vector / np.linalg.norm(vector)).tolist())

    def visible_at(self, *, valid_time: str | None = None, system_time: str | None = None) -> bool:
        for instant, start, end in ((valid_time, self.valid_from, self.valid_to), (system_time, self.system_from, self.system_to)):
            if instant is None:
                continue
            point = _instant(instant)
            if start and point < _instant(start):
                return False
            if end and point >= _instant(end):
                return False
        return True


@dataclass(frozen=True, order=True)
class GraphEdge:
    source: str
    target: str
    relation: str


@dataclass
class KnowledgeState:
    records: dict[str, V2KnowledgeRecord] = field(default_factory=dict)
    edges: tuple[GraphEdge, ...] = ()
    index_entries: dict[str, tuple[float, ...]] = field(default_factory=dict)
    epoch_id: int = 0
    transition_id: str = "GENESIS"

    @classmethod
    def from_records(cls, records: Iterable[V2KnowledgeRecord], *, epoch_id: int = 0) -> "KnowledgeState":
        by_id = {record.identifier: record for record in records}
        index = {
            identifier: record.normalized_vector()
            for identifier, record in by_id.items()
            if record.retrieval_state is RetrievalState.ACTIVE
        }
        return cls(records=by_id, index_entries=index, epoch_id=epoch_id)

    @property
    def graph_state_hash(self) -> str:
        nodes = [
            {
                "authority": record.authority,
                "identifier": identifier,
                "lifecycle_state": record.lifecycle_state,
                "metadata": record.metadata,
                "source_ids": record.source_ids,
                "system_from": record.system_from,
                "system_to": record.system_to,
                "text": record.text,
                "valid_from": record.valid_from,
                "valid_to": record.valid_to,
            }
            for identifier, record in sorted(self.records.items())
        ]
        return sha256_payload({"edges": sorted(self.edges), "nodes": nodes})

    @property
    def index_state_hash(self) -> str:
        return sha256_payload({key: self.index_entries[key] for key in sorted(self.index_entries)})

    @property
    def state_hash(self) -> str:
        return sha256_payload({
            "epoch_id": self.epoch_id,
            "graph_state_hash": self.graph_state_hash,
            "index_state_hash": self.index_state_hash,
            "transition_id": self.transition_id,
        })

    def clone(self) -> "KnowledgeState":
        return KnowledgeState(dict(self.records), tuple(self.edges), dict(self.index_entries), self.epoch_id, self.transition_id)

    def search(
        self,
        query_vector: tuple[float, ...],
        top_k: int,
        *,
        historical: bool = False,
        as_of_valid_time: str | None = None,
        as_of_system_time: str | None = None,
    ) -> list[tuple[str, float]]:
        query = np.asarray(query_vector, dtype=np.float64)
        query /= np.linalg.norm(query)
        identifiers = sorted(self.records) if historical else sorted(self.index_entries)
        results: list[tuple[str, float]] = []
        for identifier in identifiers:
            record = self.records[identifier]
            if historical and record.retrieval_state in {RetrievalState.QUARANTINED, RetrievalState.BLOCKED}:
                continue
            if not record.visible_at(valid_time=as_of_valid_time, system_time=as_of_system_time):
                continue
            if historical:
                vector = np.asarray(record.normalized_vector())
            else:
                if record.retrieval_state is not RetrievalState.ACTIVE:
                    continue
                vector = np.asarray(self.index_entries[identifier])
            results.append((identifier, float(np.dot(query, vector))))
        return sorted(results, key=lambda item: (-item[1], item[0]))[:top_k]

    def with_record_state(self, identifier: str, lifecycle: LifecycleState, retrieval: RetrievalState) -> V2KnowledgeRecord:
        return replace(self.records[identifier], lifecycle_state=lifecycle, retrieval_state=retrieval)

    def active_graph_ids(self) -> set[str]:
        return {identifier for identifier, record in self.records.items() if record.retrieval_state is RetrievalState.ACTIVE and record.lifecycle_state in ACTIVE_LIFECYCLE_STATES}
