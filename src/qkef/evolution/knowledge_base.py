"""Deterministic local evolution-aware vector KB and execution semantics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from qkef.graph.semantic_graph import SemanticGraph
from qkef.retrieval.vector_index import VectorIndex


@dataclass
class KBRecord:
    identifier: str
    text: str
    vector: np.ndarray
    source_ids: list[str]
    status: str = "active"
    provenance: list[str] = field(default_factory=list)


class EvolutionKnowledgeBase:
    def __init__(self, dimension: int, *, append_only: bool = False):
        self.dimension = dimension
        self.append_only = append_only
        self.records: dict[str, KBRecord] = {}
        self.index = VectorIndex(dimension)
        self.graph = SemanticGraph()
        self.warnings: list[str] = []

    def add(self, record: KBRecord) -> None:
        self.records[record.identifier] = record
        self.index.add(record.identifier, record.vector, status=record.status)
        self.graph.add_node(record.identifier, status=record.status, source_ids="|".join(record.source_ids))

    def _status(self, identifier: str, status: str) -> None:
        if identifier not in self.records:
            self.warnings.append(f"missing target: {identifier}")
            return
        self.records[identifier].status = status
        self.index.add(identifier, self.records[identifier].vector, status=status)
        self.graph.graph.nodes[identifier]["status"] = status

    def execute(self, action: str, incoming: KBRecord, targets: list[str], *, split_texts: list[str] | None = None) -> list[str]:
        if self.append_only:
            self.add(incoming)
            return [incoming.identifier]
        target = targets[0] if targets else None
        if action == "NEW":
            self.add(incoming)
            return [incoming.identifier]
        if action == "REPLACE":
            if not target:
                self.warnings.append("REPLACE without target")
                self.add(incoming)
                return [incoming.identifier]
            self._status(target, "superseded")
            self.add(incoming)
            self.graph.add_edge(incoming.identifier, target, "supersedes")
            return [incoming.identifier]
        if action == "MERGE":
            if not target or target not in self.records:
                self.warnings.append("MERGE without valid target")
                self.add(incoming)
                return [incoming.identifier]
            paragraphs = []
            for text in (self.records[target].text, incoming.text):
                for paragraph in text.split("\n\n"):
                    if paragraph.strip() and paragraph.strip() not in paragraphs:
                        paragraphs.append(paragraph.strip())
            identifier = "kbm_" + hashlib.sha256(f"{target}|{incoming.identifier}".encode()).hexdigest()[:20]
            vector = self.records[target].vector + incoming.vector
            vector = vector / np.linalg.norm(vector)
            merged = KBRecord(identifier, "\n\n".join(paragraphs), vector, sorted(set(self.records[target].source_ids + incoming.source_ids)), provenance=[target, incoming.identifier])
            self._status(target, "superseded")
            self.add(merged)
            self.graph.add_node(incoming.identifier, status="lineage", source_ids="|".join(incoming.source_ids))
            self.graph.add_edge(identifier, target, "merged_from")
            self.graph.add_edge(identifier, incoming.identifier, "merged_from")
            return [identifier]
        if action == "ARCHIVE":
            if target:
                self._status(target, "archived")
                self.graph.add_node(incoming.identifier, status="audit_notice", source_ids="|".join(incoming.source_ids))
                self.graph.add_edge(target, incoming.identifier, "archived_by")
            else:
                self.warnings.append("ARCHIVE without target")
            return []
        if action == "COEXIST":
            self.add(incoming)
            if target and target in self.records:
                self.graph.add_edge(incoming.identifier, target, "coexists_with")
                self.graph.add_edge(target, incoming.identifier, "coexists_with")
            return [incoming.identifier]
        if action == "SPLIT":
            segments = [text for text in (split_texts or []) if text.strip()]
            if len(segments) < 2:
                self.warnings.append("split_execution_warning")
                self.add(incoming)
                return [incoming.identifier]
            self.graph.add_node(incoming.identifier, status="container", source_ids="|".join(incoming.source_ids))
            children = []
            for index, text in enumerate(segments):
                identifier = "kbs_" + hashlib.sha256(f"{incoming.identifier}|{index}".encode()).hexdigest()[:20]
                child = KBRecord(identifier, text, incoming.vector, incoming.source_ids, provenance=[incoming.identifier])
                self.add(child)
                self.graph.add_edge(incoming.identifier, identifier, "split_into")
                children.append(identifier)
            return children
        self.warnings.append(f"unknown action: {action}")
        self.add(incoming)
        return [incoming.identifier]

    def search(self, vector: np.ndarray, top_k: int, *, include_inactive: bool = False) -> list[tuple[str, float]]:
        return self.index.search(vector, top_k) if include_inactive else self.index.search(vector, top_k, status="active")

    def statistics(self) -> dict[str, int]:
        statuses = {name: sum(record.status == name for record in self.records.values()) for name in ("active", "archived", "superseded")}
        return {"nodes": self.graph.graph.number_of_nodes(), "edges": self.graph.graph.number_of_edges(), **statuses, "warnings": len(self.warnings)}
