"""Adapters for naturally versioned local CSV/JSONL data."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TemporalDocument:
    source_document_id: str
    version_id: str
    content: str
    valid_from: str | None = None
    valid_to: str | None = None
    system_time: str | None = None
    source_authority: str | None = None
    predecessor_ids: tuple[str, ...] = ()
    lifecycle_annotation: str | None = None


class TemporalDatasetAdapter(ABC):
    @abstractmethod
    def load(self, path: Path) -> list[TemporalDocument]:
        raise NotImplementedError

    @staticmethod
    def _document(row: dict[str, Any]) -> TemporalDocument:
        predecessors = row.get("predecessor_ids") or []
        if isinstance(predecessors, str):
            predecessors = [value.strip() for value in predecessors.split("|") if value.strip()]
        return TemporalDocument(
            source_document_id=str(row["source_document_id"]), version_id=str(row["version_id"]), content=str(row["content"]),
            valid_from=row.get("valid_from") or None, valid_to=row.get("valid_to") or None, system_time=row.get("system_time") or None,
            source_authority=row.get("source_authority") or None, predecessor_ids=tuple(predecessors), lifecycle_annotation=row.get("lifecycle_annotation") or None,
        )


class JSONLTemporalAdapter(TemporalDatasetAdapter):
    def load(self, path: Path) -> list[TemporalDocument]:
        return [self._document(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class CSVTemporalAdapter(TemporalDatasetAdapter):
    def load(self, path: Path) -> list[TemporalDocument]:
        with path.open(encoding="utf-8", newline="") as stream:
            return [self._document(row) for row in csv.DictReader(stream)]
