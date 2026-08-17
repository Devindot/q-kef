"""Dependency-free loader for the standard BEIR FiQA file layout."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class FiqaDataError(ValueError):
    """Raised when FiQA source files are malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class FiqaDocument:
    document_id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class FiqaQuery:
    query_id: str
    text: str


@dataclass(frozen=True, slots=True)
class FiqaQrel:
    query_id: str
    document_id: str
    relevance: int
    source_split: str


@dataclass(frozen=True)
class FiqaDataset:
    root: Path
    documents: dict[str, FiqaDocument]
    queries: dict[str, FiqaQuery]
    qrels: tuple[FiqaQrel, ...]

    @property
    def qrels_by_split(self) -> dict[str, tuple[FiqaQrel, ...]]:
        grouped: dict[str, list[FiqaQrel]] = {}
        for qrel in self.qrels:
            grouped.setdefault(qrel.source_split, []).append(qrel)
        return {key: tuple(value) for key, value in sorted(grouped.items())}

    def documents_for_query(self, query_id: str, split: str | None = None) -> tuple[FiqaDocument, ...]:
        ids = {
            qrel.document_id
            for qrel in self.qrels
            if qrel.query_id == query_id and (split is None or qrel.source_split == split)
        }
        return tuple(self.documents[item] for item in sorted(ids))


def _require_string(record: dict[str, Any], key: str, location: str, *, blank_ok: bool = False) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise FiqaDataError(f"{location}: required field {key!r} must be a string")
    if not blank_ok and not value.strip():
        raise FiqaDataError(f"{location}: required field {key!r} must not be blank")
    return value


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise FiqaDataError(f"cannot open {path}: {exc}") from exc
    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise FiqaDataError(f"{path}:{line_number}: malformed JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise FiqaDataError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def _load_documents(path: Path) -> dict[str, FiqaDocument]:
    documents: dict[str, FiqaDocument] = {}
    for line_number, record in _read_jsonl(path):
        location = f"{path}:{line_number}"
        document_id = _require_string(record, "_id", location)
        if document_id in documents:
            raise FiqaDataError(f"{location}: duplicate document ID {document_id!r}")
        documents[document_id] = FiqaDocument(
            document_id=document_id,
            title=_require_string(record, "title", location, blank_ok=True),
            text=_require_string(record, "text", location, blank_ok=True),
        )
    if not documents:
        raise FiqaDataError(f"{path}: corpus is empty")
    return documents


def _load_queries(path: Path) -> dict[str, FiqaQuery]:
    queries: dict[str, FiqaQuery] = {}
    for line_number, record in _read_jsonl(path):
        location = f"{path}:{line_number}"
        query_id = _require_string(record, "_id", location)
        if query_id in queries:
            raise FiqaDataError(f"{location}: duplicate query ID {query_id!r}")
        queries[query_id] = FiqaQuery(
            query_id=query_id,
            text=_require_string(record, "text", location),
        )
    if not queries:
        raise FiqaDataError(f"{path}: query collection is empty")
    return queries


def _pick_column(fieldnames: list[str], options: tuple[str, ...], path: Path) -> str:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for option in options:
        if option in normalized:
            return normalized[option]
    raise FiqaDataError(f"{path}: missing one of required TSV columns {options}")


def _load_qrels(path: Path, split: str) -> list[FiqaQrel]:
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise FiqaDataError(f"cannot open {path}: {exc}") from exc
    qrels: list[FiqaQrel] = []
    with stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise FiqaDataError(f"{path}: qrels file has no header")
        query_column = _pick_column(reader.fieldnames, ("query-id", "query_id", "qid"), path)
        document_column = _pick_column(reader.fieldnames, ("corpus-id", "doc-id", "document_id", "docid"), path)
        score_column = _pick_column(reader.fieldnames, ("score", "relevance"), path)
        for line_number, row in enumerate(reader, start=2):
            query_id = (row.get(query_column) or "").strip()
            document_id = (row.get(document_column) or "").strip()
            if not query_id or not document_id:
                raise FiqaDataError(f"{path}:{line_number}: blank qrel identifier")
            try:
                relevance = int(row.get(score_column, ""))
            except (TypeError, ValueError) as exc:
                raise FiqaDataError(f"{path}:{line_number}: invalid relevance score") from exc
            qrels.append(FiqaQrel(query_id, document_id, relevance, split))
    return qrels


def validate_fiqa_layout(root: Path) -> tuple[Path, Path, Path]:
    """Validate and return corpus, queries, and qrels-directory paths."""

    root = Path(root)
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_dir = root / "qrels"
    missing = [path for path in (corpus_path, queries_path, qrels_dir) if not path.exists()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FiqaDataError(f"FiQA layout under {root} is incomplete; missing: {rendered}")
    if not any(qrels_dir.glob("*.tsv")):
        raise FiqaDataError(f"{qrels_dir}: no qrels TSV files found")
    return corpus_path, queries_path, qrels_dir


def load_fiqa_dataset(root: str | Path) -> FiqaDataset:
    """Load and cross-validate a BEIR-compatible FiQA dataset directory."""

    resolved_root = Path(root).resolve()
    corpus_path, queries_path, qrels_dir = validate_fiqa_layout(resolved_root)
    documents = _load_documents(corpus_path)
    queries = _load_queries(queries_path)
    qrels: list[FiqaQrel] = []
    for qrels_path in sorted(qrels_dir.glob("*.tsv"), key=lambda item: item.name):
        qrels.extend(_load_qrels(qrels_path, qrels_path.stem.lower()))
    if not qrels:
        raise FiqaDataError(f"{qrels_dir}: qrels collection is empty")
    for qrel in qrels:
        if qrel.query_id not in queries:
            raise FiqaDataError(
                f"qrels/{qrel.source_split}: unknown query ID {qrel.query_id!r}"
            )
        if qrel.document_id not in documents:
            raise FiqaDataError(
                f"qrels/{qrel.source_split}: unknown document ID {qrel.document_id!r}"
            )
    return FiqaDataset(resolved_root, documents, queries, tuple(qrels))
