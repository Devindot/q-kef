"""Small extractive evidence composer; this is deliberately not an LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from qkef.runtime.schemas import QAResult


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


@dataclass(frozen=True)
class EvidenceRecord:
    identifier: str
    text: str
    source_ids: list[str]
    status: str = "active"
    vector: np.ndarray | None = None


class EvidenceQA:
    def __init__(self, records: Iterable[EvidenceRecord], encoder: Any | None = None, *, system: str = "qkef"):
        self.records = tuple(record for record in records if record.status.lower() == "active")
        self.encoder = encoder
        self.system = system

    def answer(self, question: str, *, top_k: int = 3, minimum_score: float = 0.08) -> QAResult:
        if not question.strip():
            raise ValueError("question cannot be empty")
        query_tokens = _tokens(question)
        query_vector = self.encoder.encode([question])[0] if self.encoder is not None else None
        ranked = []
        for record in self.records:
            words = _tokens(record.text)
            lexical = len(query_tokens & words) / max(1, len(query_tokens | words))
            semantic = float(np.dot(query_vector, record.vector)) if query_vector is not None and record.vector is not None else 0.0
            score = 0.8 * semantic + 0.2 * lexical if query_vector is not None else lexical
            ranked.append((score, record.identifier, record))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = ranked[:top_k]
        if not selected or selected[0][0] < minimum_score:
            return QAResult(answer="Insufficient evidence in the current knowledge base.", supporting_passages=[], provenance=[], retrieval_scores=[], system=self.system, warnings=["Academic research demo — not financial advice."])
        evidence = []
        for _, _, record in selected:
            best = max(_sentences(record.text), key=lambda sentence: (len(query_tokens & _tokens(sentence)), -len(sentence)))
            evidence.append(best)
        answer = " ".join(evidence[:2])
        return QAResult(
            answer=answer, supporting_passages=[item[2].text[:600] for item in selected],
            provenance=[item[2].source_ids for item in selected], retrieval_scores=[float(item[0]) for item in selected],
            system=self.system, warnings=["Academic research demo — not financial advice."],
        )
