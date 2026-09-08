"""Deterministic dense, lexical, and reciprocal-rank-fused retrieval."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class CandidateResult:
    identifier: str
    dense_score: float | None
    lexical_score: float | None
    fused_score: float
    dense_rank: int | None
    lexical_rank: int | None


class HybridCandidateResolver:
    """Transparent local resolver preserving dense-only and lexical baselines."""

    def __init__(self, documents: dict[str, str], vectors: dict[str, tuple[float, ...]], *, rrf_k: int = 60):
        if set(documents) != set(vectors):
            raise ValueError("documents and vectors must have identical identifiers")
        self.documents = dict(documents)
        self.vectors = {key: np.asarray(value, dtype=np.float64) / np.linalg.norm(value) for key, value in vectors.items()}
        self.rrf_k = rrf_k
        self.doc_tokens = {key: Counter(_tokens(text)) for key, text in documents.items()}
        self.document_frequency = Counter(token for tokens in self.doc_tokens.values() for token in tokens)

    def _dense(self, query_vector: tuple[float, ...]) -> list[tuple[str, float]]:
        query = np.asarray(query_vector, dtype=np.float64)
        query /= np.linalg.norm(query)
        return sorted(((key, float(np.dot(query, vector))) for key, vector in self.vectors.items()), key=lambda item: (-item[1], item[0]))

    def _lexical(self, query_text: str) -> list[tuple[str, float]]:
        query = Counter(_tokens(query_text))
        count = max(1, len(self.documents))
        scored = []
        for identifier, document in self.doc_tokens.items():
            score = 0.0
            for token, frequency in query.items():
                if token in document:
                    inverse = np.log((count + 1) / (self.document_frequency[token] + 1)) + 1.0
                    score += min(frequency, document[token]) * float(inverse)
            scored.append((identifier, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))

    def resolve(self, query_text: str, query_vector: tuple[float, ...], top_k: int = 10, *, mode: str = "hybrid") -> list[CandidateResult]:
        if mode not in {"dense", "lexical", "hybrid"}:
            raise ValueError("mode must be dense, lexical, or hybrid")
        dense = self._dense(query_vector)
        lexical = self._lexical(query_text)
        dense_ranks = {key: rank for rank, (key, _) in enumerate(dense, 1)}
        lexical_ranks = {key: rank for rank, (key, _) in enumerate(lexical, 1)}
        dense_scores, lexical_scores = dict(dense), dict(lexical)
        results = []
        for identifier in sorted(self.documents):
            if mode == "dense":
                fused = 1.0 / (self.rrf_k + dense_ranks[identifier])
            elif mode == "lexical":
                fused = 1.0 / (self.rrf_k + lexical_ranks[identifier])
            else:
                fused = 1.0 / (self.rrf_k + dense_ranks[identifier]) + 1.0 / (self.rrf_k + lexical_ranks[identifier])
            results.append(CandidateResult(identifier, dense_scores[identifier], lexical_scores[identifier], fused, dense_ranks[identifier], lexical_ranks[identifier]))
        return sorted(results, key=lambda item: (-item.fused_score, item.identifier))[:top_k]


def retrieval_metrics(rankings: Iterable[list[str]], targets: Iterable[set[str]], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    pairs = list(zip(rankings, targets))
    output = {f"recall@{k}": sum(bool(set(ranking[:k]) & target) for ranking, target in pairs) / max(1, len(pairs)) for k in ks}
    output["mrr"] = sum(next((1.0 / rank for rank, identifier in enumerate(ranking, 1) if identifier in target), 0.0) for ranking, target in pairs) / max(1, len(pairs))
    output["eligible"] = float(len(pairs))
    return output
