"""Deterministic dense, lexical, and reciprocal-rank-fused retrieval."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class CandidateQuery:
    """One leakage-labelled query used for retrieval configuration selection."""

    query_id: str
    query_text: str
    query_vector: tuple[float, ...]
    target_ids: frozenset[str]
    split: str = "dev"


@dataclass(frozen=True)
class CandidateConfiguration:
    mode: str
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"dense", "lexical", "hybrid"}:
            raise ValueError("candidate mode must be dense, lexical, or hybrid")

    @property
    def identifier(self) -> str:
        if self.mode != "hybrid":
            return self.mode
        return f"hybrid_d{self.dense_weight:g}_l{self.lexical_weight:g}_k{self.rrf_k}"


@dataclass(frozen=True)
class DevelopmentSelection:
    selected: CandidateConfiguration
    selected_metrics: dict[str, float]
    all_results: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "selection_split": "dev",
            "selected": asdict(self.selected),
            "selected_identifier": self.selected.identifier,
            "selected_metrics": self.selected_metrics,
            "all_results": list(self.all_results),
        }


class HybridCandidateResolver:
    """Transparent local resolver preserving dense-only and lexical baselines."""

    def __init__(
        self,
        documents: dict[str, str],
        vectors: dict[str, tuple[float, ...]],
        *,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ):
        if set(documents) != set(vectors):
            raise ValueError("documents and vectors must have identical identifiers")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if dense_weight < 0 or lexical_weight < 0 or dense_weight + lexical_weight <= 0:
            raise ValueError("retrieval weights must be non-negative with a positive sum")
        self.documents = dict(documents)
        self.vectors = {}
        for key, value in vectors.items():
            vector = np.asarray(value, dtype=np.float64)
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or norm == 0:
                raise ValueError(f"vector for {key!r} must be finite and non-zero")
            self.vectors[key] = vector / norm
        self.rrf_k = rrf_k
        total_weight = dense_weight + lexical_weight
        self.dense_weight = dense_weight / total_weight
        self.lexical_weight = lexical_weight / total_weight
        self.doc_tokens = {key: Counter(_tokens(text)) for key, text in documents.items()}
        self.document_frequency = Counter(token for tokens in self.doc_tokens.values() for token in tokens)

    def _dense(self, query_vector: tuple[float, ...]) -> list[tuple[str, float]]:
        query = np.asarray(query_vector, dtype=np.float64)
        norm = np.linalg.norm(query)
        if not np.isfinite(query).all() or norm == 0:
            raise ValueError("query vector must be finite and non-zero")
        query /= norm
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
                fused = self.dense_weight / (self.rrf_k + dense_ranks[identifier]) + self.lexical_weight / (self.rrf_k + lexical_ranks[identifier])
            results.append(CandidateResult(identifier, dense_scores[identifier], lexical_scores[identifier], fused, dense_ranks[identifier], lexical_ranks[identifier]))
        return sorted(results, key=lambda item: (-item.fused_score, item.identifier))[:top_k]


def retrieval_metrics(rankings: Iterable[list[str]], targets: Iterable[set[str]], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    pairs = list(zip(rankings, targets))
    output = {f"recall@{k}": sum(bool(set(ranking[:k]) & target) for ranking, target in pairs) / max(1, len(pairs)) for k in ks}
    output["mrr"] = sum(next((1.0 / rank for rank, identifier in enumerate(ranking, 1) if identifier in target), 0.0) for ranking, target in pairs) / max(1, len(pairs))
    output["eligible"] = float(len(pairs))
    return output


def select_dev_configuration(
    documents: dict[str, str],
    vectors: dict[str, tuple[float, ...]],
    queries: Iterable[CandidateQuery],
    *,
    top_k: int = 10,
    hybrid_weights: tuple[tuple[float, float], ...] = ((0.75, 0.25), (0.5, 0.5), (0.25, 0.75)),
    rrf_ks: tuple[int, ...] = (20, 60),
) -> DevelopmentSelection:
    """Select a retrieval configuration using development observations only.

    The guard is intentionally strict: callers cannot accidentally pass TEST or
    calibration observations. Ties prefer Recall@5, then MRR, Recall@1, and the
    simpler dense baseline before a hybrid or lexical configuration.
    """

    observations = tuple(queries)
    if not observations:
        raise ValueError("at least one development query is required")
    invalid_splits = sorted({query.split for query in observations if query.split != "dev"})
    if invalid_splits:
        raise ValueError(f"configuration selection accepts DEV only, received: {invalid_splits}")
    if any(not query.target_ids for query in observations):
        raise ValueError("every development query must have at least one target")

    configurations = [
        CandidateConfiguration("dense", dense_weight=1.0, lexical_weight=0.0),
        CandidateConfiguration("lexical", dense_weight=0.0, lexical_weight=1.0),
        *(
            CandidateConfiguration("hybrid", rrf_k, dense_weight, lexical_weight)
            for rrf_k in rrf_ks
            for dense_weight, lexical_weight in hybrid_weights
        ),
    ]
    results: list[dict[str, object]] = []
    for configuration in configurations:
        resolver = HybridCandidateResolver(
            documents,
            vectors,
            rrf_k=configuration.rrf_k,
            dense_weight=configuration.dense_weight,
            lexical_weight=configuration.lexical_weight,
        )
        rankings = [
            [
                item.identifier
                for item in resolver.resolve(
                    query.query_text,
                    query.query_vector,
                    top_k,
                    mode=configuration.mode,
                )
            ]
            for query in observations
        ]
        scores = retrieval_metrics(rankings, [set(query.target_ids) for query in observations])
        results.append({"configuration": asdict(configuration), "identifier": configuration.identifier, "metrics": scores})

    simplicity = {"dense": 2, "hybrid": 1, "lexical": 0}
    winner = min(
        results,
        key=lambda result: (
            -result["metrics"]["recall@5"],
            -result["metrics"]["mrr"],
            -result["metrics"]["recall@1"],
            -simplicity[result["configuration"]["mode"]],
            result["identifier"],
        ),
    )
    selected = CandidateConfiguration(**winner["configuration"])
    return DevelopmentSelection(selected, dict(winner["metrics"]), tuple(results))
