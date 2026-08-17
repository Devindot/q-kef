"""Tested standard ranking metrics."""

from __future__ import annotations

import math
from typing import Sequence, Set


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k if k > 0 else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str]) -> float:
    return next((1.0 / rank for rank, item in enumerate(ranked, 1) if item in relevant), 0.0)


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, item in enumerate(ranked[:k], 1) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def obsolete_rate_at_k(ranked: Sequence[str], obsolete: Set[str], k: int) -> float:
    actual = ranked[:k]
    return sum(item in obsolete for item in actual) / len(actual) if actual else 0.0
