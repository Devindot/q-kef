"""Backwards-compatible v2 lifecycle and retrieval semantics."""

from __future__ import annotations

from enum import Enum


class LifecycleState(str, Enum):
    CURRENT = "CURRENT"
    SUPERSEDED = "SUPERSEDED"
    ARCHIVED = "ARCHIVED"
    COEXISTING = "COEXISTING"
    CONSOLIDATED = "CONSOLIDATED"
    DERIVED = "DERIVED"
    PENDING_REVIEW = "PENDING_REVIEW"


class RetrievalState(str, Enum):
    ACTIVE = "ACTIVE"
    HISTORICAL_ONLY = "HISTORICAL_ONLY"
    QUARANTINED = "QUARANTINED"
    BLOCKED = "BLOCKED"


class Decision(str, Enum):
    AUTO_COMMIT = "AUTO_COMMIT"
    QUARANTINE = "QUARANTINE"


ACTIVE_LIFECYCLE_STATES = {
    LifecycleState.CURRENT,
    LifecycleState.COEXISTING,
    LifecycleState.CONSOLIDATED,
    LifecycleState.DERIVED,
}
