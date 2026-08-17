"""Tests for the Phase 0 knowledge data contract."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from qkef.schemas import EvolutionAction, KnowledgeUnit, LifecycleStatus


def make_unit(**overrides: object) -> KnowledgeUnit:
    values: dict[str, object] = {
        "knowledge_id": "ku-2026-fee",
        "source_document_id": "fiqa-doc-42",
        "text": "Effective 2026, the annual management fee is 1.2%.",
        "source": "controlled-fiqa-update",
        "document_version": "T1",
        "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        "ingestion_timestamp": datetime(2026, 1, 2, tzinfo=UTC),
        "metadata": {"seed": 42, "mutation_type": "replace"},
    }
    values.update(overrides)
    return KnowledgeUnit.model_validate(values)


def test_valid_knowledge_unit_creation() -> None:
    unit = make_unit(
        lifecycle_status=LifecycleStatus.ACTIVE,
        evolution_action=EvolutionAction.REPLACE,
        confidence=0.9,
    )

    assert unit.knowledge_id == "ku-2026-fee"
    assert unit.lifecycle_status is LifecycleStatus.ACTIVE
    assert unit.evolution_action is EvolutionAction.REPLACE
    assert unit.metadata["seed"] == 42


@pytest.mark.parametrize("status", list(LifecycleStatus))
def test_all_lifecycle_statuses_validate(status: LifecycleStatus) -> None:
    assert make_unit(lifecycle_status=status).lifecycle_status is status


@pytest.mark.parametrize("action", list(EvolutionAction))
def test_all_evolution_actions_validate(action: EvolutionAction) -> None:
    assert make_unit(evolution_action=action).evolution_action is action


def test_optional_relationship_fields() -> None:
    unit = make_unit(
        embedding_id="emb-42",
        graph_node_id="node-42",
        supersedes=["ku-old-fee"],
        superseded_by=["ku-future-fee"],
    )

    assert unit.embedding_id == "emb-42"
    assert unit.graph_node_id == "node-42"
    assert unit.supersedes == ["ku-old-fee"]
    assert unit.superseded_by == ["ku-future-fee"]


def test_relationship_fields_default_to_none() -> None:
    unit = make_unit()

    assert unit.embedding_id is None
    assert unit.graph_node_id is None
    assert unit.supersedes is None
    assert unit.superseded_by is None


def test_invalid_lifecycle_status_fails_validation() -> None:
    with pytest.raises(ValidationError):
        make_unit(lifecycle_status="deleted")


def test_invalid_evolution_action_fails_validation() -> None:
    with pytest.raises(ValidationError):
        make_unit(evolution_action="UPDATE")


def test_invalid_confidence_fails_validation() -> None:
    with pytest.raises(ValidationError):
        make_unit(confidence=1.01)
