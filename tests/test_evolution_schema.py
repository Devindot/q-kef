"""Ground-truth evolution-event contract tests."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from qkef.schemas import EvolutionEvent


def valid_event(**overrides: object) -> EvolutionEvent:
    values: dict[str, object] = {
        "event_id": "qkef-train-replace-0001",
        "expected_action": "REPLACE",
        "benchmark_split": "train",
        "source_qrels_split": "train",
        "t0_knowledge_ids": ["unit-t0"],
        "incoming_knowledge_ids": ["unit-t1"],
        "expected_target_ids": ["unit-t0"],
        "expected_child_ids": [],
        "source_document_ids": ["d01"],
        "source_query_ids": ["q01"],
        "relation_type": "supersedes",
        "mutation_method": "single_numeric_token_mutation_v1",
        "mutation_parameters": {"old_value": "5%", "new_value": "6%"},
        "generator_version": "1.0",
        "seed": 42,
        "t0_timestamp": datetime(2025, 1, 1, tzinfo=UTC),
        "t1_timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        "requires_human_review": True,
        "automatic_rationale": "Controlled test mutation.",
        "metadata": {},
    }
    values.update(overrides)
    return EvolutionEvent.model_validate(values)


def test_valid_evolution_event() -> None:
    assert valid_event().expected_action.value == "REPLACE"


def test_invalid_action_is_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_event(expected_action="UPDATE")


def test_bad_timestamp_order_is_rejected() -> None:
    with pytest.raises(ValidationError, match="later than"):
        valid_event(t1_timestamp=datetime(2024, 1, 1, tzinfo=UTC))


def test_missing_required_provenance_is_rejected() -> None:
    with pytest.raises(ValidationError):
        valid_event(source_document_ids=[])


def test_new_event_rejects_target() -> None:
    with pytest.raises(ValidationError, match="NEW events cannot"):
        valid_event(expected_action="NEW", relation_type="none/new")
