"""Raw/model separation and metadata-driven sanitization tests."""

from datetime import UTC, datetime

from qkef.datasets.evolution import MERGE_DELIMITER, SPLIT_DELIMITER
from qkef.ingestion.pipeline import ingest_knowledge_unit
from qkef.schemas import EvolutionEvent, KnowledgeUnit, TemporalState


CONFIG = {
    "normalization": {
        "unicode_form": "NFKC",
        "normalize_line_endings": True,
        "remove_zero_width": True,
        "trim_whitespace": True,
        "normalize_horizontal_whitespace": True,
        "max_blank_lines": 2,
    },
    "sanitization": {
        "remove_merge_scaffold": True,
        "neutralize_split_boundary": True,
        "remove_archive_target_id_from_text": True,
        "max_unapproved_removal_fraction": 0.25,
    },
}


def make_unit(text: str, origin: str, action: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        knowledge_id=f"qkef-train-{action.lower()}-0001-t1",
        source_document_id="d01",
        text=text,
        source="synthetic test fixture",
        document_version="T1-v1",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ingestion_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"text_origin": origin, "expected_action": action},
        evolution_action=action,
    )


def make_event(unit: KnowledgeUnit, action: str, sources: list[str]) -> EvolutionEvent:
    targets = ["parent-t0"] if action in {"REPLACE", "MERGE", "ARCHIVE", "COEXIST"} else []
    children = ["child-a", "child-b"] if action == "SPLIT" else []
    relations = {
        "NEW": "none/new",
        "REPLACE": "supersedes",
        "MERGE": "complementary",
        "ARCHIVE": "retraction",
        "COEXIST": "coexistence",
        "SPLIT": "compound/split",
    }
    return EvolutionEvent(
        event_id=unit.knowledge_id.removesuffix("-t1"),
        expected_action=action,
        benchmark_split="train",
        source_qrels_split="train",
        t0_knowledge_ids=targets,
        incoming_knowledge_ids=[unit.knowledge_id],
        expected_target_ids=targets,
        expected_child_ids=children,
        source_document_ids=sources,
        source_query_ids=["q01", "q02"] if len(sources) > 1 else ["q01"],
        relation_type=relations[action],
        mutation_method="fixture_rule",
        mutation_parameters={},
        generator_version="test",
        seed=42,
        t0_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        t1_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        automatic_rationale="Synthetic test fixture.",
    )


def ingest(text: str, origin: str, action: str, sources: list[str] | None = None):
    unit = make_unit(text, origin, action)
    return ingest_knowledge_unit(
        unit, TemporalState.T1, [make_event(unit, action, sources or ["d01"])], CONFIG
    )


def test_raw_text_preserved_and_normalization_trace_is_exact() -> None:
    raw = "  Rate\r\n\u200bvalue   is 6%.\r\n\r\n\r\n\r\nStill valid.  "
    unit = ingest(raw, "deterministic_numeric_mutation", "REPLACE")

    assert unit.raw_text == raw
    assert unit.model_text == "Rate\nvalue is 6%.\n\n\nStill valid."
    assert unit.sanitization_operations == [
        "normalize_line_endings",
        "remove_zero_width_characters",
        "normalize_horizontal_whitespace",
        "reduce_repeated_blank_lines",
        "trim_leading_trailing_whitespace",
    ]
    assert "6%" in unit.model_text


def test_merge_scaffold_removed_but_both_sources_preserved() -> None:
    unit = ingest(
        "Base source content." + MERGE_DELIMITER + "Complementary source content.",
        "deterministic_merge_construction",
        "MERGE",
        ["d01", "d02"],
    )

    assert unit.raw_text.endswith("Complementary source content.")
    assert "Additional related information:" not in unit.model_text
    assert "Base source content." in unit.model_text
    assert "Complementary source content." in unit.model_text
    assert unit.source_document_ids == ["d01", "d02"]
    assert "remove_synthetic_merge_header" in unit.sanitization_operations


def test_split_marker_neutralized_and_components_preserved() -> None:
    unit = ingest(
        "First source section." + SPLIT_DELIMITER + "Second source section.",
        "deterministic_compound_split",
        "SPLIT",
        ["d01", "d02"],
    )

    assert "KNOWLEDGE SEGMENT BOUNDARY" not in unit.model_text
    assert unit.model_text == "First source section.\n\nSecond source section."
    assert unit.source_document_ids == ["d01", "d02"]


def test_archive_identifier_removed_but_semantics_and_target_preserved() -> None:
    raw = (
        "Administrative knowledge-base notice.\n\nTarget knowledge ID: parent-t0\n\n"
        "The referenced knowledge item has been withdrawn and should no longer be treated as active."
    )
    unit = ingest(raw, "synthetic_administrative_notice", "ARCHIVE")

    assert "Target knowledge ID" not in unit.model_text
    assert "withdrawn" in unit.model_text
    assert "no longer be treated as active" in unit.model_text
    assert unit.provenance["archive_target_knowledge_id"] == "parent-t0"


def test_original_fiqa_text_is_not_blindly_sanitized() -> None:
    raw = "An authentic phrase says Additional related information: keep this text."
    unit = ingest(raw, "original_fiqa", "NEW")

    assert unit.model_text == raw
    assert unit.sanitization_operations == []
