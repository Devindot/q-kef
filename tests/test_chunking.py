"""Deterministic identity, window, sentence, and lexical-boundary tests."""

from datetime import UTC, datetime

import pytest

from qkef.chunking import ChunkingConfig, chunk_unit, split_sentences
from qkef.schemas import ChunkStrategy, IngestedKnowledgeUnit
from qkef.schemas.ingestion import text_sha256


def config(**overrides: object) -> ChunkingConfig:
    values = {
        "enabled_strategies": ["identity", "fixed_window", "tfidf_boundary"],
        "fixed_window": {"max_words": 10, "overlap_words": 2},
        "tfidf_boundary": {
            "min_words": 3,
            "target_words": 6,
            "max_words": 10,
            "boundary_similarity_threshold": 0.15,
        },
    }
    values.update(overrides)
    return ChunkingConfig.from_mapping(values)


def unit(text: str) -> IngestedKnowledgeUnit:
    return IngestedKnowledgeUnit(
        knowledge_id="qkef-train-replace-0001-t1",
        benchmark_split="train",
        temporal_state="T1",
        raw_text=text,
        model_text=text,
        source_document_id="d01",
        source_document_ids=["d01", "d02"],
        source="fixture",
        document_version="T1-v1",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        lifecycle_status="active",
        event_ids=["qkef-train-replace-0001"],
        provenance={},
        raw_sha256=text_sha256(text),
        model_text_sha256=text_sha256(text),
    )


def test_sentence_splitter_handles_decimals_and_questions() -> None:
    sentences = split_sentences(
        "The value is $10.50. Another sentence follows. Is this allowed? Yes, it is."
    )

    assert len(sentences) == 4
    assert sentences[0] == "The value is $10.50."


def test_identity_is_exact_and_opaque() -> None:
    parent = unit("The rate is 6%. It remains applicable.")
    chunks = chunk_unit(parent, ChunkStrategy.IDENTITY, config())

    assert len(chunks) == 1
    assert chunks[0].text == parent.model_text
    assert chunks[0].model_char_start == 0
    assert chunks[0].model_char_end == len(parent.model_text)
    assert chunks[0].chunk_id.startswith("chk_")
    assert "replace" not in chunks[0].chunk_id
    assert chunks[0].source_document_ids == ["d01", "d02"]


def test_fixed_window_max_overlap_order_and_tail() -> None:
    words = [f"w{index}" for index in range(25)]
    chunks = chunk_unit(unit(" ".join(words)), ChunkStrategy.FIXED_WINDOW, config())

    assert [chunk.word_count for chunk in chunks] == [10, 10, 9]
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert chunks[1].text.split()[-2:] == chunks[2].text.split()[:2]
    assert all(chunk.word_count <= 10 for chunk in chunks)


@pytest.mark.parametrize("count", [1, 10])
def test_fixed_window_short_and_exact_boundary(count: int) -> None:
    chunks = chunk_unit(
        unit(" ".join(f"w{i}" for i in range(count))),
        ChunkStrategy.FIXED_WINDOW,
        config(),
    )
    assert len(chunks) == 1
    assert chunks[0].word_count == count


def test_tfidf_one_sentence_and_empty_vocabulary_fallbacks() -> None:
    one = chunk_unit(unit("A single short sentence."), ChunkStrategy.TFIDF_BOUNDARY, config())
    empty = chunk_unit(unit("!!! ???"), ChunkStrategy.TFIDF_BOUNDARY, config())

    assert one[0].chunking_metadata["fallback_reason"] == "single_sentence"
    assert empty[0].chunking_metadata["fallback_reason"] == "empty_vocabulary"


def test_tfidf_creates_multi_topic_lexical_boundary() -> None:
    text = "Cats chase mice sleep. Quantum circuits use gates qubits."
    chunks = chunk_unit(unit(text), ChunkStrategy.TFIDF_BOUNDARY, config())

    assert len(chunks) == 2
    assert chunks[0].chunking_metadata["boundary_reason"] == "lexical_similarity"
    assert "".join(chunk.text for chunk in chunks) == text


def test_tfidf_forces_long_sentence_without_losing_content() -> None:
    text = " ".join(f"term{i}" for i in range(24)) + "."
    chunks = chunk_unit(unit(text), ChunkStrategy.TFIDF_BOUNDARY, config())

    assert len(chunks) == 3
    assert all(chunk.word_count <= 10 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def test_tfidf_short_tail_merges_when_within_maximum() -> None:
    text = "alpha beta gamma delta epsilon zeta. unrelated tail."
    chunks = chunk_unit(unit(text), ChunkStrategy.TFIDF_BOUNDARY, config())

    assert len(chunks) == 1
    assert chunks[0].chunking_metadata["short_tail_merged"] is True


def test_tfidf_is_deterministic() -> None:
    parent = unit("One topic has several terms. Another subject uses different words.")
    first = chunk_unit(parent, ChunkStrategy.TFIDF_BOUNDARY, config())
    second = chunk_unit(parent, ChunkStrategy.TFIDF_BOUNDARY, config())

    assert [chunk.model_dump() for chunk in first] == [chunk.model_dump() for chunk in second]


@pytest.mark.parametrize(
    "mapping, message",
    [
        ({"enabled_strategies": ["unknown"], "fixed_window": {"max_words": 10, "overlap_words": 2}, "tfidf_boundary": {"min_words": 3, "target_words": 6, "max_words": 10, "boundary_similarity_threshold": 0.15}}, "unknown"),
        ({"enabled_strategies": ["identity"], "fixed_window": {"max_words": 10, "overlap_words": 10}, "tfidf_boundary": {"min_words": 3, "target_words": 6, "max_words": 10, "boundary_similarity_threshold": 0.15}}, "overlap"),
        ({"enabled_strategies": ["identity"], "fixed_window": {"max_words": 10, "overlap_words": 2}, "tfidf_boundary": {"min_words": 11, "target_words": 6, "max_words": 10, "boundary_similarity_threshold": 0.15}}, "limits"),
        ({"enabled_strategies": ["identity"], "fixed_window": {"max_words": 10, "overlap_words": 2}, "tfidf_boundary": {"min_words": 3, "target_words": 6, "max_words": 10, "boundary_similarity_threshold": -0.1}}, "threshold"),
    ],
)
def test_invalid_configuration_is_rejected(mapping: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ChunkingConfig.from_mapping(mapping)
