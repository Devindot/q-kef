"""Regression tests for strong benchmark-integrity failures."""

from __future__ import annotations

import json
from pathlib import Path

from qkef.datasets.evolution import (
    EvolutionBenchmarkConfig,
    build_benchmark,
    canonical_json,
    validate_benchmark,
)
from qkef.datasets.fiqa import load_fiqa_dataset
from qkef.schemas import BenchmarkSplit, EvolutionEvent


FIXTURE = Path(__file__).parent / "fixtures" / "mini_fiqa"


def config() -> EvolutionBenchmarkConfig:
    return EvolutionBenchmarkConfig.from_mapping(
        {
            "dataset": {"temporal_variant": "synthetic-mini-evolution"},
            "evolution_benchmark": {
                "seed": 42,
                "generator_version": "test-1.0",
                "events_per_action": 1,
                "train_events_per_action": 1,
                "dev_events_per_action": 0,
                "test_events_per_action": 0,
                "t0_timestamp": "2025-01-01T00:00:00Z",
                "t1_timestamp": "2026-01-01T00:00:00Z",
                "actions": ["NEW", "REPLACE", "MERGE", "ARCHIVE", "COEXIST", "SPLIT"],
            },
        }
    )


def build_fixture(tmp_path: Path) -> tuple[Path, object]:
    dataset = load_fiqa_dataset(FIXTURE)
    output = tmp_path / "benchmark"
    build_benchmark(dataset, config(), output, expected_fiqa_md5="fixture")
    return output, dataset


def rewrite_events(path: Path, events: list[EvolutionEvent]) -> None:
    path.write_text("".join(canonical_json(event) + "\n" for event in events), encoding="utf-8")


def test_valid_fixture_benchmark_passes(tmp_path: Path) -> None:
    output, dataset = build_fixture(tmp_path)
    result = validate_benchmark(output, dataset, config())

    assert result.passed, result.errors
    assert result.statistics["source_overlap"] == {"train_dev": 0, "train_test": 0, "dev_test": 0}


def test_duplicate_event_is_detected(tmp_path: Path) -> None:
    output, dataset = build_fixture(tmp_path)
    events_path = output / "events.jsonl"
    first_line = events_path.read_text(encoding="utf-8").splitlines()[0]
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(first_line + "\n")

    result = validate_benchmark(output, dataset, config())
    assert any("duplicate event ID" in error for error in result.errors)


def test_unresolved_reference_is_detected(tmp_path: Path) -> None:
    output, dataset = build_fixture(tmp_path)
    events = [
        EvolutionEvent.model_validate_json(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    target_index = next(index for index, event in enumerate(events) if event.expected_target_ids)
    events[target_index] = events[target_index].model_copy(update={"expected_target_ids": ["missing-t0"]})
    rewrite_events(output / "events.jsonl", events)

    result = validate_benchmark(output, dataset, config())
    assert any("unresolved T0 reference" in error for error in result.errors)


def test_cross_split_ancestry_leakage_is_detected(tmp_path: Path) -> None:
    output, dataset = build_fixture(tmp_path)
    events = [
        EvolutionEvent.model_validate_json(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first, second = events[0], events[1]
    events[1] = second.model_copy(
        update={
            "benchmark_split": BenchmarkSplit.DEV,
            "source_document_ids": [first.source_document_ids[0], second.source_document_ids[-1]],
        }
    )
    rewrite_events(output / "events.jsonl", events)

    result = validate_benchmark(output, dataset, config())
    assert any("source ancestry leakage train/dev" in error for error in result.errors)
