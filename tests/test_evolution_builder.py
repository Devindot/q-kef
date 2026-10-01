"""Deterministic action-construction tests using synthetic fixture data."""

from __future__ import annotations

import json
from pathlib import Path

from qkef.datasets.evolution import (
    MERGE_DELIMITER,
    SPLIT_DELIMITER,
    EvolutionBenchmarkConfig,
    build_benchmark,
    content_signature,
    generate_benchmark_content,
    mutate_numeric_token,
    validate_benchmark,
)
from qkef.datasets.fiqa import load_fiqa_dataset
from qkef.schemas import EvolutionAction
from qkef.v2.benchmark import partition_by_ancestry


FIXTURE = Path(__file__).parent / "fixtures" / "mini_fiqa"


def mini_config() -> EvolutionBenchmarkConfig:
    return EvolutionBenchmarkConfig.from_mapping(
        {
            "dataset": {"temporal_variant": "synthetic-mini-evolution"},
            "evolution_benchmark": {
                "seed": 42,
                "generator_version": "test-1.0",
                "action_definitions_version": "1.0",
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


def test_numeric_mutation_is_deterministic_and_single_span() -> None:
    original = "A rate of 5% applied in 2020."
    first = mutate_numeric_token(original)
    second = mutate_numeric_token(original)

    assert first == second
    assert first is not None
    assert first.text == "A rate of 6% applied in 2020."
    assert original[first.start : first.end] == first.old_value


def test_generator_builds_every_action_with_expected_semantics() -> None:
    dataset = load_fiqa_dataset(FIXTURE)
    content = generate_benchmark_content(dataset, mini_config())
    by_action = {event.expected_action: event for event in content.events}
    t0 = {unit.knowledge_id: unit for unit in content.t0_units}
    t1 = {unit.knowledge_id: unit for unit in content.t1_units}

    assert set(by_action) == set(EvolutionAction)
    replace = by_action[EvolutionAction.REPLACE]
    assert t1[replace.incoming_knowledge_ids[0]].text != t0[replace.t0_knowledge_ids[0]].text
    merge = by_action[EvolutionAction.MERGE]
    assert t1[merge.incoming_knowledge_ids[0]].text.startswith(t0[merge.t0_knowledge_ids[0]].text + MERGE_DELIMITER)
    archive = by_action[EvolutionAction.ARCHIVE]
    assert archive.expected_target_ids[0] in t1[archive.incoming_knowledge_ids[0]].text
    new = by_action[EvolutionAction.NEW]
    assert not new.t0_knowledge_ids and not new.expected_target_ids
    coexist = by_action[EvolutionAction.COEXIST]
    assert coexist.t0_knowledge_ids and coexist.expected_target_ids
    split = by_action[EvolutionAction.SPLIT]
    assert SPLIT_DELIMITER in t1[split.incoming_knowledge_ids[0]].text
    assert len(split.expected_child_ids) == 2


def test_generation_and_ids_are_deterministic() -> None:
    dataset = load_fiqa_dataset(FIXTURE)
    first = generate_benchmark_content(dataset, mini_config())
    second = generate_benchmark_content(dataset, mini_config())

    assert content_signature(first) == content_signature(second)
    assert [event.event_id for event in first.events] == [event.event_id for event in second.events]


def test_written_rebuild_has_identical_research_file_hashes(tmp_path: Path) -> None:
    dataset = load_fiqa_dataset(FIXTURE)
    first_manifest, _ = build_benchmark(
        dataset, mini_config(), tmp_path / "one", expected_fiqa_md5="fixture", verify_determinism=True
    )
    second_manifest, _ = build_benchmark(
        dataset, mini_config(), tmp_path / "two", expected_fiqa_md5="fixture", verify_determinism=True
    )

    assert first_manifest["deterministic_regeneration_status"] is True
    assert first_manifest["generated_file_sha256"] == second_manifest["generated_file_sha256"]
    assert first_manifest["deterministic_content_sha256"] == second_manifest["deterministic_content_sha256"]
    assert json.loads((tmp_path / "one" / "benchmark_manifest.json").read_text())["seed"] == 42


def test_ancestry_partition_is_deterministic_and_disjoint() -> None:
    dataset = load_fiqa_dataset(FIXTURE)
    weights = {"train": 2, "dev": 1, "calibration": 1, "test": 1}
    first = partition_by_ancestry(dataset, weights, seed=42)
    second = partition_by_ancestry(dataset, weights, seed=42)
    assert first.manifest["assignment_sha256"] == second.manifest["assignment_sha256"]
    assert all(not values["query_count"] and not values["document_count"] for values in first.manifest["cross_split_overlaps"].values())
    assert set(first.query_assignments.values()) <= set(weights)
    assert set(first.document_assignments.values()) <= set(weights)


def test_calibration_split_writes_and_validates_when_configured(tmp_path: Path) -> None:
    dataset = load_fiqa_dataset(FIXTURE)
    mapping = {
        "dataset": {"temporal_variant": "synthetic-mini-v2"},
        "evolution_benchmark": {
            "seed": 42,
            "generator_version": "test-v2",
            "events_per_action": 1,
            "train_events_per_action": 1,
            "dev_events_per_action": 0,
            "calibration_events_per_action": 0,
            "test_events_per_action": 0,
            "t0_timestamp": "2025-01-01T00:00:00Z",
            "t1_timestamp": "2026-01-01T00:00:00Z",
            "actions": ["NEW", "REPLACE", "MERGE", "ARCHIVE", "COEXIST", "SPLIT"],
        },
    }
    v2_config = EvolutionBenchmarkConfig.from_mapping(mapping)
    output = tmp_path / "v2"
    build_benchmark(dataset, v2_config, output, expected_fiqa_md5="fixture")
    assert (output / "calibration_events.jsonl").read_text() == ""
    assert validate_benchmark(output, dataset, v2_config).passed
