"""Offline end-to-end Phase 2 determinism, coverage, and leakage validation."""

from pathlib import Path

from qkef.datasets.evolution import EvolutionBenchmarkConfig, build_benchmark
from qkef.datasets.fiqa import load_fiqa_dataset
from qkef.ingestion.corpus import build_phase2_corpus, validate_phase2_corpus
from qkef.schemas.ingestion import chunk_schema_label_fields


FIXTURE = Path(__file__).parent / "fixtures" / "mini_fiqa"


def phase1_config() -> EvolutionBenchmarkConfig:
    return EvolutionBenchmarkConfig.from_mapping(
        {
            "dataset": {"temporal_variant": "synthetic-mini-evolution"},
            "evolution_benchmark": {
                "seed": 42,
                "generator_version": "test",
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


def phase2_config() -> dict[str, object]:
    return {
        "project": {"seed": 42},
        "ingestion": {
            "version": "test",
            "expected_t0_units": 4,
            "expected_t1_units": 6,
            "normalization_version": "test",
            "sanitizer_version": "test",
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
        },
        "chunking": {
            "enabled_strategies": ["identity", "fixed_window", "tfidf_boundary"],
            "fixed_window": {"max_words": 10, "overlap_words": 2},
            "tfidf_boundary": {
                "min_words": 3,
                "target_words": 6,
                "max_words": 10,
                "boundary_similarity_threshold": 0.15,
            },
        },
    }


def test_offline_phase2_build_is_deterministic_and_valid(tmp_path: Path) -> None:
    dataset = load_fiqa_dataset(FIXTURE)
    phase1_dir = tmp_path / "phase1"
    build_benchmark(dataset, phase1_config(), phase1_dir, expected_fiqa_md5="fixture")
    first_dir = tmp_path / "phase2-first"
    second_dir = tmp_path / "phase2-second"
    first, _ = build_phase2_corpus(phase1_dir, first_dir, phase2_config())
    second, _ = build_phase2_corpus(phase1_dir, second_dir, phase2_config())
    validation = validate_phase2_corpus(phase1_dir, first_dir, phase2_config())

    assert validation.passed, validation.errors
    assert validation.statistics["model_content_coverage_percent"] == 100.0
    assert validation.statistics["source_overlap"] == {
        "train_dev": 0,
        "train_test": 0,
        "dev_test": 0,
    }
    assert first["generated_file_sha256"] == second["generated_file_sha256"]
    assert first["deterministic_rebuild_status"] is True
    assert chunk_schema_label_fields() == set()
