from __future__ import annotations

from pathlib import Path

import numpy as np

from qkef.datasets.evolution import EvolutionBenchmarkConfig, build_benchmark
from qkef.datasets.fiqa import load_fiqa_dataset
from qkef.embeddings.encoder import DeterministicLexicalEncoder
from qkef.v2.confirmatory import MODEL_NAMES, candidate_rows, feature_matrices, fit_state_encoder, load_events, load_units
from qkef.v2.retrieval import CandidateConfiguration


FIXTURE = Path(__file__).parent / "fixtures" / "mini_fiqa"
NORMALIZATION = {"unicode_form": "NFKC", "normalize_line_endings": True, "remove_zero_width": True, "trim_whitespace": True, "normalize_horizontal_whitespace": True, "max_blank_lines": 2}
SANITIZATION = {"remove_merge_scaffold": True, "neutralize_split_boundary": True, "remove_archive_target_id_from_text": True}


def mini_config() -> EvolutionBenchmarkConfig:
    return EvolutionBenchmarkConfig.from_mapping({
        "dataset": {"temporal_variant": "mini-confirmatory"},
        "evolution_benchmark": {
            "seed": 42,
            "generator_version": "test",
            "events_per_action": 1,
            "train_events_per_action": 1,
            "dev_events_per_action": 0,
            "calibration_events_per_action": 0,
            "test_events_per_action": 0,
            "t0_timestamp": "2025-01-01T00:00:00Z",
            "t1_timestamp": "2026-01-01T00:00:00Z",
            "actions": ["NEW", "REPLACE", "MERGE", "ARCHIVE", "COEXIST", "SPLIT"],
        },
    })


def test_confirmatory_feature_pipeline_is_label_safe_and_complete(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    build_benchmark(load_fiqa_dataset(FIXTURE), mini_config(), benchmark, expected_fiqa_md5="fixture")
    events = load_events(benchmark, ("train",))
    units = load_units(benchmark, {"train"}, NORMALIZATION, SANITIZATION)
    encoder = DeterministicLexicalEncoder(dimension=32)
    vectors = encoder.encode([unit.text for unit in units])
    vector_index = {unit.identifier: index for index, unit in enumerate(units)}
    rows = candidate_rows(events, units, vectors, vector_index, CandidateConfiguration("dense", dense_weight=1.0, lexical_weight=0.0), 5)
    state_encoder = fit_state_encoder(units, vectors, vector_index, 4, 42)
    features = feature_matrices(rows, units, vectors, vector_index, state_encoder)
    assert len(rows) == len(events) == 6
    assert tuple(features) == MODEL_NAMES
    assert all(values.shape[0] == 6 and np.isfinite(values).all() for values in features.values())
    archive = next(row for row in rows if row.event.expected_action.value == "ARCHIVE")
    assert "qkef-train-archive" not in archive.incoming.text
