from pathlib import Path

import pandas as pd
import pytest

from qkef.evaluation.final_analysis import paired_bootstrap, supplementary_rates_for_rankings
from qkef.runtime.artifacts import FinalArtifactLoader
from qkef.runtime.results import load_final_results


ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_is_deterministic_and_mcnemar_counts_exact():
    frame = pd.DataFrame({"true_action": ["NEW", "NEW", "MERGE", "MERGE"], "conventional_prediction": ["NEW", "MERGE", "MERGE", "NEW"], "qkef_prediction": ["NEW", "NEW", "MERGE", "NEW"]})
    first = paired_bootstrap(frame, iterations=100, seed=42)
    second = paired_bootstrap(frame, iterations=100, seed=42)
    assert first == second
    assert first["mcnemar"]["b_correct_c_wrong"] == 0
    assert first["mcnemar"]["b_wrong_c_correct"] == 1


def test_supplementary_metric_formulas_known_example():
    values = supplementary_rates_for_rankings([["old", "new"], ["good"]], {"old"}, [{"new"}, {"good"}], [{"new"}, {"good"}], 2)
    assert values["active_valid_hit_rate"] == 2 / 3
    assert values["obsolete_free_query_rate"] == 0.5
    assert values["lineage_normalized_hit_rate"] == 1.0
    assert values["current_version_hit_rate"] == 1.0


def test_final_results_match_frozen_phase3_when_present():
    path = ROOT / "reports/final/final_results.json"
    if not path.exists(): pytest.skip("generated after the prescribed first pytest")
    final = load_final_results(ROOT); phase3 = FinalArtifactLoader(ROOT).load()
    assert final["candidate_retrieval"] == phase3.manifest["candidate_metrics"]
    assert final["lifecycle_classification"] == phase3.model_metrics
    assert final["scientific_results_frozen"] is True
