import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_application_helpers_import_without_starting_server():
    spec = importlib.util.spec_from_file_location("qkef_app_test", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert callable(module.graph_figure) and callable(module.probability_frame)
    assert "not financial advice" in module.DISCLAIMER


def test_release_checker_core_when_outputs_exist():
    if not (ROOT / "reports/final/final_results.json").exists(): pytest.skip("generated after first pytest")
    from scripts.final_release_check import core_checks
    state = core_checks(ROOT)
    assert state["results"]["release"] == "Phase 4 Final Academic Release"


def test_runtime_feature_csvs_remain_label_free():
    forbidden = ("NEW", "REPLACE", "MERGE", "ARCHIVE", "COEXIST", "SPLIT", "qkef-train", "qkef-dev", "qkef-test")
    for name in ("conventional_features.csv", "qkef_features.csv"):
        content = (ROOT / "data/processed/qkef_phase3" / name).read_text(encoding="utf-8")
        assert not any(value in content for value in forbidden)
