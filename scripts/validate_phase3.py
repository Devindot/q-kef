"""Fail-fast validation of Phase 3 artifacts and leakage controls."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd

from _phase1_common import PROJECT_ROOT, load_config, repository_path
from qkef.datasets.evolution import sha256_file
from qkef.graph.semantic_graph import SemanticGraph


def main() -> None:
    config, _ = load_config("configs/default.yaml")
    output = repository_path(config["phase3"]["output_dir"])
    models = repository_path(config["phase3"]["model_dir"])
    reports = repository_path(config["phase3"]["report_dir"])
    manifest = json.loads((output / "phase3_manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedding"]["backend"] == "sentence_transformers"
    assert manifest["embedding"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert manifest["embedding"]["dimension"] == 384
    assert manifest["reload_verified"] is True
    assert manifest["selected"]["test_used_for_selection"] is False
    assert manifest["pca_fit_split"] == "TRAIN"
    assert manifest["pca_fit_ids"] and all("-train-" in item for item in manifest["pca_fit_ids"])
    assert manifest["leakage_audit"]["feature_names_safe"] is True
    assert manifest["event_counts"] == {"dev": 60, "test": 60, "train": 180}
    labels = pd.read_csv(output / "evaluation_labels.csv")
    split_ids = {name: set(labels.loc[labels["split"] == name, "sample_id"]) for name in ("train", "dev", "test")}
    assert not split_ids["train"] & split_ids["dev"]
    assert not split_ids["train"] & split_ids["test"]
    assert not split_ids["dev"] & split_ids["test"]
    forbidden_values = ("NEW", "REPLACE", "MERGE", "ARCHIVE", "COEXIST", "SPLIT", "qkef-train", "qkef-dev", "qkef-test")
    for filename in ("conventional_features.csv", "qkef_features.csv"):
        content = (output / filename).read_text(encoding="utf-8")
        assert not any(value in content for value in forbidden_values), f"label/ID leakage in {filename}"
    for filename, digest in manifest["artifact_sha256"].items():
        path = models / filename if filename.endswith(".joblib") else output / filename
        assert path.is_file() and sha256_file(path) == digest
    for path in (models / "conventional_evolution.joblib", models / "qkef_evolution.joblib", models / "quantum_state_encoder.joblib"):
        assert joblib.load(path) is not None
    for name in ("append_only", "conventional", "qkef", "oracle"):
        graph = SemanticGraph.load(output / f"graph_{name}.json")
        assert graph.graph.number_of_nodes() > 0
    required = ["locked_config.json", "test_predictions.csv", "classification_errors.csv", "evolution_model_metrics.json", "retrieval_metrics.json", "LABEL_LEAKAGE_AUDIT.md"]
    assert all((reports / item).is_file() for item in required)
    assert (PROJECT_ROOT / "reports" / "PHASE3_CORE_RESEARCH_RESULTS.md").is_file()
    for system in ("conventional", "qkef"):
        for split in ("DEV", "TEST"):
            assert math.isfinite(manifest["model_metrics"][system][split]["macro_f1"])
    print("PASS: Phase 3 artifacts, provenance, held-out protocol, reloads, graphs, metrics, and leakage controls validated.")


if __name__ == "__main__":
    main()
