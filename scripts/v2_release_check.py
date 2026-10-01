"""Fail-fast release validation for the Q-KEF v2 research prototype."""

from __future__ import annotations

import importlib.util
import csv
import json
from pathlib import Path

import joblib

from _phase1_common import PROJECT_ROOT
from qkef.v2.canonical import sha256_file
from qkef.v2.planning import TransitionPlanner
from qkef.v2.risk import CounterfactualEvaluator, SafeTransitionSelector
from qkef.v2.runtime import QKEFV2Runtime
from qkef.v2.state import KnowledgeState, V2KnowledgeRecord
from qkef.v2.transaction import FailureStage


V2_DOCS = [
    "V2_RESEARCH_SCOPE.md", "V2_HYPOTHESES.md", "V2_EXPERIMENT_PLAN.md", "V2_SYSTEM_ARCHITECTURE.md",
    "V2_COUNTERFACTUAL_TRANSITIONS.md", "V2_EPOCH_PROTOCOL.md", "V2_UNCERTAINTY_MODEL.md", "V2_BITEMPORAL_MODEL.md",
    "V2_AUTHORITY_MODEL.md", "V2_QUANTUM_ABLATION.md", "V2_TECHNICAL_EFFECTS.md", "V2_LIMITATIONS.md", "REAL_TEMPORAL_DATA_PLAN.md",
    "V2_DEVELOPMENT_PROTOCOL.md",
]
PATENT_DOCS = ["PRIOR_ART_BOUNDARIES.md", "INVENTIVE_NUCLEUS.md", "DIFFERENTIATION_MATRIX.md", "SECTION_3K_TECHNICAL_EFFECT_NOTES.md", "INVENTION_DISCLOSURE_DRAFT.md", "CLAIM_CONCEPT_MAP.md", "TECHNICAL_EFFECT_EVIDENCE_PLAN.md", "PRIOR_ART_SEARCH_TERMS.md"]
CONFIRMATORY_FIGURES = {
    "figure_01_model_macro_f1.png",
    "figure_02_candidate_recall5.png",
    "figure_03_conformal_coverage.png",
    "figure_04_execution_precision.png",
    "figure_05_index_size.png",
    "figure_06_hypothesis_outcomes.png",
}


def baseline(root: Path) -> None:
    manifest = json.loads((root / "reports/v1_frozen/QKEF_V1_BASELINE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["git"]["head"] == "52bd9f3c703b002c0d03acc0b4ef1a689d86ad9c"
    for relative, expected in manifest["file_sha256"].items():
        assert sha256_file(root / relative) == expected, f"v1 frozen hash mismatch: {relative}"


def architecture() -> None:
    old = V2KnowledgeRecord("old", "old policy", (1.0, 0.0), ("source",))
    incoming = V2KnowledgeRecord("new", "new policy", (1.0, 0.0), ("source",))
    state = KnowledgeState.from_records([old], epoch_id=12)
    before = state.state_hash
    runtime = QKEFV2Runtime(state, selector=SafeTransitionSelector(maximum_risk=10))
    analysis = runtime.analyze_incoming_knowledge(incoming, ["old"], {"REPLACE": 0.99}, ["REPLACE"])
    assert runtime.state.state_hash == before
    result = runtime.commit_transition(analysis)
    assert result.committed and runtime.state.epoch_id == 13
    assert runtime.state.active_graph_ids() == set(runtime.state.index_entries)
    assert runtime.transactions.rollback(result.certificate.payload["transition_id"]).state_hash == before
    for stage in FailureStage:
        demo = QKEFV2Runtime(KnowledgeState.from_records([old], epoch_id=12), selector=SafeTransitionSelector(maximum_risk=10))
        candidate = demo.analyze_incoming_knowledge(incoming, ["old"], {"REPLACE": 0.99}, ["REPLACE"])
        failed = demo.commit_transition(candidate, fail_at=stage)
        assert not failed.committed and demo.state.state_hash == before


def confirmatory_release(root: Path) -> None:
    report_dir = root / "reports/v2/confirmatory"
    results = json.loads((report_dir / "confirmatory_results.json").read_text(encoding="utf-8"))
    statistics = json.loads((report_dir / "confirmatory_statistics.json").read_text(encoding="utf-8"))
    prelock = json.loads((report_dir / "experiment_lock_pretest.json").read_text(encoding="utf-8"))
    postlock = json.loads((report_dir / "experiment_lock_posttest.json").read_text(encoding="utf-8"))
    assert results["status"] == "CONFIRMATORY_TEST_EXECUTED"
    assert results["test_event_count"] == 480
    assert results["test_evaluation_count"] == 1
    planned = statistics["planned_contrasts"]["Q_FULL_vs_B2_MATCHED_NON_Q"]
    assert planned["challenger"] == "Q_FULL" and planned["baseline"] == "B2_MATCHED_NON_Q"
    assert planned["mcnemar"]["exact_p"] == 0.14961278438568115
    assert prelock["test_executed"] is False and postlock["test_executed"] is True
    assert prelock["configuration_hash"] == postlock["configuration_hash"]
    assert prelock["configuration_hash"] == "142ad4f201a80bfbbadfc0647106d1dcbfa3bc941ed82910e50df3522f4c1272"
    assert prelock["benchmark_hash"] == postlock["benchmark_hash"]
    assert prelock["split_hash"] == postlock["split_hash"]
    assert prelock["model_hashes"] == postlock["model_hashes"]
    for name, expected in prelock["model_hashes"].items():
        path = root / "models/v2_confirmatory" / name
        assert sha256_file(path) == expected, f"confirmatory model hash mismatch: {name}"
        joblib.load(path)

    assert {path.name for path in (report_dir / "figures").glob("figure_*.png")} == CONFIRMATORY_FIGURES
    audit = json.loads((root / "reports/v2/human_audit/HUMAN_AUDIT_STATUS.json").read_text(encoding="utf-8"))
    assert audit["automated_integrity_validation"] == "PASS_ALL_2400"
    assert audit["status"] == "AWAITING_HUMAN_SIGNOFF"
    assert audit["human_review_complete"] is False
    with (root / "reports/v2/human_audit/stratified_review_sample.csv").open(newline="", encoding="utf-8") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 120

    manifest = json.loads((root / "reports/v2/final_release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_status"] == "MACHINE_COMPLETE_HUMAN_SIGNOFF_PENDING"
    assert manifest["test_evaluation_count"] == 1 and manifest["test_event_count"] == 480
    assert manifest["human_audit_status"] == "AWAITING_HUMAN_SIGNOFF"
    assert len(manifest["file_sha256"]) >= 40
    for relative, expected in manifest["file_sha256"].items():
        assert sha256_file(root / relative) == expected, f"final release hash mismatch: {relative}"


def outputs(root: Path) -> None:
    for name in V2_DOCS: assert (root / "docs/v2" / name).is_file(), name
    for name in PATENT_DOCS: assert (root / "docs/patent" / name).is_file(), name
    for name in ("v2_results.json", "statistical_analysis.json", "V2_EXPERIMENT_REPORT.md", "STATISTICAL_ANALYSIS.md", "POWER_ANALYSIS.md", "TECHNICAL_EFFECT_RESULTS.md", "ERROR_ANALYSIS.md", "PRIOR_ART_DIFFERENTIATION_SUMMARY.md", "experiment_lock.json"):
        assert (root / "reports/v2" / name).is_file(), name
    assert len(list((root / "reports/v2/patent_figures").glob("figure_*.png"))) == 12
    lock = json.loads((root / "reports/v2/experiment_lock.json").read_text(encoding="utf-8"))
    assert lock["test_executed"] is True
    assert len(lock["model_hashes"]) == len(list((root / "models/v2").glob("*.joblib")))
    for name, expected in lock["model_hashes"].items():
        path = root / "models/v2" / name
        assert sha256_file(path) == expected
        joblib.load(path)
    development = json.loads((root / "reports/v2/development/dev_retrieval_selection.json").read_text(encoding="utf-8"))
    assert development["selection_split"] == "dev"
    assert development["test_observations_used"] == 0
    assert development["confirmatory_result"] is False
    assert development["eligible_dev_query_count"] > 0
    witness_audit = json.loads((root / "reports/v2/development/dev_witness_audit.json").read_text(encoding="utf-8"))
    assert witness_audit["selection_split"] == "dev"
    assert witness_audit["test_observations_used"] == 0
    assert witness_audit["confirmatory_result"] is False
    assert witness_audit["event_count"] > 0
    assert witness_audit["overall"]["revised_probe_mean_current_evidence_miss"] <= witness_audit["overall"]["legacy_probe_mean_current_evidence_miss"]
    confirmatory = json.loads((root / "reports/v2/confirmatory_benchmark/benchmark_manifest.json").read_text(encoding="utf-8"))
    partition = json.loads((root / "reports/v2/confirmatory_benchmark/ancestry_partition_manifest.json").read_text(encoding="utf-8"))
    assert confirmatory["actual_number_of_events"] == 2400
    assert set(confirmatory["count_per_action"].values()) == {400}
    assert confirmatory["count_per_benchmark_split"] == {"train": 1200, "dev": 360, "calibration": 360, "test": 480}
    assert confirmatory["integrity_validation_status"] == confirmatory["leakage_validation_status"] == "pass"
    assert confirmatory["test_evaluation_status"] == "not_executed"
    assert all(not values["query_count"] and not values["document_count"] for values in partition["cross_split_overlaps"].values())
    confirmatory_release(root)
    spec = importlib.util.spec_from_file_location("qkef_v2_app", root / "app_v2.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def main() -> None:
    baseline(PROJECT_ROOT); architecture(); outputs(PROJECT_ROOT)
    print("PASS: v1 hashes; v2 leakage separation; 2,400-event ancestry-isolated benchmark; one-time 480-event confirmatory TEST receipt; pre/post locks; model and final-release hashes; audit pack; shadow immutability; epoch consistency; failure restoration; rollback; certificates; reports; figures; and offline app import validated.")


if __name__ == "__main__":
    main()
