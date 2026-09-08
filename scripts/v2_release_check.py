"""Fail-fast release validation for the Q-KEF v2 research prototype."""

from __future__ import annotations

import importlib.util
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
]
PATENT_DOCS = ["PRIOR_ART_BOUNDARIES.md", "INVENTIVE_NUCLEUS.md", "DIFFERENTIATION_MATRIX.md", "SECTION_3K_TECHNICAL_EFFECT_NOTES.md", "INVENTION_DISCLOSURE_DRAFT.md", "CLAIM_CONCEPT_MAP.md", "TECHNICAL_EFFECT_EVIDENCE_PLAN.md", "PRIOR_ART_SEARCH_TERMS.md"]


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
    spec = importlib.util.spec_from_file_location("qkef_v2_app", root / "app_v2.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def main() -> None:
    baseline(PROJECT_ROOT); architecture(); outputs(PROJECT_ROOT)
    print("PASS: v1 hashes, v2 leakage separation, shadow immutability, epoch consistency, failure restoration, rollback, certificates, models, reports, figures, and offline app import validated.")


if __name__ == "__main__":
    main()
