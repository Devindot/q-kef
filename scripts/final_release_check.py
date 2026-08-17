"""Fail-fast end-to-end validator for the Q-KEF final academic release."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from _phase1_common import PROJECT_ROOT
from qkef.datasets.evolution import load_jsonl_models
from qkef.evaluation.final_analysis import release_digest
from qkef.qa import EvidenceQA
from qkef.qa.extractive import EvidenceRecord
from qkef.runtime.artifacts import FinalArtifactLoader
from qkef.runtime.demo import load_demo_examples, runtime_input
from qkef.runtime.results import load_final_results
from qkef.runtime.service import QKEFService
from qkef.schemas import EvolutionEvent, IngestedKnowledgeUnit


DOCS = [
    "RESEARCH_CLAIMS_AND_EVIDENCE.md", "FINAL_TECHNICAL_FACT_SHEET.md", "FINAL_SYSTEM_ARCHITECTURE.md",
    "REPRODUCIBILITY.md", "FINAL_DEMO_GUIDE.md", "DEMO_DAY_CHECKLIST.md", "FINAL_VIVA_GUIDE.md",
    "PROJECT_SUMMARY_FOR_TEAMMATE.md", "LIMITATIONS_AND_RESPONSIBLE_USE.md", "PAPER_WRITING_NOTES.md", "PRESENTATION_OUTLINE.md",
]
FIGURES = ["candidate_retrieval.png", "lifecycle_model_comparison.png", "per_class_f1.png", "conventional_confusion_matrix.png", "qkef_confusion_matrix.png", "retrieval_comparison.png", "obsolete_retrieval_rate.png", "quantum_ablation.png"]


def core_checks(root: Path = PROJECT_ROOT) -> dict:
    metadata = json.loads((root / "data/metadata/fiqa_source.json").read_text(encoding="utf-8"))
    assert metadata["expected_md5"] == "17918ed23cd04fb15047f73e6c3bd9d9"
    events = load_jsonl_models(root / "data/processed/qkef_fiqa_evolution/events.jsonl", EvolutionEvent)
    assert len(events) == 300
    phase2 = json.loads((root / "data/processed/qkef_fiqa_chunks/phase2_manifest.json").read_text(encoding="utf-8"))
    assert phase2["ingested_counts"]["total"] == 500
    assert phase2["integrity_validation_status"] == "pass"
    assert phase2["label_leakage_audit_status"] == "pass"
    artifacts = FinalArtifactLoader(root).load()
    assert artifacts.manifest["leakage_audit"]["feature_names_safe"] is True
    results = load_final_results(root)
    assert results["scientific_results_frozen"] is True
    assert results["candidate_retrieval"]["recall@5"] == artifacts.manifest["candidate_metrics"]["recall@5"]
    assert results["lifecycle_classification"] == artifacts.model_metrics
    for name in FIGURES: assert (root / "reports/final/figures" / name).is_file()
    for name in DOCS: assert (root / "docs" / name).is_file()
    for name in ("app.py", "FINAL_PROJECT_STATUS.md", "reports/FINAL_EXPERIMENT_REPORT.md", "reports/final/STATISTICAL_ANALYSIS.md", "reports/final/final_release_manifest.json"):
        assert (root / name).is_file()
    release_manifest = json.loads((root / "reports/final/final_release_manifest.json").read_text(encoding="utf-8"))
    assert release_manifest["source_phase3_commit"] == "1a7c3e2de2dfb81d0eb0b3897493540de6205431"
    for relative, expected_hash in release_manifest["files"].items():
        assert release_digest(root / relative) == expected_hash, f"final release hash mismatch: {relative}"
    examples = load_demo_examples(root / "data/demo/demo_examples.json")
    assert len(examples) == 6 and all(runtime_input(item) for item in examples)
    spec = importlib.util.spec_from_file_location("qkef_final_app", root / "app.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return {"artifacts": artifacts, "examples": examples, "results": results}


def runtime_checks(state: dict) -> None:
    service = QKEFService.load_default(local_files_only=True)
    text = runtime_input(state["examples"][1])
    conventional = service.analyze_incoming_knowledge(text, "conventional")
    qkef = service.analyze_incoming_knowledge(text, "qkef")
    forbidden = {"expected_action", "true_action", "ground_truth", "benchmark_split", "expected_target_ids"}
    assert not forbidden & set(conventional.model_dump())
    assert conventional.predicted_action and qkef.predicted_action and qkef.state_amplitudes
    assert service.artifacts.graphs["qkef"].graph.number_of_nodes() > 0
    unit = next(item for item in service.artifacts.units if item.temporal_state.value == "T0")
    vector = service.artifacts.embedding_vectors[service.artifacts.embedding_index[unit.knowledge_id]]
    qa = EvidenceQA([EvidenceRecord(unit.knowledge_id, unit.model_text, unit.source_document_ids, "active", vector)], service.encoder).answer(unit.model_text[:80], minimum_score=0.0)
    assert qa.supporting_passages


def main() -> None:
    state = core_checks()
    runtime_checks(state)
    print("PASS: Phases 1–4, trusted artifacts, runtime B/C inference, graphs, QA, figures, reports, docs, and frozen results validated.")


if __name__ == "__main__":
    main()
