from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qkef.v2.adapters import CSVTemporalAdapter, JSONLTemporalAdapter
from qkef.v2.conformal import MondrianConformalClassifier
from qkef.v2.experiment_lock import ExperimentLock
from qkef.v2.hierarchy import CardinalityAwareResolver
from qkef.v2.retrieval import HybridCandidateResolver, retrieval_metrics
from qkef.v2.state import AuthorityMetadata, KnowledgeState, V2KnowledgeRecord


def test_hybrid_candidate_retrieval_and_metrics_are_deterministic():
    documents = {"a": "annual management fee revised", "b": "weather forecast", "c": "management policy"}
    vectors = {"a": (1.0, 0.0), "b": (0.0, 1.0), "c": (0.8, 0.2)}
    resolver = HybridCandidateResolver(documents, vectors)
    first = resolver.resolve("management fee", (1.0, 0.0), 3)
    second = resolver.resolve("management fee", (1.0, 0.0), 3)
    assert first == second
    assert first[0].identifier == "a"
    assert resolver.resolve("management fee", (1.0, 0.0), 1, mode="dense")[0].identifier == "a"
    assert resolver.resolve("management fee", (1.0, 0.0), 1, mode="lexical")[0].identifier == "a"
    metrics = retrieval_metrics([[item.identifier for item in first]], [{"a"}])
    assert metrics["recall@1"] == metrics["mrr"] == 1.0


def test_mondrian_conformal_sets_and_coverage():
    classes = ["A", "B"]
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    labels = ["A", "A", "B", "B"]
    model = MondrianConformalClassifier(alpha=0.25).fit(probabilities, labels, classes)
    assert model.prediction_set({"A": 0.95, "B": 0.05}) == ("A",)
    coverage = model.coverage([dict(zip(classes, row)) for row in probabilities], labels)
    assert coverage["marginal_coverage"] == 1.0
    assert coverage["average_set_size"] >= 1.0


def test_cardinality_aware_hierarchy():
    resolver = CardinalityAwareResolver()
    assert resolver.resolve({"NEW": 0.8, "REPLACE": 0.2}, 0).cardinality == "0->1"
    assert resolver.resolve({"MERGE": 0.8, "NEW": 0.2}, 2).cardinality == "m->1"
    assert resolver.resolve({"SPLIT": 0.8, "NEW": 0.2}, 1, split_segment_count=2).cardinality == "1->r"
    assert resolver.resolve({"COEXIST": 0.8, "REPLACE": 0.1, "ARCHIVE": 0.1}, 1).action == "COEXIST"


def test_experiment_lock_refuses_silent_post_test_change(tmp_path):
    lock = ExperimentLock("v2-pilot-1", {"alpha": 0.1}, "b", "s", "commit", {}, test_executed=True)
    path = tmp_path / "lock.json"
    lock.save(path)
    assert json.loads(path.read_text())["configuration_hash"] == lock.configuration_hash
    lock.validate_reuse({"alpha": 0.1}, experiment_version="v2-pilot-1")
    with pytest.raises(ValueError, match="new experiment version"):
        lock.validate_reuse({"alpha": 0.2}, experiment_version="v2-pilot-1")


def test_state_hash_is_deterministic_and_authority_unknown_is_preserved():
    item = V2KnowledgeRecord("x", "text", (1.0, 0.0), ("source",), authority=AuthorityMetadata())
    first = KnowledgeState.from_records([item])
    second = KnowledgeState.from_records([item])
    assert first.graph_state_hash == second.graph_state_hash
    assert first.index_state_hash == second.index_state_hash
    assert first.state_hash == second.state_hash
    assert first.records["x"].authority.source_priority is None


def test_v1_baseline_manifest_has_expected_head_and_hashes():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "reports/v1_frozen/QKEF_V1_BASELINE_MANIFEST.json").read_text())
    assert manifest["git"]["head"] == "52bd9f3c703b002c0d03acc0b4ef1a689d86ad9c"
    assert manifest["results"]["qkef_test_macro_f1"] == 0.893331608005521
    assert manifest["baseline_label"] == "Q-KEF v1 frozen experiment"


def test_jsonl_temporal_adapter_preserves_unknown_values(tmp_path):
    path = tmp_path / "versions.jsonl"
    path.write_text('{"source_document_id":"d","version_id":"v1","content":"text"}\n', encoding="utf-8")
    item = JSONLTemporalAdapter().load(path)[0]
    assert item.valid_from is None and item.source_authority is None


def test_csv_temporal_adapter_parses_predecessors(tmp_path):
    path = tmp_path / "versions.csv"
    path.write_text("source_document_id,version_id,content,predecessor_ids\nd,v2,text,v1|v0\n", encoding="utf-8")
    item = CSVTemporalAdapter().load(path)[0]
    assert item.predecessor_ids == ("v1", "v0")
