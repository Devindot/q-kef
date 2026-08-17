from pathlib import Path

import numpy as np

from qkef.embeddings.encoder import DeterministicLexicalEncoder
from qkef.runtime.artifacts import FinalArtifactLoader
from qkef.runtime.demo import initial_demo_records, load_demo_examples, runtime_input
from qkef.runtime.schemas import CLASS_ORDER
from qkef.runtime.service import DemoKnowledgeBase, QKEFService


ROOT = Path(__file__).resolve().parents[1]


def service():
    artifacts = FinalArtifactLoader(ROOT).load()
    return QKEFService(artifacts, DeterministicLexicalEncoder(384))


def test_artifact_loader_verifies_models_and_graphs():
    artifacts = FinalArtifactLoader(ROOT).load()
    assert artifacts.locked_config["test_used_for_selection"] is False
    assert artifacts.state_encoder.dimension == 16
    assert artifacts.graphs["qkef"].graph.number_of_nodes() == 811


def test_runtime_conventional_and_qkef_predictions_are_structured():
    instance = service()
    text = "Mortgage risk depends on income, rates, and contractual terms."
    conventional = instance.analyze_incoming_knowledge(text, "conventional")
    qkef = instance.analyze_incoming_knowledge(text, "qkef")
    assert conventional.predicted_action in CLASS_ORDER
    assert qkef.predicted_action in CLASS_ORDER
    assert list(qkef.class_probabilities) == CLASS_ORDER
    np.testing.assert_allclose(sum(qkef.class_probabilities.values()), 1.0, atol=1e-6)


def test_interactive_output_has_no_ground_truth_or_oracle_fields():
    payload = service().analyze_incoming_knowledge("A neutral financial knowledge update.", "qkef").model_dump()
    forbidden = {"expected_action", "true_action", "ground_truth", "benchmark_split", "expected_target_ids", "mutation_method"}
    assert not forbidden & set(payload)
    assert not any(any(fragment in name for fragment in ("action", "target", "split", "provenance")) for name in payload["feature_values"])


def test_demo_metadata_never_becomes_runtime_input():
    examples = load_demo_examples(ROOT / "data/demo/demo_examples.json")
    for example in examples:
        text = runtime_input(example)
        assert text == example["incoming"]["text"]
        assert example["action_metadata_for_benchmark_display_only"] not in {text}


def test_demo_reset_restores_isolated_records():
    examples = load_demo_examples(ROOT / "data/demo/demo_examples.json")
    records = initial_demo_records(examples)
    kb = DemoKnowledgeBase.from_records(records)
    kb.records["temporary"] = {"identifier": "temporary"}
    kb.reset(records)
    assert "temporary" not in kb.records and len(kb.records) == len(records)


def test_runtime_rejects_empty_and_oversized_input():
    instance = service()
    for text in ("", "x" * 20_001):
        try:
            instance.analyze_incoming_knowledge(text)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe input should fail")
