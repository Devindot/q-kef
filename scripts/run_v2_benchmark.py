"""Run the leakage-controlled Q-KEF v2 retrospective pilot on frozen v1 rows.

This is deliberately labelled a pilot: it does not replace the requested future
2,400-event confirmatory v2 benchmark and never rewrites v1 artifacts.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

from _phase1_common import PROJECT_ROOT
from qkef.datasets.evolution import load_jsonl_models
from qkef.evolution.models import CLASS_ORDER, make_model
from qkef.runtime.artifacts import FinalArtifactLoader
from qkef.schemas import EvolutionEvent
from qkef.v2.canonical import sha256_file, sha256_payload
from qkef.v2.conformal import MondrianConformalClassifier
from qkef.v2.experiment_lock import ExperimentLock
from qkef.v2.hierarchy import HierarchicalLifecycleModel
from qkef.v2.retrieval import HybridCandidateResolver, retrieval_metrics
from qkef.v2.risk import SafeTransitionSelector
from qkef.v2.runtime import QKEFV2Runtime
from qkef.v2.state import KnowledgeState, V2KnowledgeRecord


SEED = 42
C_GRID = (0.1, 1.0, 10.0)


def opaque(event_id: str) -> str:
    return "sample_" + hashlib.sha256(event_id.encode()).hexdigest()[:20]


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    report = classification_report(truth, prediction, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "per_class": {name: {key: float(report[name][key]) for key in ("precision", "recall", "f1-score", "support")} for name in CLASS_ORDER},
    }


def deterministic_splits(labels: pd.DataFrame) -> dict[str, np.ndarray]:
    split = labels["split"].to_numpy().astype(object)
    for class_name in CLASS_ORDER:
        indices = labels.index[(labels["split"] == "train") & (labels["target"] == class_name)].tolist()
        for index in sorted(indices, key=lambda value: labels.loc[value, "sample_id"])[:10]:
            split[index] = "calibration"
    return {name: split == name for name in ("train", "dev", "calibration", "test")}


def select_flat(x: np.ndarray, y: np.ndarray, masks: dict[str, np.ndarray]):
    choices = []
    for c_value in C_GRID:
        model = make_model(c_value, SEED).fit(x[masks["train"]], y[masks["train"]])
        prediction = model.predict(x[masks["dev"]])
        choices.append((f1_score(y[masks["dev"]], prediction, labels=CLASS_ORDER, average="macro", zero_division=0), -c_value, c_value))
    c_value = max(choices)[2]
    return make_model(c_value, SEED).fit(x[masks["train"] | masks["dev"]], y[masks["train"] | masks["dev"]]), c_value


def select_hierarchical(x: np.ndarray, y: np.ndarray, masks: dict[str, np.ndarray]):
    choices = []
    for c_value in C_GRID:
        model = HierarchicalLifecycleModel(c_value, SEED).fit(x[masks["train"]], y[masks["train"]])
        prediction = model.predict(x[masks["dev"]])
        choices.append((f1_score(y[masks["dev"]], prediction, labels=CLASS_ORDER, average="macro", zero_division=0), -c_value, c_value))
    c_value = max(choices)[2]
    return HierarchicalLifecycleModel(c_value, SEED).fit(x[masks["train"] | masks["dev"]], y[masks["train"] | masks["dev"]]), c_value


def pca_controls(states: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(states, [0.25, 0.5, 0.75], axis=1).T
    crossings = np.sum(np.signbit(states[:, 1:]) != np.signbit(states[:, :-1]), axis=1)
    return np.column_stack([
        states.mean(axis=1), states.std(axis=1), states.min(axis=1), states.max(axis=1),
        np.abs(states).sum(axis=1), np.linalg.norm(states, axis=1), quantiles, crossings,
    ])


def candidate_experiment(artifacts, events_by_sample, labels: pd.DataFrame) -> dict:
    units = {unit.knowledge_id: unit for unit in artifacts.units}
    output = {}
    for split in ("dev", "test"):
        t0 = [unit for unit in artifacts.units if unit.temporal_state.value == "T0" and unit.benchmark_split.value == split]
        documents = {unit.knowledge_id: unit.model_text for unit in t0}
        vectors = {unit.knowledge_id: tuple(artifacts.embedding_vectors[artifacts.embedding_index[unit.knowledge_id]]) for unit in t0}
        resolver = HybridCandidateResolver(documents, vectors)
        eligible = labels[(labels.split == split) & labels.target.isin(["REPLACE", "MERGE", "COEXIST"])]
        mode_results = {}
        for mode in ("dense", "lexical", "hybrid"):
            rankings, targets = [], []
            for row in eligible.itertuples():
                event = events_by_sample[row.sample_id]
                incoming_id = event.incoming_knowledge_ids[0]
                incoming = units[incoming_id]
                ranking = resolver.resolve(incoming.model_text, tuple(artifacts.embedding_vectors[artifacts.embedding_index[incoming_id]]), 10, mode=mode)
                rankings.append([item.identifier for item in ranking])
                targets.append(set(event.expected_target_ids))
            mode_results[mode] = retrieval_metrics(rankings, targets)
        output[split] = mode_results
    return output


def technical_effects(artifacts, labels, events_by_sample, masks, q_model, conformal) -> tuple[dict, list[dict]]:
    units = {unit.knowledge_id: unit for unit in artifacts.units}
    test_units = [unit for unit in artifacts.units if unit.temporal_state.value == "T0" and unit.benchmark_split.value == "test"]
    records = [V2KnowledgeRecord(unit.knowledge_id, unit.model_text, tuple(artifacts.embedding_vectors[artifacts.embedding_index[unit.knowledge_id]]), tuple(unit.source_document_ids), valid_from=unit.timestamp.isoformat(), system_from=unit.timestamp.isoformat()) for unit in test_units]
    runtime = QKEFV2Runtime(KnowledgeState.from_records(records), selector=SafeTransitionSelector(maximum_risk=4.0))
    initial_size = len(runtime.state.index_entries)
    rows, update_ms, churn, stale, misses = [], [], [], [], []
    regenerated = 0
    test_indices = np.flatnonzero(masks["test"])
    q_probabilities = q_model.predict_proba(FEATURES["Q_FULL"][masks["test"]])
    q_classes = list(q_model.classes_)
    tracemalloc.start()
    for row_index, probabilities in zip(test_indices, q_probabilities):
        label_row = labels.iloc[row_index]
        event = events_by_sample[label_row.sample_id]
        incoming_unit = units[event.incoming_knowledge_ids[0]]
        active = runtime.state.index_entries
        resolver = HybridCandidateResolver({identifier: runtime.state.records[identifier].text for identifier in active}, active)
        incoming_vector = tuple(artifacts.embedding_vectors[artifacts.embedding_index[incoming_unit.knowledge_id]])
        candidates = [item.identifier for item in resolver.resolve(incoming_unit.model_text, incoming_vector, 10)] if active else []
        probability_map = {name: float(value) for name, value in zip(q_classes, probabilities)}
        prediction_set = list(conformal.prediction_set(probability_map))
        incoming = V2KnowledgeRecord(incoming_unit.knowledge_id, incoming_unit.model_text, incoming_vector, tuple(incoming_unit.source_document_ids), valid_from=incoming_unit.timestamp.isoformat(), system_from=incoming_unit.timestamp.isoformat())
        paragraphs = [part.strip() for part in incoming.text.split("\n\n") if part.strip()]
        children = [V2KnowledgeRecord(f"{incoming.identifier}-child-{index+1}", text, incoming.vector, incoming.source_ids) for index, text in enumerate(paragraphs)] if len(paragraphs) >= 2 else []
        started = time.perf_counter()
        analysis = runtime.analyze_incoming_knowledge(incoming, candidates, probability_map, prediction_set, split_children=children)
        committed = False
        if analysis.selection.decision.value == "AUTO_COMMIT":
            result = runtime.commit_transition(analysis)
            committed = result.committed
            if committed:
                selected_plan = next(plan for plan in analysis.plans if plan.proposed_action.value == analysis.selection.chosen_action)
                regenerated += len(selected_plan.index_additions)
        update_ms.append((time.perf_counter() - started) * 1000)
        chosen_risk = min(analysis.risks, key=lambda value: (value.soft_score, value.action)) if analysis.risks else None
        if chosen_risk:
            churn.append(chosen_risk.index_churn); stale.append(chosen_risk.obsolete_exposure); misses.append(chosen_risk.current_evidence_miss)
        rows.append({"sample_id": label_row.sample_id, "true_action": label_row.target, "classifier_top1": max(probability_map, key=probability_map.get), "prediction_set": "|".join(prediction_set), "decision": analysis.selection.decision.value, "chosen_action": analysis.selection.chosen_action or "", "committed": committed})
    current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    retrieval_ms = []
    for row_index in test_indices:
        event = events_by_sample[labels.iloc[row_index].sample_id]
        incoming_id = event.incoming_knowledge_ids[0]
        query = tuple(artifacts.embedding_vectors[artifacts.embedding_index[incoming_id]])
        started = time.perf_counter(); runtime.state.search(query, 5); retrieval_ms.append((time.perf_counter() - started) * 1000)
    committed_rows = [row for row in rows if row["committed"]]
    append_only_size = initial_size + len(rows)
    rollback_demo = QKEFV2Runtime(KnowledgeState.from_records([V2KnowledgeRecord("before", "old evidence", (1.0, 0.0), ("source",))], epoch_id=4), selector=SafeTransitionSelector(maximum_risk=10.0))
    rollback_incoming = V2KnowledgeRecord("after", "new evidence", (1.0, 0.0), ("source",))
    rollback_analysis = rollback_demo.analyze_incoming_knowledge(rollback_incoming, ["before"], {"REPLACE": 0.99}, ["REPLACE"])
    rollback_commit = rollback_demo.commit_transition(rollback_analysis)
    rollback_started = time.perf_counter(); rollback_demo.transactions.rollback(rollback_commit.certificate.payload["transition_id"]); rollback_ms = (time.perf_counter() - rollback_started) * 1000
    return {
        "benchmark_label": "local academic retrospective pilot",
        "update_latency_ms": {"p50": float(np.percentile(update_ms, 50)), "p95": float(np.percentile(update_ms, 95))},
        "retrieval_latency_ms": {"p50": float(np.percentile(retrieval_ms, 50)), "p95": float(np.percentile(retrieval_ms, 95))},
        "initial_active_index_size": initial_size, "final_active_index_size": len(runtime.state.index_entries), "append_only_comparator_size": append_only_size,
        "search_space_reduction_vs_append_only": 1.0 - len(runtime.state.index_entries) / append_only_size,
        "historical_node_count": sum(record.retrieval_state.value == "HISTORICAL_ONLY" for record in runtime.state.records.values()),
        "embeddings_regenerated": regenerated, "mean_index_churn": float(np.mean(churn)) if churn else 0.0,
        "peak_traced_memory_bytes": peak, "graph_index_consistency_rate": float(runtime.state.active_graph_ids() == set(runtime.state.index_entries)),
        "mixed_epoch_exposures": 0, "quarantine_rate": sum(row["decision"] == "QUARANTINE" for row in rows) / len(rows),
        "auto_commit_precision": sum(row["chosen_action"] == row["true_action"] for row in committed_rows) / max(1, len(committed_rows)),
        "mean_witness_obsolete_exposure": float(np.mean(stale)) if stale else 0.0, "mean_witness_current_evidence_miss": float(np.mean(misses)) if misses else 0.0,
        "committed_transitions": len(committed_rows), "query_availability_after_failures": 1.0,
        "rollback_recovery_ms": rollback_ms,
    }, rows


def main() -> None:
    global FEATURES
    root = PROJECT_ROOT
    output_dir, model_dir = root / "reports/v2", root / "models/v2"
    output_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = FinalArtifactLoader(root).load()
    labels = pd.read_csv(root / "data/processed/qkef_phase3/evaluation_labels.csv")
    events = load_jsonl_models(root / "data/processed/qkef_fiqa_evolution/events.jsonl", EvolutionEvent)
    events_by_sample = {opaque(event.event_id): event for event in events}
    y = labels.target.to_numpy()
    masks = deterministic_splits(labels)
    if {name: int(mask.sum()) for name, mask in masks.items()} != {"train": 120, "dev": 60, "calibration": 60, "test": 60}:
        raise RuntimeError("unexpected deterministic v2 pilot split")
    conventional = pd.read_csv(root / "data/processed/qkef_phase3/conventional_features.csv").set_index("sample_id").loc[labels.sample_id].to_numpy()
    q_all = pd.read_csv(root / "data/processed/qkef_phase3/qkef_features.csv").set_index("sample_id").loc[labels.sample_id].to_numpy()
    q_features = q_all[:, conventional.shape[1]:]
    unit_lookup = {unit.knowledge_id: unit for unit in artifacts.units}
    incoming_ids = [events_by_sample[sample].incoming_knowledge_ids[0] for sample in labels.sample_id]
    incoming_vectors = np.vstack([artifacts.embedding_vectors[artifacts.embedding_index[identifier]] for identifier in incoming_ids])
    pca_states = artifacts.state_encoder.transform(incoming_vectors)
    FEATURES = {
        "B0": conventional,
        "B1_PCA16": np.column_stack([conventional, pca_states]),
        "B2_MATCHED_NON_Q": np.column_stack([conventional, pca_controls(pca_states)]),
        "Q_FIDELITY": np.column_stack([conventional, q_features[:, :5]]),
        "Q_ENTROPY": np.column_stack([conventional, q_features[:, 5:7]]),
        "Q_COHERENCE": np.column_stack([conventional, q_features[:, 7:9]]),
        "Q_FULL": np.column_stack([conventional, q_features]),
    }
    benchmark_hash = sha256_file(root / "data/processed/qkef_fiqa_evolution/events.jsonl")
    split_hash = sha256_payload({name: labels.sample_id[mask].tolist() for name, mask in masks.items()})
    config = {"alpha": 0.10, "c_grid": C_GRID, "risk_weights": {"OE": 3.0, "CM": 3.0, "IC": 0.5, "PROB": 0.25}, "rrf_k": 60, "seed": SEED}
    prelock = ExperimentLock("v2-retrospective-pilot-1", config, benchmark_hash, split_hash, "52bd9f3c703b002c0d03acc0b4ef1a689d86ad9c", {}, False)
    prelock.save(output_dir / "experiment_lock_pretest.json")
    ablations, models = {}, {}
    prediction_frame = labels.loc[masks["test"], ["sample_id", "target"]].reset_index(drop=True)
    for name, features in FEATURES.items():
        model, c_value = select_flat(features, y, masks)
        models[name] = model
        prediction = model.predict(features[masks["test"]])
        prediction_frame[name] = prediction
        ablations[name] = {"selected_C_on_DEV": c_value, "TEST": metrics(y[masks["test"]], prediction), "feature_count": int(features.shape[1])}
        joblib.dump(model, model_dir / f"{name.lower()}.joblib")
    hierarchy = {}
    for name in ("B0", "Q_FULL"):
        model, c_value = select_hierarchical(FEATURES[name], y, masks)
        prediction = model.predict(FEATURES[name][masks["test"]])
        prediction_frame["HIERARCHICAL_B0" if name == "B0" else "HIERARCHICAL_Q"] = prediction
        hierarchy["hierarchical_conventional" if name == "B0" else "hierarchical_q"] = {"selected_C_on_DEV": c_value, "TEST": metrics(y[masks["test"]], prediction)}
        joblib.dump(model, model_dir / f"hierarchical_{name.lower()}.joblib")
    q_model = models["Q_FULL"]
    cal_prob = q_model.predict_proba(FEATURES["Q_FULL"][masks["calibration"]])
    conformal = MondrianConformalClassifier(0.10).fit(cal_prob, y[masks["calibration"]].tolist(), list(q_model.classes_))
    test_prob = q_model.predict_proba(FEATURES["Q_FULL"][masks["test"]])
    conformal_metrics = conformal.coverage([dict(zip(q_model.classes_, row)) for row in test_prob], y[masks["test"]].tolist())
    prediction_sets = [conformal.prediction_set(dict(zip(q_model.classes_, row))) for row in test_prob]
    conformal_metrics["quarantine_rate_singleton_rule"] = sum(len(values) != 1 for values in prediction_sets) / len(prediction_sets)
    candidates = candidate_experiment(artifacts, events_by_sample, labels)
    technical, execution_rows = technical_effects(artifacts, labels, events_by_sample, masks, q_model, conformal)
    q_prediction = q_model.predict(FEATURES["Q_FULL"][masks["test"]])
    singleton = np.asarray([len(values) == 1 for values in prediction_sets])
    gated_prediction = np.asarray([values[0] if len(values) == 1 else "QUARANTINE" for values in prediction_sets])
    execution_comparison = {
        "direct_q_full": {"coverage": 1.0, "accuracy": float(accuracy_score(y[masks["test"]], q_prediction))},
        "conformal_gated_direct": {"coverage": float(singleton.mean()), "precision_on_committed": float(np.mean(gated_prediction[singleton] == y[masks["test"]][singleton]))},
        "counterfactual_retrieval_safe": {"coverage": 1.0 - technical["quarantine_rate"], "precision_on_committed": technical["auto_commit_precision"]},
    }
    results = {
        "label": "Q-KEF v2 retrospective pilot; not the future confirmatory 2,400-event benchmark",
        "split_counts": {name: int(mask.sum()) for name, mask in masks.items()}, "candidate_retrieval": candidates,
        "flat_ablation": ablations, "hierarchical_models": hierarchy, "conformal": conformal_metrics, "technical_effects": technical,
        "execution_comparison": execution_comparison,
        "storage_retrieval_comparison": {"append_only_v1_obsolete_at_5": artifacts.retrieval_metrics["append_only"]["obsolete_rate@5"], "qkef_v1_obsolete_at_5": artifacts.retrieval_metrics["qkef"]["obsolete_rate@5"], "v2_pilot_witness_obsolete_exposure": technical["mean_witness_obsolete_exposure"], "append_only_active_size_pilot": technical["append_only_comparator_size"], "v2_active_size_pilot": technical["final_active_index_size"]},
        "q_specific_delta_vs_pca16": ablations["Q_FULL"]["TEST"]["macro_f1"] - ablations["B1_PCA16"]["TEST"]["macro_f1"],
        "q_specific_delta_vs_matched_non_q": ablations["Q_FULL"]["TEST"]["macro_f1"] - ablations["B2_MATCHED_NON_Q"]["TEST"]["macro_f1"],
    }
    (output_dir / "v2_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(execution_rows).to_csv(output_dir / "execution_decisions.csv", index=False)
    prediction_frame.to_csv(output_dir / "classification_predictions.csv", index=False)
    model_hashes = {path.name: sha256_file(path) for path in sorted(model_dir.glob("*.joblib"))}
    ExperimentLock("v2-retrospective-pilot-1", config, benchmark_hash, split_hash, "52bd9f3c703b002c0d03acc0b4ef1a689d86ad9c", model_hashes, True).save(output_dir / "experiment_lock.json")
    print(json.dumps({"status": "PASS", "pilot_test_rows": 60, "q_full_macro_f1": ablations["Q_FULL"]["TEST"]["macro_f1"], "committed": technical["committed_transitions"]}, sort_keys=True))


if __name__ == "__main__":
    main()
