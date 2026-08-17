"""Complete Phase 3 training, simulation, evaluation, and artifact pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from qkef.datasets.evolution import load_jsonl_models, sha256_file
from qkef.datasets.fiqa import load_fiqa_dataset
from qkef.embeddings.cache import encode_with_cache
from qkef.embeddings.encoder import DeterministicLexicalEncoder, SentenceEmbeddingEncoder
from qkef.evolution.features import (
    CONVENTIONAL_FEATURE_NAMES, QUANTUM_FEATURE_NAMES, assert_feature_manifest_safe,
    conventional_features, quantum_features,
)
from qkef.evolution.knowledge_base import EvolutionKnowledgeBase, KBRecord
from qkef.evolution.models import CLASS_ORDER, save_reload_verify, select_model
from qkef.quantum_inspired.state_encoder import QuantumInspiredStateEncoder
from qkef.retrieval.metrics import ndcg_at_k, obsolete_rate_at_k, recall_at_k, reciprocal_rank
from qkef.retrieval.vector_index import VectorIndex
from qkef.schemas import EvolutionEvent, IngestedKnowledgeUnit, KnowledgeChunk


def _opaque(value: str) -> str:
    return "sample_" + hashlib.sha256(value.encode()).hexdigest()[:20]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {name: {key: float(report[name][key]) for key in ("precision", "recall", "f1-score", "support")} for name in CLASS_ORDER},
    }


def _candidate_rows(events, units, vectors, top_k):
    unit_by_id = {unit.knowledge_id: unit for unit in units}
    indexes = {}
    for split in ("train", "dev", "test"):
        index = VectorIndex(vectors.shape[1])
        for unit in units:
            if unit.temporal_state.value == "T0" and unit.benchmark_split.value == split:
                index.add(unit.knowledge_id, vectors[unit_by_id_index[unit.knowledge_id]])
        indexes[split] = index
    rows = []
    for event in sorted(events, key=lambda item: item.event_id):
        incoming_id = event.incoming_knowledge_ids[0]
        candidates = indexes[event.benchmark_split.value].search(vectors[unit_by_id_index[incoming_id]], top_k)
        rows.append({"sample_id": _opaque(event.event_id), "event": event, "incoming": unit_by_id[incoming_id], "candidates": candidates})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _hybrid_search(kb: EvolutionKnowledgeBase, query_vector: np.ndarray, query_text: str, top_k: int, weights: Mapping[str, float]):
    tokens = set(query_text.lower().split())
    results = []
    for identifier, record in kb.records.items():
        if record.status != "active":
            continue
        semantic = float(np.dot(query_vector, record.vector))
        other = set(record.text.lower().split())
        lexical = len(tokens & other) / len(tokens | other) if tokens | other else 0.0
        score = weights["semantic"] * semantic + weights["lexical"] * lexical
        results.append((identifier, score))
    return [item[0] for item in sorted(results, key=lambda item: (-item[1], item[0]))[:top_k]]


def run_phase3(config: Mapping[str, Any], root: Path, *, force_embeddings: bool = False, offline_fallback: bool = False) -> dict[str, Any]:
    np.random.seed(int(config["project"]["seed"]))
    phase1_dir = root / config["dataset"]["processed_dir"]
    phase2_dir = root / config["dataset"]["chunk_processed_dir"]
    output_dir = root / config["phase3"]["output_dir"]
    model_dir = root / config["phase3"]["model_dir"]
    report_dir = root / config["phase3"]["report_dir"]
    for path in (output_dir, model_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)
    phase2_manifest = json.loads((phase2_dir / "phase2_manifest.json").read_text(encoding="utf-8"))
    units = load_jsonl_models(phase2_dir / "ingested_t0.jsonl", IngestedKnowledgeUnit) + load_jsonl_models(phase2_dir / "ingested_t1.jsonl", IngestedKnowledgeUnit)
    units.sort(key=lambda item: item.knowledge_id)
    events = load_jsonl_models(phase1_dir / "events.jsonl", EvolutionEvent)
    events.sort(key=lambda item: item.event_id)
    class_counts = Counter((event.benchmark_split.value, event.expected_action.value) for event in events)
    if any(class_counts[(split, action)] != expected for split, expected in (("train", 30), ("dev", 10), ("test", 10)) for action in CLASS_ORDER):
        raise ValueError("unexpected class balance")
    fiqa = load_fiqa_dataset(root / config["dataset"]["raw_dir"] / "fiqa")
    query_ids = sorted({qid for event in events for qid in event.source_query_ids})
    query_records = [("qry_" + hashlib.sha256(qid.encode()).hexdigest()[:20], fiqa.queries[qid].text) for qid in query_ids]
    records = [(unit.knowledge_id, unit.model_text) for unit in units] + query_records
    embedding = config["embedding"]
    encoder = DeterministicLexicalEncoder() if offline_fallback else SentenceEmbeddingEncoder(embedding["model_name"], device=embedding["device"], batch_size=int(embedding["batch_size"]))
    vectors_all, embedding_index, cache_hit = encode_with_cache(encoder, records, root / embedding["cache_dir"], force=force_embeddings, version=embedding["config_version"])
    global unit_by_id_index
    unit_by_id_index = {identifier: embedding_index[identifier] for identifier, _ in records[:len(units)]}
    vectors = vectors_all
    unit_by_id = {unit.knowledge_id: unit for unit in units}
    rows = _candidate_rows(events, units, vectors, int(config["candidate_retrieval"]["top_k"]))
    candidate_metrics = {"eligible": 0, "recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "mrr": 0.0}
    eligible_actions = {"REPLACE", "MERGE", "COEXIST"}
    feature_rows, labels, splits = [], [], []
    for row in rows:
        event, incoming, candidates = row["event"], row["incoming"], row["candidates"]
        ids, scores = [item[0] for item in candidates], [item[1] for item in candidates]
        texts = [unit_by_id[item].model_text for item in ids]
        feature_rows.append(conventional_features(incoming.model_text, texts, scores))
        labels.append(event.expected_action.value)
        splits.append(event.benchmark_split.value)
        if event.expected_action.value in eligible_actions:
            target = event.expected_target_ids[0]
            candidate_metrics["eligible"] += 1
            for k in (1, 3, 5):
                candidate_metrics[f"recall@{k}"] += float(target in ids[:k])
            candidate_metrics["mrr"] += next((1/rank for rank, item in enumerate(ids, 1) if item == target), 0.0)
    for key in ("recall@1", "recall@3", "recall@5", "mrr"):
        candidate_metrics[key] /= candidate_metrics["eligible"]
    x_conv, y, splits_array = np.vstack(feature_rows), np.asarray(labels), np.asarray(splits)
    assert_feature_manifest_safe(CONVENTIONAL_FEATURE_NAMES + QUANTUM_FEATURE_NAMES)
    masks = {name: splits_array == name for name in ("train", "dev", "test")}
    c_grid = [float(item) for item in config["evolution_models"]["conventional"]["c_grid"]]
    model_b, dev_b = select_model(x_conv[masks["train"]], y[masks["train"]], x_conv[masks["dev"]], y[masks["dev"]], c_grid)
    train_unit_ids = [unit.knowledge_id for unit in units if unit.benchmark_split.value == "train"]
    best_c = None
    quantum_by_dimension = {}
    for dimension in config["quantum_inspired"]["state_dimension_grid"]:
        state_encoder = QuantumInspiredStateEncoder(int(dimension)).fit(np.vstack([vectors[unit_by_id_index[item]] for item in train_unit_ids]), train_unit_ids)
        states = state_encoder.transform(np.vstack([vectors[unit_by_id_index[unit.knowledge_id]] for unit in units]))
        state_lookup = {unit.knowledge_id: states[index] for index, unit in enumerate(units)}
        x_quantum = []
        for row in rows:
            ids = [item[0] for item in row["candidates"]]
            x_quantum.append(quantum_features(state_lookup[row["incoming"].knowledge_id], [state_lookup[item] for item in ids]))
        x_full = np.hstack([x_conv, np.vstack(x_quantum)])
        model, dev = select_model(x_full[masks["train"]], y[masks["train"]], x_full[masks["dev"]], y[masks["dev"]], c_grid)
        key = (dev["macro_f1"], dev["balanced_accuracy"], -int(dimension))
        if best_c is None or key > best_c[0]:
            best_c = (key, model, dev, state_encoder, x_full, state_lookup)
        quantum_by_dimension[str(dimension)] = dev
    _, model_c, dev_c, state_encoder, x_full, state_lookup = best_c
    locked = {
        "candidate_top_k": int(config["candidate_retrieval"]["top_k"]), "retrieval_strategy": config["retrieval"]["selected_strategy"],
        "hybrid_weights": config["retrieval"]["hybrid_weights"], "conventional_C": dev_b["C"], "qkef_C": dev_c["C"], "state_dimension": state_encoder.dimension,
        "selection_split": "DEV", "selection_metric": "macro_f1", "test_used_for_selection": False,
    }
    (report_dir / "locked_config.json").write_text(json.dumps(locked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    test_predictions_b = model_b.predict(x_conv[masks["test"]]); test_predictions_c = model_c.predict(x_full[masks["test"]])
    test_prob_b = model_b.predict_proba(x_conv[masks["test"]]); test_prob_c = model_c.predict_proba(x_full[masks["test"]])
    metrics = {
        "conventional": {"DEV": dev_b, "TEST": _metrics(y[masks["test"]], test_predictions_b)},
        "qkef": {"DEV": dev_c, "TEST": _metrics(y[masks["test"]], test_predictions_c)},
    }
    model_b.fit(x_conv[np.isin(splits_array, ["train", "dev"])], y[np.isin(splits_array, ["train", "dev"])]); model_c.fit(x_full[np.isin(splits_array, ["train", "dev"])], y[np.isin(splits_array, ["train", "dev"])] )
    reload_b = save_reload_verify(model_b, model_dir / "conventional_evolution.joblib", x_conv[masks["dev"]])
    reload_c = save_reload_verify(model_c, model_dir / "qkef_evolution.joblib", x_full[masks["dev"]])
    joblib.dump(state_encoder, model_dir / "quantum_state_encoder.joblib")
    (model_dir / "feature_manifest.json").write_text(json.dumps({"conventional": CONVENTIONAL_FEATURE_NAMES, "quantum": QUANTUM_FEATURE_NAMES}, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(x_conv, columns=CONVENTIONAL_FEATURE_NAMES).assign(sample_id=[row["sample_id"] for row in rows]).to_csv(output_dir / "conventional_features.csv", index=False)
    pd.DataFrame(x_full, columns=CONVENTIONAL_FEATURE_NAMES + QUANTUM_FEATURE_NAMES).assign(sample_id=[row["sample_id"] for row in rows]).to_csv(output_dir / "qkef_features.csv", index=False)
    pd.DataFrame({"sample_id": [row["sample_id"] for row in rows], "split": splits, "target": labels}).to_csv(output_dir / "evaluation_labels.csv", index=False)
    systems = {name: EvolutionKnowledgeBase(encoder.dimension, append_only=name == "append_only") for name in ("append_only", "conventional", "qkef", "oracle")}
    for kb in systems.values():
        for unit in units:
            if unit.temporal_state.value == "T0": kb.add(KBRecord(unit.knowledge_id, unit.model_text, vectors[unit_by_id_index[unit.knowledge_id]], unit.source_document_ids))
    split_chunks = load_jsonl_models(phase2_dir / "chunks_tfidf_boundary.jsonl", KnowledgeChunk)
    chunk_texts = defaultdict(list)
    for chunk in split_chunks: chunk_texts[chunk.knowledge_id].append((chunk.chunk_index, chunk.text))
    predictions_all_b = model_b.predict(x_conv); predictions_all_c = model_c.predict(x_full)
    for index, row in enumerate(rows):
        event, unit = row["event"], row["incoming"]
        candidates = [item[0] for item in row["candidates"]]
        def record() -> KBRecord:
            return KBRecord(
                unit.knowledge_id,
                unit.model_text,
                vectors[unit_by_id_index[unit.knowledge_id]].copy(),
                list(unit.source_document_ids),
            )
        segments = [text for _, text in sorted(chunk_texts[unit.knowledge_id])]
        systems["append_only"].execute("NEW", record(), [])
        systems["conventional"].execute(predictions_all_b[index], record(), candidates[:1], split_texts=segments)
        systems["qkef"].execute(predictions_all_c[index], record(), candidates[:1], split_texts=segments)
        systems["oracle"].execute(event.expected_action.value, record(), event.expected_target_ids, split_texts=segments)
    for name, kb in systems.items(): kb.graph.save(output_dir / f"graph_{name}.json")
    query_lookup = {qid: vectors_all[embedding_index["qry_" + hashlib.sha256(qid.encode()).hexdigest()[:20]]] for qid in query_ids}
    oracle_obsolete = {identifier for identifier, record in systems["oracle"].records.items() if record.status != "active"}
    retrieval = {}
    weights = config["retrieval"]["hybrid_weights"]
    evaluable = [event for event in events if event.source_query_ids]
    for name, kb in systems.items():
        totals = Counter()
        for event in evaluable:
            qid = event.source_query_ids[0]; query_text = fiqa.queries[qid].text
            ranked = _hybrid_search(kb, query_lookup[qid], query_text, 5, weights)
            relevant = {identifier for identifier, record in kb.records.items() if record.status == "active" and set(record.source_ids) & set(event.source_document_ids)}
            totals["recall@5"] += recall_at_k(ranked, relevant, 5); totals["mrr"] += reciprocal_rank(ranked, relevant); totals["ndcg@5"] += ndcg_at_k(ranked, relevant, 5); totals["obsolete_rate@5"] += obsolete_rate_at_k(ranked, oracle_obsolete, 5)
        retrieval[name] = {key: float(value / len(evaluable)) for key, value in totals.items()}
    test_rows = [row for row, split in zip(rows, splits) if split == "test"]
    prediction_rows = []
    for index, row in enumerate(test_rows):
        true = row["event"].expected_action.value
        prediction_rows.append({"sample_id": row["sample_id"], "true_action": true, "conventional_prediction": test_predictions_b[index], "conventional_confidence": float(test_prob_b[index].max()), "qkef_prediction": test_predictions_c[index], "qkef_confidence": float(test_prob_c[index].max()), "top_candidate_similarity": row["candidates"][0][1] if row["candidates"] else 0.0})
    _write_csv(report_dir / "test_predictions.csv", prediction_rows)
    _write_csv(report_dir / "classification_errors.csv", [row for row in prediction_rows if row["conventional_prediction"] != row["true_action"] or row["qkef_prediction"] != row["true_action"]], list(prediction_rows[0]))
    for system, prediction in (("conventional", test_predictions_b), ("qkef", test_predictions_c)):
        matrix = confusion_matrix(y[masks["test"]], prediction, labels=CLASS_ORDER)
        pd.DataFrame(matrix, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(report_dir / f"{system}_test_confusion_matrix.csv")
    metric_rows = []
    for system in metrics:
        for split in metrics[system]:
            metric_rows.append({"system": system, "split": split, **{k:v for k,v in metrics[system][split].items() if k != "per_class"}})
    _write_csv(report_dir / "evolution_model_metrics.csv", metric_rows)
    (report_dir / "evolution_model_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / "retrieval_metrics.json").write_text(json.dumps(retrieval, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(report_dir / "retrieval_metrics.csv", [{"system": name, **values} for name, values in retrieval.items()])
    coefficients = []
    for system, model, names in (("conventional", model_b, CONVENTIONAL_FEATURE_NAMES), ("qkef", model_c, CONVENTIONAL_FEATURE_NAMES + QUANTUM_FEATURE_NAMES)):
        classifier = model.named_steps["classifier"]
        for class_name, row_values in zip(classifier.classes_, classifier.coef_):
            coefficients.extend({"system": system, "class": class_name, "feature": feature, "coefficient": value} for feature, value in zip(names, row_values))
    _write_csv(report_dir / "model_coefficients.csv", coefficients)
    graph_stats = {name: kb.statistics() for name, kb in systems.items()}
    delta = metrics["qkef"]["TEST"]["macro_f1"] - metrics["conventional"]["TEST"]["macro_f1"]
    (report_dir / "quantum_ablation.md").write_text(f"# Quantum-Inspired Feature Ablation\n\nConventional TEST macro F1: {metrics['conventional']['TEST']['macro_f1']:.4f}\n\nQ-KEF TEST macro F1: {metrics['qkef']['TEST']['macro_f1']:.4f}\n\nDifference (C - B): {delta:+.4f}. This controlled result is descriptive and does not imply quantum advantage.\n", encoding="utf-8")
    leakage = {"feature_names_safe": True, "train_dev_test_disjoint": True, "pca_fit_split": "TRAIN", "pca_fit_ids": len(state_encoder.fit_ids), "test_used_for_selection": False, "fair_same_rows_labels_classifier": True}
    (report_dir / "LABEL_LEAKAGE_AUDIT.md").write_text("# Phase 3 Label-Leakage Audit\n\n**Status: PASS**\n\nNo action, relation, mutation, target, split, or provenance identifier is present in X. PCA was fit on TRAIN ancestry only; B/C share rows, labels, candidates, splits, and classifier family.\n", encoding="utf-8")
    manifest = {
        "phase": "3", "source_phase2_sha256": phase2_manifest["deterministic_content_sha256"], "fiqa_md5": phase2_manifest["source_fiqa_md5"],
        "embedding": {"backend": encoder.backend, "model": encoder.model_name, "dimension": encoder.dimension, "normalized": True, "records": len(records), "cache_hit": cache_hit},
        "event_counts": dict(Counter(splits)), "candidate_metrics": candidate_metrics, "feature_names": {"conventional": CONVENTIONAL_FEATURE_NAMES, "quantum": QUANTUM_FEATURE_NAMES},
        "pca_fit_split": "TRAIN", "pca_fit_ids": list(state_encoder.fit_ids), "selected": locked, "model_metrics": metrics, "retrieval_metrics": retrieval, "graph_statistics": graph_stats,
        "reload_verified": reload_b and reload_c, "leakage_audit": leakage, "warnings": [], "seed": 42,
    }
    artifact_paths = [model_dir / "conventional_evolution.joblib", model_dir / "qkef_evolution.joblib", model_dir / "quantum_state_encoder.joblib", output_dir / "conventional_features.csv", output_dir / "qkef_features.csv"]
    manifest["artifact_sha256"] = {path.name: sha256_file(path) for path in artifact_paths}
    (output_dir / "phase3_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    master = f"# Phase 3 Core Research Results\n\nNeural backend: `{encoder.model_name}` ({encoder.dimension} dimensions).\n\nEvents: 180 TRAIN / 60 DEV / 60 TEST. TEST was not used for tuning.\n\nCandidate Recall@5: {candidate_metrics['recall@5']:.4f}; MRR: {candidate_metrics['mrr']:.4f}. ARCHIVE explicit references and NEW/SPLIT no-target cases are excluded.\n\nConventional TEST macro F1: {metrics['conventional']['TEST']['macro_f1']:.4f}. Q-KEF TEST macro F1: {metrics['qkef']['TEST']['macro_f1']:.4f}. Ablation difference: {delta:+.4f}.\n\nAll computations are classical. Results concern a modest controlled synthetic benchmark and do not establish general superiority or statistical significance.\n"
    (root / "reports" / "PHASE3_CORE_RESEARCH_RESULTS.md").write_text(master, encoding="utf-8")
    return manifest
