"""Fit and lock Q-KEF v2 models using TRAIN/DEV/CAL only."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import yaml

from _phase1_common import PROJECT_ROOT
from qkef.embeddings.cache import encode_with_cache
from qkef.embeddings.encoder import SentenceEmbeddingEncoder
from qkef.evolution.models import CLASS_ORDER, make_model
from qkef.v2.canonical import sha256_file, sha256_payload
from qkef.v2.confirmatory import (
    MODEL_NAMES,
    candidate_rows,
    feature_matrices,
    fit_state_encoder,
    load_events,
    load_json,
    load_units,
    metrics,
    select_flat,
    select_hierarchical,
)
from qkef.v2.conformal import MondrianConformalClassifier
from qkef.v2.experiment_lock import ExperimentLock
from qkef.v2.retrieval import CandidateQuery, select_dev_configuration


def main() -> None:
    root = PROJECT_ROOT
    config = yaml.safe_load((root / "configs/v2_confirmatory_experiment.yaml").read_text(encoding="utf-8"))
    defaults = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    experiment = config["experiment"]
    benchmark_dir = root / experiment["benchmark_dir"]
    output_dir = root / experiment["output_dir"]
    model_dir = root / experiment["model_dir"]
    report_dir = root / experiment["report_dir"]
    for path in (output_dir, model_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    pretest_splits = ("train", "dev", "calibration")
    events = load_events(benchmark_dir, pretest_splits)
    if len(events) != 1920 or any(event.benchmark_split.value == "test" for event in events):
        raise RuntimeError("pre-TEST preparation must contain exactly 1,920 non-TEST events")
    units = load_units(
        benchmark_dir,
        set(pretest_splits),
        defaults["ingestion"]["normalization"],
        defaults["ingestion"]["sanitization"],
    )
    embedding = config["embedding"]
    encoder = SentenceEmbeddingEncoder(embedding["model_name"], device=embedding["device"], batch_size=int(embedding["batch_size"]))
    records = [(unit.identifier, unit.text) for unit in units]
    vectors, vector_index, cache_hit = encode_with_cache(
        encoder,
        records,
        root / embedding["cache_dir"] / "pretest",
        version=str(embedding["version"]),
    )

    unit_by_id = {unit.identifier: unit for unit in units}
    dev_t0 = [unit for unit in units if unit.split == "dev" and unit.temporal_state == "T0"]
    dev_documents = {unit.identifier: unit.text for unit in dev_t0}
    dev_vectors = {unit.identifier: tuple(vectors[vector_index[unit.identifier]]) for unit in dev_t0}
    eligible_actions = {"REPLACE", "MERGE", "COEXIST"}
    dev_queries = []
    for event in events:
        if event.benchmark_split.value != "dev" or event.expected_action.value not in eligible_actions:
            continue
        incoming = unit_by_id[event.incoming_knowledge_ids[0]]
        dev_queries.append(CandidateQuery(event.event_id, incoming.text, tuple(vectors[vector_index[incoming.identifier]]), frozenset(event.expected_target_ids), split="dev"))
    retrieval = config["candidate_retrieval"]
    selection = select_dev_configuration(
        dev_documents,
        dev_vectors,
        dev_queries,
        top_k=int(retrieval["top_k"]),
        hybrid_weights=tuple(tuple(map(float, pair)) for pair in retrieval["hybrid_weights"]),
        rrf_ks=tuple(map(int, retrieval["rrf_ks"])),
    )
    rows = candidate_rows(events, units, vectors, vector_index, selection.selected, int(retrieval["top_k"]))
    labels = np.asarray([row.event.expected_action.value for row in rows])
    split_values = np.asarray([row.event.benchmark_split.value for row in rows])
    masks = {split: split_values == split for split in pretest_splits}
    state_encoder = fit_state_encoder(units, vectors, vector_index, int(config["models"]["state_dimension"]), int(experiment["seed"]))
    features = feature_matrices(rows, units, vectors, vector_index, state_encoder)
    c_grid = list(map(float, config["models"]["c_grid"]))
    models = {}
    selected_c = {}
    dev_metrics = {}
    for name in MODEL_NAMES:
        model, c_value = select_flat(features[name], labels, masks["train"], masks["dev"], c_grid, int(experiment["seed"]))
        models[name] = model
        selected_c[name] = c_value
        dev_only_model = make_model(c_value, int(experiment["seed"])).fit(features[name][masks["train"]], labels[masks["train"]])
        dev_metrics[name] = metrics(labels[masks["dev"]], dev_only_model.predict(features[name][masks["dev"]]))
    hierarchical = {}
    for source in ("B0", "Q_FULL"):
        model, c_value = select_hierarchical(features[source], labels, masks["train"], masks["dev"], c_grid, int(experiment["seed"]))
        hierarchical[source] = model
        selected_c[f"HIERARCHICAL_{source}"] = c_value

    q_model = models["Q_FULL"]
    calibration_probabilities = q_model.predict_proba(features["Q_FULL"][masks["calibration"]])
    conformal = MondrianConformalClassifier(float(config["conformal"]["alpha"])).fit(
        calibration_probabilities,
        labels[masks["calibration"]].tolist(),
        list(q_model.classes_),
    )
    for name, model in models.items():
        joblib.dump(model, model_dir / f"{name.lower()}.joblib")
    for name, model in hierarchical.items():
        joblib.dump(model, model_dir / f"hierarchical_{name.lower()}.joblib")
    joblib.dump(state_encoder, model_dir / "state_encoder.joblib")
    joblib.dump(conformal, model_dir / "mondrian_conformal.joblib")
    model_hashes = {path.name: sha256_file(path) for path in sorted(model_dir.glob("*.joblib"))}
    benchmark_manifest = load_json(root / "reports/v2/confirmatory_benchmark/benchmark_manifest.json")
    split_file_hashes = {split: sha256_file(benchmark_dir / f"{split}_events.jsonl") for split in ("train", "dev", "calibration", "test")}
    locked_configuration = {
        "experiment": config,
        "candidate_selection": selection.as_dict(),
        "selected_C": selected_c,
        "class_order": CLASS_ORDER,
        "feature_shapes": {name: int(values.shape[1]) for name, values in features.items()},
        "test_labels_read_during_preparation": False,
    }
    try:
        code_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        code_commit = "unavailable"
    lock = ExperimentLock(
        str(experiment["version"]),
        locked_configuration,
        str(benchmark_manifest["deterministic_content_sha256"]),
        sha256_payload(split_file_hashes),
        code_commit,
        model_hashes,
        False,
    )
    lock.save(report_dir / "experiment_lock_pretest.json")
    summary = {
        "status": "PRETEST_LOCKED",
        "experiment_version": experiment["version"],
        "event_counts": {split: int(masks[split].sum()) for split in pretest_splits},
        "test_events_loaded": 0,
        "test_evaluation_status": "not_executed",
        "embedding": {"backend": encoder.backend, "model": encoder.model_name, "dimension": encoder.dimension, "cache_hit": cache_hit},
        "candidate_selection": selection.as_dict(),
        "selected_C": selected_c,
        "dev_metrics_train_only": dev_metrics,
        "conformal_quantiles": conformal.class_quantiles,
        "model_hashes": model_hashes,
        "split_file_hashes": split_file_hashes,
        "configuration_hash": lock.configuration_hash,
    }
    (report_dir / "pretest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PRETEST_LOCKED", "events": len(events), "selected_retrieval": selection.selected.identifier, "model_count": len(model_hashes)}, sort_keys=True))


if __name__ == "__main__":
    main()
