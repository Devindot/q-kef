"""Shared leakage-controlled utilities for the Q-KEF v2 confirmatory run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

from qkef.datasets.evolution import load_jsonl_models
from qkef.evolution.features import conventional_features, quantum_features
from qkef.evolution.models import CLASS_ORDER, make_model
from qkef.ingestion.normalize import normalize_model_text
from qkef.ingestion.sanitize import sanitize_model_text
from qkef.quantum_inspired.state_encoder import QuantumInspiredStateEncoder
from qkef.schemas import EvolutionEvent, KnowledgeUnit
from qkef.v2.hierarchy import HierarchicalLifecycleModel
from qkef.v2.retrieval import CandidateConfiguration, HybridCandidateResolver


MODEL_NAMES = ("B0", "B1_PCA16", "B2_MATCHED_NON_Q", "Q_FIDELITY", "Q_ENTROPY", "Q_COHERENCE", "Q_FULL")


@dataclass(frozen=True)
class ConfirmatoryUnit:
    identifier: str
    text: str
    split: str
    temporal_state: str
    source_ids: tuple[str, ...]
    timestamp: str


@dataclass(frozen=True)
class CandidateRow:
    event: EvolutionEvent
    incoming: ConfirmatoryUnit
    candidate_ids: tuple[str, ...]
    candidate_scores: tuple[float, ...]


def load_events(benchmark_dir: Path, splits: Iterable[str]) -> list[EvolutionEvent]:
    events = []
    for split in splits:
        events.extend(load_jsonl_models(benchmark_dir / f"{split}_events.jsonl", EvolutionEvent))
    return sorted(events, key=lambda event: event.event_id)


def load_units(
    benchmark_dir: Path,
    splits: set[str],
    normalization: dict[str, object],
    sanitization: dict[str, object],
) -> list[ConfirmatoryUnit]:
    output = []
    for filename, temporal_state in (("t0_knowledge.jsonl", "T0"), ("t1_incoming.jsonl", "T1")):
        for unit in load_jsonl_models(benchmark_dir / filename, KnowledgeUnit):
            split = str(unit.metadata.get("benchmark_split", ""))
            if split not in splits:
                continue
            sanitized = sanitize_model_text(unit.text, unit.metadata.get("text_origin"), sanitization)
            model_text, _ = normalize_model_text(sanitized.text, normalization)
            source_ids = tuple(str(value) for value in unit.metadata.get("source_document_ids", [unit.source_document_id]))
            output.append(ConfirmatoryUnit(unit.knowledge_id, model_text, split, temporal_state, source_ids, unit.timestamp.isoformat()))
    return sorted(output, key=lambda unit: unit.identifier)


def candidate_rows(
    events: list[EvolutionEvent],
    units: list[ConfirmatoryUnit],
    vectors: np.ndarray,
    vector_index: dict[str, int],
    configuration: CandidateConfiguration,
    top_k: int,
) -> list[CandidateRow]:
    unit_by_id = {unit.identifier: unit for unit in units}
    resolvers = {}
    for split in sorted({event.benchmark_split.value for event in events}):
        t0 = [unit for unit in units if unit.split == split and unit.temporal_state == "T0"]
        resolvers[split] = HybridCandidateResolver(
            {unit.identifier: unit.text for unit in t0},
            {unit.identifier: tuple(vectors[vector_index[unit.identifier]]) for unit in t0},
            rrf_k=configuration.rrf_k,
            dense_weight=configuration.dense_weight,
            lexical_weight=configuration.lexical_weight,
        )
    rows = []
    for event in events:
        incoming = unit_by_id[event.incoming_knowledge_ids[0]]
        results = resolvers[event.benchmark_split.value].resolve(
            incoming.text,
            tuple(vectors[vector_index[incoming.identifier]]),
            top_k,
            mode=configuration.mode,
        )
        if configuration.mode == "dense":
            scores = tuple(float(result.dense_score) for result in results)
        elif configuration.mode == "lexical":
            scores = tuple(float(result.lexical_score) for result in results)
        else:
            scores = tuple(result.fused_score for result in results)
        rows.append(CandidateRow(event, incoming, tuple(result.identifier for result in results), scores))
    return rows


def pca_controls(states: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(states, [0.25, 0.5, 0.75], axis=1).T
    crossings = np.sum(np.signbit(states[:, 1:]) != np.signbit(states[:, :-1]), axis=1)
    return np.column_stack([states.mean(axis=1), states.std(axis=1), states.min(axis=1), states.max(axis=1), np.abs(states).sum(axis=1), np.linalg.norm(states, axis=1), quantiles, crossings])


def feature_matrices(
    rows: list[CandidateRow],
    units: list[ConfirmatoryUnit],
    vectors: np.ndarray,
    vector_index: dict[str, int],
    state_encoder: QuantumInspiredStateEncoder,
) -> dict[str, np.ndarray]:
    unit_by_id = {unit.identifier: unit for unit in units}
    all_ids = [unit.identifier for unit in units]
    states = state_encoder.transform(np.vstack([vectors[vector_index[identifier]] for identifier in all_ids]))
    state_lookup = {identifier: states[index] for index, identifier in enumerate(all_ids)}
    conventional, quantum, incoming_states = [], [], []
    for row in rows:
        candidate_texts = [unit_by_id[identifier].text for identifier in row.candidate_ids]
        conventional.append(conventional_features(row.incoming.text, candidate_texts, row.candidate_scores))
        incoming_state = state_lookup[row.incoming.identifier]
        incoming_states.append(incoming_state)
        quantum.append(quantum_features(incoming_state, [state_lookup[identifier] for identifier in row.candidate_ids]))
    conventional_array = np.vstack(conventional)
    quantum_array = np.vstack(quantum)
    incoming_array = np.vstack(incoming_states)
    return {
        "B0": conventional_array,
        "B1_PCA16": np.column_stack([conventional_array, incoming_array]),
        "B2_MATCHED_NON_Q": np.column_stack([conventional_array, pca_controls(incoming_array)]),
        "Q_FIDELITY": np.column_stack([conventional_array, quantum_array[:, :5]]),
        "Q_ENTROPY": np.column_stack([conventional_array, quantum_array[:, 5:7]]),
        "Q_COHERENCE": np.column_stack([conventional_array, quantum_array[:, 7:9]]),
        "Q_FULL": np.column_stack([conventional_array, quantum_array]),
    }


def fit_state_encoder(units: list[ConfirmatoryUnit], vectors: np.ndarray, vector_index: dict[str, int], dimension: int, seed: int) -> QuantumInspiredStateEncoder:
    train_ids = [unit.identifier for unit in units if unit.split == "train"]
    return QuantumInspiredStateEncoder(dimension, seed).fit(np.vstack([vectors[vector_index[identifier]] for identifier in train_ids]), train_ids)


def select_flat(features: np.ndarray, labels: np.ndarray, train: np.ndarray, dev: np.ndarray, c_grid: list[float], seed: int):
    choices = []
    for c_value in c_grid:
        model = make_model(c_value, seed).fit(features[train], labels[train])
        prediction = model.predict(features[dev])
        score = f1_score(labels[dev], prediction, labels=CLASS_ORDER, average="macro", zero_division=0)
        choices.append((score, -c_value, c_value))
    selected_c = max(choices)[2]
    model = make_model(selected_c, seed).fit(features[train | dev], labels[train | dev])
    return model, selected_c


def select_hierarchical(features: np.ndarray, labels: np.ndarray, train: np.ndarray, dev: np.ndarray, c_grid: list[float], seed: int):
    choices = []
    for c_value in c_grid:
        model = HierarchicalLifecycleModel(c_value, seed).fit(features[train], labels[train])
        prediction = model.predict(features[dev])
        score = f1_score(labels[dev], prediction, labels=CLASS_ORDER, average="macro", zero_division=0)
        choices.append((score, -c_value, c_value))
    selected_c = max(choices)[2]
    model = HierarchicalLifecycleModel(selected_c, seed).fit(features[train | dev], labels[train | dev])
    return model, selected_c


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    report = classification_report(truth, prediction, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "per_class": {name: {key: float(report[name][key]) for key in ("precision", "recall", "f1-score", "support")} for name in CLASS_ORDER},
    }


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
