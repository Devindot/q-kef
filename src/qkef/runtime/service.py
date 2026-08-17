"""End-to-end, label-free Conventional/Q-KEF inference service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qkef.embeddings.encoder import SentenceEmbeddingEncoder
from qkef.evolution.features import CONVENTIONAL_FEATURE_NAMES, QUANTUM_FEATURE_NAMES, conventional_features, quantum_features
from qkef.ingestion.normalize import normalize_model_text
from qkef.quantum_inspired.state_encoder import coherence_like_score, fidelity_like_similarity, state_entropy
from qkef.retrieval.vector_index import VectorIndex
from qkef.runtime.artifacts import FinalArtifactLoader, LoadedArtifacts
from qkef.runtime.schemas import AnalysisResult, CandidateMatch, CLASS_ORDER


PREVIEWS = {
    "NEW": "Incoming knowledge would be added as ACTIVE.",
    "REPLACE": "Selected predecessor would become SUPERSEDED; incoming unit becomes ACTIVE.",
    "MERGE": "A deterministic merged active representation would be created while preserving both provenance branches.",
    "ARCHIVE": "Selected target would become ARCHIVED; audit history is retained.",
    "COEXIST": "Existing and incoming knowledge remain ACTIVE and are linked.",
    "SPLIT": "Incoming knowledge is segmented into independently active children when at least two usable segments are available.",
}


class QKEFService:
    def __init__(self, artifacts: LoadedArtifacts, encoder: Any):
        self.artifacts = artifacts
        self.encoder = encoder
        self.units = {unit.knowledge_id: unit for unit in artifacts.units}
        self.index = VectorIndex(artifacts.manifest["embedding"]["dimension"])
        for unit in sorted(artifacts.units, key=lambda value: value.knowledge_id):
            if unit.temporal_state.value == "T0":
                self.index.add(unit.knowledge_id, artifacts.embedding_vectors[artifacts.embedding_index[unit.knowledge_id]])

    @classmethod
    def load_default(cls, *, local_files_only: bool = True) -> "QKEFService":
        artifacts = FinalArtifactLoader.default().load()
        embedding = artifacts.config["embedding"]
        encoder = SentenceEmbeddingEncoder(embedding["model_name"], device="cpu", batch_size=int(embedding["batch_size"]), local_files_only=local_files_only)
        return cls(artifacts, encoder)

    def analyze_incoming_knowledge(self, text: str, system: str = "qkef", *, confidence_threshold: float = 0.60) -> AnalysisResult:
        if system not in {"conventional", "qkef"}:
            raise ValueError("system must be 'conventional' or 'qkef'")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("incoming text cannot be empty")
        if len(text) > 20_000:
            raise ValueError("incoming text exceeds the 20,000 character limit")
        normalized, _ = normalize_model_text(text, self.artifacts.config["ingestion"]["normalization"])
        vector = np.asarray(self.encoder.encode([normalized])[0], dtype=np.float32)
        raw_candidates = self.index.search(vector, int(self.artifacts.locked_config["candidate_top_k"]))
        candidate_ids = [identifier for identifier, _ in raw_candidates]
        candidate_scores = [float(score) for _, score in raw_candidates]
        candidate_texts = [self.units[identifier].model_text for identifier in candidate_ids]
        conventional = conventional_features(normalized, candidate_texts, candidate_scores)
        state = self.artifacts.state_encoder.transform(vector.reshape(1, -1))[0]
        candidate_states = self.artifacts.state_encoder.transform(np.vstack([
            self.artifacts.embedding_vectors[self.artifacts.embedding_index[identifier]] for identifier in candidate_ids
        ])) if candidate_ids else np.empty((0, len(state)))
        quantum = quantum_features(state, list(candidate_states))
        feature_vector = conventional if system == "conventional" else np.r_[conventional, quantum]
        model = self.artifacts.conventional_model if system == "conventional" else self.artifacts.qkef_model
        probabilities_raw = model.predict_proba(feature_vector.reshape(1, -1))[0]
        probability_by_class = {name: float(value) for name, value in zip(model.classes_, probabilities_raw)}
        probabilities = {name: probability_by_class.get(name, 0.0) for name in CLASS_ORDER}
        action = str(model.predict(feature_vector.reshape(1, -1))[0])
        confidence = max(probabilities.values())
        target = candidate_ids[0] if candidate_ids and action in {"REPLACE", "MERGE", "ARCHIVE", "COEXIST"} else None
        warnings = []
        if confidence < confidence_threshold:
            warnings.append("Low-confidence model prediction — human review recommended.")
        if action == "SPLIT" and len([part for part in normalized.split("\n\n") if part.strip()]) < 2:
            warnings.append("Automatic SPLIT execution produced fewer than two usable segments. The system preserved provenance and avoided fabricating artificial child knowledge.")
        matches = []
        for position, (identifier, similarity) in enumerate(raw_candidates):
            candidate_state = candidate_states[position]
            matches.append(CandidateMatch(
                identifier=identifier, excerpt=self.units[identifier].model_text[:400], semantic_similarity=float(similarity),
                source_document_ids=list(self.units[identifier].source_document_ids),
                fidelity_like_similarity=fidelity_like_similarity(state, candidate_state) if system == "qkef" else None,
            ))
        names = CONVENTIONAL_FEATURE_NAMES if system == "conventional" else CONVENTIONAL_FEATURE_NAMES + QUANTUM_FEATURE_NAMES
        return AnalysisResult(
            system=system, predicted_action=action, confidence=confidence, class_probabilities=probabilities,
            candidates=matches, selected_target=target, semantic_similarity=candidate_scores[0] if candidate_scores else None,
            fidelity_like_similarity=fidelity_like_similarity(state, candidate_states[0]) if system == "qkef" and len(candidate_states) else None,
            state_entropy=state_entropy(state) if system == "qkef" else None,
            coherence_like_score=coherence_like_score(state) if system == "qkef" else None,
            state_amplitudes=state.astype(float).tolist() if system == "qkef" else None,
            state_probabilities=np.square(state).astype(float).tolist() if system == "qkef" else None,
            kb_effect_preview=PREVIEWS[action], warnings=warnings, model_version="phase3-1.0",
            embedding_model=self.artifacts.manifest["embedding"]["model"], quantum_features_enabled=system == "qkef",
            feature_values={name: float(value) for name, value in zip(names, feature_vector)},
        )


@dataclass
class DemoKnowledgeBase:
    records: dict[str, dict[str, Any]]

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "DemoKnowledgeBase":
        return cls({record["identifier"]: dict(record) for record in records})

    def reset(self, records: list[dict[str, Any]]) -> None:
        self.records = {record["identifier"]: dict(record) for record in records}

    def apply(self, result: AnalysisResult, incoming_text: str) -> None:
        identifier = f"demo_incoming_{len(self.records) + 1}"
        target = result.selected_target
        if result.predicted_action in {"REPLACE", "ARCHIVE"} and target in self.records:
            self.records[target]["status"] = "SUPERSEDED" if result.predicted_action == "REPLACE" else "ARCHIVED"
        self.records[identifier] = {"identifier": identifier, "text": incoming_text, "status": "ACTIVE", "provenance": [target] if target else []}
