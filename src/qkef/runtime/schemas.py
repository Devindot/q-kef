"""Runtime-only schemas that deliberately contain no benchmark ground truth."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


CLASS_ORDER = ["NEW", "REPLACE", "MERGE", "ARCHIVE", "COEXIST", "SPLIT"]


class CandidateMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identifier: str
    excerpt: str
    semantic_similarity: float
    source_document_ids: list[str]
    fidelity_like_similarity: float | None = None


class AnalysisResult(BaseModel):
    """Prediction response; forbidden benchmark fields are intentionally absent."""

    model_config = ConfigDict(extra="forbid")
    system: Literal["conventional", "qkef"]
    predicted_action: str
    confidence: float = Field(ge=0.0, le=1.0)
    class_probabilities: dict[str, float]
    candidates: list[CandidateMatch]
    selected_target: str | None
    semantic_similarity: float | None
    fidelity_like_similarity: float | None
    state_entropy: float | None
    coherence_like_score: float | None
    state_amplitudes: list[float] | None
    state_probabilities: list[float] | None
    kb_effect_preview: str
    warnings: list[str]
    model_version: str
    embedding_model: str
    quantum_features_enabled: bool
    feature_values: dict[str, float]


class QAResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    supporting_passages: list[str]
    provenance: list[list[str]]
    retrieval_scores: list[float]
    system: str
    warnings: list[str]
