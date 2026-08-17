"""Repository-trusted artifact discovery, verification, and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from qkef.datasets.evolution import load_jsonl_models, sha256_file
from qkef.graph.semantic_graph import SemanticGraph
from qkef.schemas import IngestedKnowledgeUnit


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedArtifacts:
    root: Path
    config: dict[str, Any]
    manifest: dict[str, Any]
    locked_config: dict[str, Any]
    feature_manifest: dict[str, list[str]]
    conventional_model: Any
    qkef_model: Any
    state_encoder: Any
    graphs: dict[str, SemanticGraph]
    model_metrics: dict[str, Any]
    retrieval_metrics: dict[str, Any]
    units: tuple[IngestedKnowledgeUnit, ...]
    embedding_vectors: np.ndarray
    embedding_index: dict[str, int]


class FinalArtifactLoader:
    """Loads only fixed, repository-relative files; never user-provided pickles."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    @classmethod
    def default(cls) -> "FinalArtifactLoader":
        return cls(Path(__file__).resolve().parents[3])

    def _required(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ArtifactError("artifact path escaped repository root")
        if not path.is_file():
            raise ArtifactError(f"required trusted artifact is missing: {relative}")
        return path

    def load(self) -> LoadedArtifacts:
        config = yaml.safe_load(self._required("configs/default.yaml").read_text(encoding="utf-8"))
        manifest = json.loads(self._required("data/processed/qkef_phase3/phase3_manifest.json").read_text(encoding="utf-8"))
        locked = json.loads(self._required("reports/phase3/locked_config.json").read_text(encoding="utf-8"))
        features = json.loads(self._required("models/phase3/feature_manifest.json").read_text(encoding="utf-8"))
        locations = {
            "conventional_evolution.joblib": "models/phase3/conventional_evolution.joblib",
            "qkef_evolution.joblib": "models/phase3/qkef_evolution.joblib",
            "quantum_state_encoder.joblib": "models/phase3/quantum_state_encoder.joblib",
            "conventional_features.csv": "data/processed/qkef_phase3/conventional_features.csv",
            "qkef_features.csv": "data/processed/qkef_phase3/qkef_features.csv",
        }
        for name, expected in manifest["artifact_sha256"].items():
            path = self._required(locations[name])
            if sha256_file(path) != expected:
                raise ArtifactError(f"artifact hash mismatch: {name}")
        if locked.get("test_used_for_selection") is not False:
            raise ArtifactError("locked configuration violates TEST freeze")
        if manifest["embedding"]["model"] != config["embedding"]["model_name"]:
            raise ArtifactError("embedding metadata mismatch")
        cache_dir = self.root / config["embedding"]["cache_dir"]
        vectors = np.load(self._required(str((cache_dir / "embeddings.npy").relative_to(self.root))))
        index = json.loads(self._required(str((cache_dir / "embedding_index.json").relative_to(self.root))).read_text(encoding="utf-8"))
        if vectors.shape[1] != manifest["embedding"]["dimension"] or len(index) != vectors.shape[0]:
            raise ArtifactError("embedding cache dimensions are inconsistent")
        units = tuple(
            load_jsonl_models(self._required("data/processed/qkef_fiqa_chunks/ingested_t0.jsonl"), IngestedKnowledgeUnit)
            + load_jsonl_models(self._required("data/processed/qkef_fiqa_chunks/ingested_t1.jsonl"), IngestedKnowledgeUnit)
        )
        graphs = {name: SemanticGraph.load(self._required(f"data/processed/qkef_phase3/graph_{name}.json")) for name in ("append_only", "conventional", "qkef", "oracle")}
        return LoadedArtifacts(
            root=self.root, config=config, manifest=manifest, locked_config=locked, feature_manifest=features,
            conventional_model=joblib.load(self._required("models/phase3/conventional_evolution.joblib")),
            qkef_model=joblib.load(self._required("models/phase3/qkef_evolution.joblib")),
            state_encoder=joblib.load(self._required("models/phase3/quantum_state_encoder.joblib")),
            graphs=graphs,
            model_metrics=json.loads(self._required("reports/phase3/evolution_model_metrics.json").read_text(encoding="utf-8")),
            retrieval_metrics=json.loads(self._required("reports/phase3/retrieval_metrics.json").read_text(encoding="utf-8")),
            units=units, embedding_vectors=vectors, embedding_index={str(k): int(v) for k, v in index.items()},
        )
