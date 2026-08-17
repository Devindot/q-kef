"""TRAIN-only PCA contextual basis and normalized real state vectors."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class QuantumInspiredStateEncoder:
    def __init__(self, dimension: int, seed: int = 42):
        self.requested_dimension = dimension
        self.seed = seed
        self.pca: PCA | None = None
        self.fit_ids: tuple[str, ...] = ()

    def fit(self, vectors: np.ndarray, fit_ids: list[str]) -> "QuantumInspiredStateEncoder":
        dimension = min(self.requested_dimension, vectors.shape[0], vectors.shape[1])
        if dimension < 1:
            raise ValueError("PCA requires non-empty training vectors")
        self.pca = PCA(n_components=dimension, svd_solver="full", random_state=self.seed)
        self.pca.fit(vectors)
        self.fit_ids = tuple(fit_ids)
        return self

    @property
    def dimension(self) -> int:
        if self.pca is None:
            raise ValueError("state encoder is not fitted")
        return int(self.pca.n_components_)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise ValueError("state encoder is not fitted")
        values = self.pca.transform(vectors)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        zero = norms[:, 0] == 0
        values[zero, 0] = 1.0
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return (values / norms).astype(np.float32)


def fidelity_like_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(float(np.dot(left, right)) ** 2, 0.0, 1.0))


def state_entropy(state: np.ndarray, epsilon: float = 1e-12) -> float:
    probabilities = np.square(state)
    probabilities = probabilities / probabilities.sum()
    return float(-np.sum(probabilities * np.log(probabilities + epsilon)))


def coherence_like_score(state: np.ndarray) -> float:
    raw = float(np.sum(np.abs(state)) ** 2 - 1.0)
    maximum = max(1.0, len(state) - 1.0)
    return float(np.clip(raw / maximum, 0.0, 1.0))


def hellinger_distance(left: np.ndarray, right: np.ndarray) -> float:
    p, q = np.square(left), np.square(right)
    p, q = p / p.sum(), q / q.sum()
    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2.0))
