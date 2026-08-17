"""Small deterministic exact cosine/dot-product index."""

from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict[str, object]] = {}

    def add(self, identifier: str, vector: np.ndarray, **metadata: object) -> None:
        value = np.asarray(vector, dtype=np.float32)
        if value.shape != (self.dimension,) or not np.all(np.isfinite(value)):
            raise ValueError("invalid vector shape/content")
        norm = np.linalg.norm(value)
        if norm == 0:
            raise ValueError("zero vector")
        self._vectors[identifier] = value / norm
        self._metadata[identifier] = dict(metadata)

    def remove(self, identifier: str) -> None:
        self._vectors.pop(identifier, None)
        self._metadata.pop(identifier, None)

    def search(self, query: np.ndarray, top_k: int, **filters: object) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        value = np.asarray(query, dtype=np.float32)
        if value.shape != (self.dimension,):
            raise ValueError("query dimension mismatch")
        norm = np.linalg.norm(value)
        if norm == 0:
            raise ValueError("zero query")
        value = value / norm
        results = []
        for identifier in sorted(self._vectors):
            metadata = self._metadata[identifier]
            if all(metadata.get(key) == expected for key, expected in filters.items()):
                results.append((identifier, float(np.dot(value, self._vectors[identifier]))))
        return sorted(results, key=lambda item: (-item[1], item[0]))[:top_k]

    def __len__(self) -> int:
        return len(self._vectors)
