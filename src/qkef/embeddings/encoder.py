"""Neural MiniLM encoder plus explicitly named offline test fallback."""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(~np.isfinite(vectors)) or np.any(norms == 0):
        raise ValueError("embeddings must be finite and non-zero")
    return vectors / norms


class SentenceEmbeddingEncoder:
    """Evaluation-only CPU wrapper around SentenceTransformers."""

    backend = "sentence_transformers"

    def __init__(self, model_name: str, *, device: str = "cpu", batch_size: int = 32, local_files_only: bool = False):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = int(self.model.get_embedding_dimension())
        else:  # Compatibility with SentenceTransformers releases before the rename.
            self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if any(not text.strip() for text in texts):
            raise ValueError("cannot embed empty text")
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError("embedding output shape mismatch")
        return _normalize(vectors).astype(np.float32)


class DeterministicLexicalEncoder:
    """Hashing lexical fallback for tests/smoke only; not a neural embedding."""

    backend = "deterministic_lexical_fallback"
    model_name = "deterministic-lexical-sha256-v1"

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if any(not text.strip() for text in texts):
            raise ValueError("cannot encode empty text")
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\b\w+\b", text.lower()):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign
            if not np.any(matrix[row]):
                matrix[row, 0] = 1.0
        return _normalize(matrix).astype(np.float32)
