"""Atomic deterministic embedding cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def cache_key(model_name: str, text_sha256: str, version: str = "1.0") -> str:
    return hashlib.sha256(f"{model_name}|{text_sha256}|{version}".encode()).hexdigest()


def encode_with_cache(
    encoder: object,
    records: Sequence[tuple[str, str]],
    cache_dir: Path,
    *,
    force: bool = False,
    version: str = "1.0",
) -> tuple[np.ndarray, dict[str, int], bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path, index_path, manifest_path = (
        cache_dir / "embeddings.npy",
        cache_dir / "embedding_index.json",
        cache_dir / "embedding_manifest.json",
    )
    hashes = [hashlib.sha256(text.encode()).hexdigest() for _, text in records]
    keys = [cache_key(encoder.model_name, value, version) for value in hashes]
    expected = {identifier: index for index, (identifier, _) in enumerate(records)}
    if not force and all(path.exists() for path in (array_path, index_path, manifest_path)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if manifest.get("keys") == keys and index == expected:
            vectors = np.load(array_path)
            validate_embeddings(vectors, len(records), int(manifest["dimension"]))
            return vectors, index, True
    unique: dict[str, str] = {}
    for key, (_, text) in zip(keys, records):
        unique.setdefault(key, text)
    unique_keys = list(unique)
    unique_vectors = encoder.encode([unique[key] for key in unique_keys])
    lookup = {key: unique_vectors[index] for index, key in enumerate(unique_keys)}
    vectors = np.stack([lookup[key] for key in keys]).astype(np.float32)
    validate_embeddings(vectors, len(records), encoder.dimension)
    temp = array_path.with_suffix(".tmp.npy")
    np.save(temp, vectors)
    temp.replace(array_path)
    index_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "backend": encoder.backend,
        "model_name": encoder.model_name,
        "dimension": encoder.dimension,
        "normalized": True,
        "keys": keys,
        "unique_text_count": len(unique),
        "record_count": len(records),
        "version": version,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return vectors, expected, False


def validate_embeddings(vectors: np.ndarray, count: int, dimension: int) -> None:
    if vectors.shape != (count, dimension):
        raise ValueError("embedding cache shape mismatch")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("embedding cache contains NaN/Inf")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0) or not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("embedding cache vectors are not normalized")
