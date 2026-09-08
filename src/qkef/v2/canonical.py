"""Canonical serialization and hashing for deterministic v2 evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any


def canonical_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [canonical_value(item) for item in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True)) if isinstance(value, (set, frozenset)) else values
    if hasattr(value, "tolist"):
        return canonical_value(value.tolist())
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(canonical_value(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
