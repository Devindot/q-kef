"""Immutable experiment configuration lock and TEST reuse guard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qkef.v2.canonical import canonical_value, sha256_payload


@dataclass(frozen=True)
class ExperimentLock:
    experiment_version: str
    configuration: dict[str, Any]
    benchmark_hash: str
    split_hash: str
    code_commit: str
    model_hashes: dict[str, str]
    test_executed: bool = False

    @property
    def configuration_hash(self) -> str:
        return sha256_payload({"configuration": self.configuration, "benchmark_hash": self.benchmark_hash, "split_hash": self.split_hash, "model_hashes": self.model_hashes})

    def validate_reuse(self, configuration: dict[str, Any], *, experiment_version: str) -> None:
        proposed = sha256_payload({"configuration": configuration, "benchmark_hash": self.benchmark_hash, "split_hash": self.split_hash, "model_hashes": self.model_hashes})
        if self.test_executed and proposed != self.configuration_hash and experiment_version == self.experiment_version:
            raise ValueError("configuration changed after TEST; create a new experiment version")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**canonical_value(self), "configuration_hash": self.configuration_hash}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
