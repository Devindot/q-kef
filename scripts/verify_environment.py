"""Run a small, offline verification of the Phase 0 environment."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def main() -> int:
    """Report prerequisites, package imports, and configuration readability."""

    print(f"Python: {sys.version.split()[0]}")
    failures: list[str] = []

    if sys.version_info < (3, 11):
        failures.append("Python 3.11 or newer is required")

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))

    try:
        package = importlib.import_module("qkef")
        importlib.import_module("qkef.datasets.fiqa")
        importlib.import_module("qkef.datasets.evolution")
        print(f"Package import: OK (qkef {package.__version__})")
    except Exception as exc:  # pragma: no cover - diagnostic boundary
        failures.append(f"package import failed: {exc}")

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if config["project"]["seed"] != 42 or config["evolution_benchmark"]["seed"] != 42:
            failures.append("configuration seed is missing or not 42")
        elif config["dataset"]["expected_md5"] != "17918ed23cd04fb15047f73e6c3bd9d9":
            failures.append("canonical FiQA MD5 is missing or incorrect")
        else:
            print(f"Configuration: OK ({CONFIG_PATH.relative_to(PROJECT_ROOT)})")
    except Exception as exc:  # pragma: no cover - diagnostic boundary
        failures.append(f"configuration check failed: {exc}")

    if failures:
        print("Environment verification: FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Environment verification: SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
