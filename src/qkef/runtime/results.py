"""Single-source final results loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_final_results(root: Path | None = None) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[3]
    path = root / "reports/final/final_results.json"
    if not path.is_file():
        raise FileNotFoundError("final results are missing; run scripts/run_final_analysis.py")
    return json.loads(path.read_text(encoding="utf-8"))
