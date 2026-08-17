"""Generate frozen Phase 4 analysis artifacts. No training or model selection occurs."""

from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import version

from _phase1_common import PROJECT_ROOT
from qkef.evaluation.final_analysis import build_final_results, release_hashes, write_outputs


def main() -> None:
    manifest_only = "--manifest-only" in sys.argv
    if not manifest_only:
        results = build_final_results(PROJECT_ROOT, iterations=5000)
        write_outputs(PROJECT_ROOT, results)
    manifest = {
        "release": "phase4-final-academic-release", "source_phase3_commit": "1a7c3e2de2dfb81d0eb0b3897493540de6205431",
        "phase3_results_frozen": True, "python": platform.python_version(), "seed": 42, "bootstrap_iterations": 5000,
        "dependency_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "networkx", "sentence-transformers", "streamlit", "plotly", "matplotlib", "pydantic", "PyYAML")},
        "files": release_hashes(PROJECT_ROOT),
    }
    path = PROJECT_ROOT / "reports/final/final_release_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS: refreshed cross-platform release manifest." if manifest_only else "PASS: generated frozen final analysis with 5000 paired bootstrap iterations.")


if __name__ == "__main__":
    main()
