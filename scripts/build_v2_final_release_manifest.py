"""Build the immutable Q-KEF v2 machine-release manifest."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from _phase1_common import PROJECT_ROOT
from qkef.v2.canonical import sha256_file


FIXED_FILES = (
    "README.md",
    "V2_PROJECT_STATUS.md",
    "app_v2.py",
    "configs/v2_confirmatory_benchmark.yaml",
    "configs/v2_confirmatory_experiment.yaml",
    "configs/v2_development.yaml",
    "docs/v2/V2_DEVELOPMENT_PROTOCOL.md",
    "docs/v2/V2_EXPERIMENT_PLAN.md",
    "docs/v2/V2_HYPOTHESES.md",
    "docs/v2/V2_LIMITATIONS.md",
    "docs/v2/V2_RESEARCH_SCOPE.md",
    "reports/v2/CONFIRMATORY_BENCHMARK_BUILD.md",
    "reports/v2/FINAL_RELEASE_SUMMARY.md",
    "reports/v2/confirmatory/CONFIRMATORY_EXPERIMENT_REPORT.md",
    "reports/v2/confirmatory/confirmatory_execution_decisions.csv",
    "reports/v2/confirmatory/confirmatory_results.json",
    "reports/v2/confirmatory/confirmatory_statistics.json",
    "reports/v2/confirmatory/confirmatory_test_predictions.csv",
    "reports/v2/confirmatory/experiment_lock_posttest.json",
    "reports/v2/confirmatory/experiment_lock_pretest.json",
    "reports/v2/confirmatory/pretest_summary.json",
    "reports/v2/human_audit/HUMAN_AUDIT_PROTOCOL.md",
    "reports/v2/human_audit/HUMAN_AUDIT_STATUS.json",
    "reports/v2/human_audit/stratified_review_sample.csv",
)


def source_commit(root: Path) -> str:
    environment = os.environ.copy()
    if os.name == "nt":
        environment["GIT_CONFIG_GLOBAL"] = "NUL"
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, env=environment, text=True
    ).strip()


def main() -> None:
    root = PROJECT_ROOT
    patterns = (
        "src/qkef/v2/**/*.py",
        "scripts/*v2*.py",
        "tests/test_v2*.py",
        "docs/v2/*.md",
        "configs/v2*.yaml",
        "models/v2_confirmatory/*.joblib",
        "reports/v2/confirmatory/figures/figure_*.png",
    )
    extras = {
        path.relative_to(root).as_posix()
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file()
    }
    files = sorted(set(FIXED_FILES) | extras)
    missing = [relative for relative in files if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot build release manifest; missing: {missing}")

    results = json.loads((root / "reports/v2/confirmatory/confirmatory_results.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "reports/v2/human_audit/HUMAN_AUDIT_STATUS.json").read_text(encoding="utf-8"))
    manifest = {
        "release": "Q-KEF v2 final machine-verifiable release",
        "release_status": "MACHINE_COMPLETE_HUMAN_SIGNOFF_PENDING",
        "source_commit": source_commit(root),
        "experiment_version": results["experiment_version"],
        "test_evaluation_count": results["test_evaluation_count"],
        "test_event_count": results["test_event_count"],
        "human_audit_status": audit["status"],
        "human_review_complete": audit["human_review_complete"],
        "file_sha256": {relative: sha256_file(root / relative) for relative in sorted(files)},
    }
    output = root / "reports/v2/final_release_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output} with {len(files)} hashed artifacts.")


if __name__ == "__main__":
    main()
