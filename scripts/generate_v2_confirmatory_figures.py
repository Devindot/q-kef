"""Generate final confirmatory result figures."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qkef-v2-confirmatory-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _phase1_common import PROJECT_ROOT


def bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str, target: float | None = None) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(labels, values, color="#3b5b92")
    if target is not None:
        axis.axhline(target, color="#b23a48", linestyle="--", label=f"target {target:.2f}")
        axis.legend()
    axis.set_title(title); axis.set_ylabel(ylabel); axis.set_ylim(0, max(1.0, max(values) * 1.12))
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout(); figure.savefig(path, dpi=180, facecolor="white"); plt.close(figure)


def main() -> None:
    root = PROJECT_ROOT
    data = json.loads((root / "reports/v2/confirmatory/confirmatory_results.json").read_text(encoding="utf-8"))
    destination = root / "reports/v2/confirmatory/figures"
    destination.mkdir(parents=True, exist_ok=True)
    models = data["model_metrics"]
    bar(destination / "figure_01_model_macro_f1.png", "Confirmatory lifecycle macro-F1", list(models), [models[name]["macro_f1"] for name in models], "Macro-F1")
    candidates = data["candidate_metrics"]
    bar(destination / "figure_02_candidate_recall5.png", "Confirmatory predecessor Recall@5", list(candidates), [candidates[name]["recall@5"] for name in candidates], "Recall@5")
    coverage = data["conformal"]["per_class_coverage"]
    bar(destination / "figure_03_conformal_coverage.png", "Mondrian coverage by action", list(coverage), list(coverage.values()), "Coverage", 0.90)
    technical = data["technical_effects"]
    bar(destination / "figure_04_execution_precision.png", "Direct accuracy versus safe auto-commit precision", ["Direct top-1", "Safe auto-commit"], [technical["direct_top1_accuracy"], technical["auto_commit_precision"]], "Accuracy / precision")
    bar(destination / "figure_05_index_size.png", "Active index versus append-only comparator", ["Q-KEF active", "Append-only"], [technical["final_active_index_size"], technical["append_only_comparator_size"]], "Records")
    hypotheses = data["hypotheses"]
    bar(destination / "figure_06_hypothesis_outcomes.png", "Predeclared hypothesis outcomes", list(hypotheses), [float(value) for value in hypotheses.values()], "Supported (1) / not supported (0)")
    print(f"PASS: generated {len(list(destination.glob('figure_*.png')))} confirmatory figures")


if __name__ == "__main__":
    main()
