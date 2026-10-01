"""Complete planned post-test contrasts from the frozen prediction table.

This script performs reporting-only analysis. It never loads benchmark TEST
features, models, or mutable configuration and cannot generate new predictions.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
from sklearn.metrics import f1_score

from _phase1_common import PROJECT_ROOT


def exact_mcnemar(truth: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left_only = int(np.sum((left == truth) & (right != truth)))
    right_only = int(np.sum((left != truth) & (right == truth)))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    return {"left_only_correct": left_only, "right_only_correct": right_only, "exact_p": p_value}


def bootstrap_difference(
    truth: np.ndarray, challenger: np.ndarray, baseline: np.ndarray, *, seed: int, iterations: int
) -> list[float]:
    generator = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        sample = generator.integers(0, len(truth), len(truth))
        differences.append(
            f1_score(truth[sample], challenger[sample], average="macro", zero_division=0)
            - f1_score(truth[sample], baseline[sample], average="macro", zero_division=0)
        )
    return [float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))]


def main() -> None:
    report_dir = PROJECT_ROOT / "reports/v2/confirmatory"
    results_path = report_dir / "confirmatory_results.json"
    statistics_path = report_dir / "confirmatory_statistics.json"
    predictions_path = report_dir / "confirmatory_test_predictions.csv"
    postlock_path = report_dir / "experiment_lock_posttest.json"
    if not all(path.is_file() for path in (results_path, statistics_path, predictions_path, postlock_path)):
        raise FileNotFoundError("Frozen confirmatory results, predictions, statistics, and post-test lock are required")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 480 or results["test_evaluation_count"] != 1:
        raise RuntimeError("Expected the frozen single-evaluation 480-row prediction table")

    truth = np.asarray([row["true_action"] for row in rows])
    q_full = np.asarray([row["Q_FULL"] for row in rows])
    matched = np.asarray([row["B2_MATCHED_NON_Q"] for row in rows])
    iterations = int(statistics["iterations"])
    contrast = {
        "challenger": "Q_FULL",
        "baseline": "B2_MATCHED_NON_Q",
        "macro_f1_difference": (
            results["model_metrics"]["Q_FULL"]["macro_f1"]
            - results["model_metrics"]["B2_MATCHED_NON_Q"]["macro_f1"]
        ),
        "paired_bootstrap_95_ci": bootstrap_difference(
            truth, q_full, matched, seed=int(statistics["seed"]), iterations=iterations
        ),
        "mcnemar": exact_mcnemar(truth, q_full, matched),
        "multiplicity_note": "Planned direct H3 contrast; reported unadjusted and not included in the B0-family Holm correction.",
    }
    statistics["planned_contrasts"] = {"Q_FULL_vs_B2_MATCHED_NON_Q": contrast}
    statistics_path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(contrast, sort_keys=True))


if __name__ == "__main__":
    main()
