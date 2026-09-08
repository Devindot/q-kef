"""Paired uncertainty analysis for the Q-KEF v2 retrospective pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.metrics import accuracy_score, f1_score

from _phase1_common import PROJECT_ROOT
from qkef.evolution.models import CLASS_ORDER


SEED = 42
ITERATIONS = 5000


def wilson(successes: int, total: int, confidence: float = 0.95) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = norm.ppf(0.5 + confidence / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [float(centre - margin), float(centre + margin)]


def paired_comparison(truth: np.ndarray, baseline: np.ndarray, challenger: np.ndarray, rng: np.random.Generator) -> dict:
    b_correct, c_correct = baseline == truth, challenger == truth
    b_only = int(np.sum(b_correct & ~c_correct)); c_only = int(np.sum(~b_correct & c_correct))
    discordant = b_only + c_only
    differences = []
    for _ in range(ITERATIONS):
        indices = rng.integers(0, len(truth), len(truth))
        differences.append(float(f1_score(truth[indices], challenger[indices], labels=CLASS_ORDER, average="macro", zero_division=0) - f1_score(truth[indices], baseline[indices], labels=CLASS_ORDER, average="macro", zero_division=0)))
    return {
        "accuracy_difference": float(accuracy_score(truth, challenger) - accuracy_score(truth, baseline)),
        "macro_f1_difference": float(f1_score(truth, challenger, labels=CLASS_ORDER, average="macro", zero_division=0) - f1_score(truth, baseline, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "macro_f1_difference_bootstrap_95_ci": [float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))],
        "mcnemar": {"baseline_only_correct": b_only, "challenger_only_correct": c_only, "discordant": discordant, "exact_p": float(binomtest(min(b_only, c_only), discordant, 0.5).pvalue) if discordant else 1.0},
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda name: p_values[name])
    adjusted, running = {}, 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def main() -> None:
    root = PROJECT_ROOT
    results_path = root / "reports/v2/v2_results.json"
    predictions = pd.read_csv(root / "reports/v2/classification_predictions.csv")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    truth = predictions.target.to_numpy()
    baseline = predictions.B0.to_numpy()
    rng = np.random.default_rng(SEED)
    comparisons = {name: paired_comparison(truth, baseline, predictions[name].to_numpy(), rng) for name in predictions.columns if name not in {"sample_id", "target", "B0"}}
    raw = {name: value["mcnemar"]["exact_p"] for name, value in comparisons.items() if name in {"B1_PCA16", "B2_MATCHED_NON_Q", "Q_FIDELITY", "Q_ENTROPY", "Q_COHERENCE", "Q_FULL"}}
    adjusted = holm(raw)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_mcnemar_p"] = value
    candidate_intervals = {}
    for split, modes in results["candidate_retrieval"].items():
        candidate_intervals[split] = {}
        for mode, values in modes.items():
            total = int(values["eligible"])
            candidate_intervals[split][mode] = {key: wilson(round(values[key] * total), total) for key in ("recall@1", "recall@3", "recall@5", "recall@10")}
    output = {"iterations": ITERATIONS, "seed": SEED, "paired_against_B0": comparisons, "candidate_recall_wilson_95_ci": candidate_intervals, "multiple_comparison_method": "Holm correction across six planned ablations"}
    (root / "reports/v2/statistical_analysis.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    q = comparisons["Q_FULL"]
    markdown = f"""# Q-KEF v2 Statistical Analysis

These are retrospective pilot results on the frozen 60-row v1 TEST partition, not confirmatory Q-KEF v2 evidence. Configuration selection used TRAIN/DEV and a dedicated 60-row TRAIN-derived calibration subset; TEST was not used for tuning.

## Q-full versus B0

- Macro-F1 difference: {q['macro_f1_difference']:+.4f}
- Paired bootstrap 95% CI: [{q['macro_f1_difference_bootstrap_95_ci'][0]:+.4f}, {q['macro_f1_difference_bootstrap_95_ci'][1]:+.4f}]
- Exact McNemar p: {q['mcnemar']['exact_p']:.4f}
- Holm-adjusted p across planned ablations: {q['holm_adjusted_mcnemar_p']:.4f}

The interval crosses zero and the corrected comparison is not statistically significant at 0.05. No quantum-specific advantage is established. Full paired results and candidate-recall Wilson intervals are in `statistical_analysis.json`.
"""
    (root / "reports/v2/STATISTICAL_ANALYSIS.md").write_text(markdown, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "q_full_vs_b0_macro_f1_difference": q["macro_f1_difference"], "mcnemar_p": q["mcnemar"]["exact_p"]}, sort_keys=True))


if __name__ == "__main__":
    main()
