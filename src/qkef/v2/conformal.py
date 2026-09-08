"""Class-conditional split-conformal lifecycle prediction sets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MondrianConformalClassifier:
    alpha: float = 0.10
    class_quantiles: dict[str, float] = field(default_factory=dict)

    def fit(self, probabilities: np.ndarray, labels: list[str], class_order: list[str]) -> "MondrianConformalClassifier":
        values = np.asarray(probabilities, dtype=float)
        if values.shape != (len(labels), len(class_order)):
            raise ValueError("calibration probability shape mismatch")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")
        for column, class_name in enumerate(class_order):
            scores = sorted(1.0 - values[row, column] for row, label in enumerate(labels) if label == class_name)
            if not scores:
                raise ValueError(f"calibration has no examples for {class_name}")
            rank = min(len(scores), math.ceil((len(scores) + 1) * (1 - self.alpha)))
            self.class_quantiles[class_name] = float(scores[rank - 1])
        return self

    def prediction_set(self, probabilities: dict[str, float]) -> tuple[str, ...]:
        if not self.class_quantiles:
            raise ValueError("conformal classifier is not fitted")
        return tuple(sorted(class_name for class_name, quantile in self.class_quantiles.items() if 1.0 - probabilities.get(class_name, 0.0) <= quantile))

    def coverage(self, probability_rows: list[dict[str, float]], labels: list[str]) -> dict[str, object]:
        sets = [self.prediction_set(row) for row in probability_rows]
        classes = sorted(self.class_quantiles)
        return {
            "marginal_coverage": sum(label in values for label, values in zip(labels, sets)) / max(1, len(labels)),
            "average_set_size": sum(map(len, sets)) / max(1, len(sets)),
            "per_class_coverage": {
                name: sum(label in values for label, values in zip(labels, sets) if label == name) / max(1, sum(label == name for label in labels))
                for name in classes
            },
        }
