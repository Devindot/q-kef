"""Fair DEV-selected logistic-regression lifecycle models."""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASS_ORDER = ["NEW", "REPLACE", "MERGE", "ARCHIVE", "COEXIST", "SPLIT"]


def make_model(c_value: float, seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000, random_state=seed)),
    ])


def select_model(x_train: np.ndarray, y_train: np.ndarray, x_dev: np.ndarray, y_dev: np.ndarray, c_grid: list[float]) -> tuple[Pipeline, dict[str, float]]:
    best = None
    for c_value in c_grid:
        model = make_model(c_value).fit(x_train, y_train)
        prediction = model.predict(x_dev)
        metrics = {
            "C": c_value,
            "accuracy": float(accuracy_score(y_dev, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_dev, prediction)),
            "macro_f1": float(f1_score(y_dev, prediction, average="macro", zero_division=0)),
        }
        key = (metrics["macro_f1"], metrics["balanced_accuracy"], -c_value)
        if best is None or key > best[0]:
            best = (key, model, metrics)
    return best[1], best[2]


def save_reload_verify(model: Pipeline, path: object, features: np.ndarray) -> bool:
    joblib.dump(model, path)
    loaded = joblib.load(path)
    return np.array_equal(model.predict(features), loaded.predict(features)) and np.allclose(model.predict_proba(features), loaded.predict_proba(features))
