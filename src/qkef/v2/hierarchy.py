"""Cardinality-aware interpretation layered over six-class probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASS_ORDER = ("NEW", "REPLACE", "MERGE", "ARCHIVE", "COEXIST", "SPLIT")
CARDINALITY = {"NEW": "0->1", "REPLACE": "1->1", "ARCHIVE": "1->1", "COEXIST": "1->1", "MERGE": "m->1", "SPLIT": "1->r"}


@dataclass(frozen=True)
class HierarchicalDecision:
    predecessor_relation: bool
    cardinality: str
    action: str


class CardinalityAwareResolver:
    def resolve(self, probabilities: dict[str, float], candidate_count: int, *, split_segment_count: int = 0) -> HierarchicalDecision:
        if candidate_count == 0 or probabilities.get("NEW", 0.0) == max(probabilities.values()):
            return HierarchicalDecision(False, "0->1", "NEW")
        if split_segment_count >= 2 and probabilities.get("SPLIT", 0.0) == max(probabilities.values()):
            return HierarchicalDecision(True, "1->r", "SPLIT")
        if candidate_count >= 2 and probabilities.get("MERGE", 0.0) == max(probabilities.values()):
            return HierarchicalDecision(True, "m->1", "MERGE")
        action = max(("REPLACE", "ARCHIVE", "COEXIST"), key=lambda name: (probabilities.get(name, 0.0), name))
        return HierarchicalDecision(True, "1->1", action)


class HierarchicalLifecycleModel:
    """Two-stage fitted model: structural cardinality, then one-to-one action."""

    def __init__(self, c_value: float = 1.0, seed: int = 42):
        def pipeline() -> Pipeline:
            return Pipeline([
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000, random_state=seed)),
            ])
        self.cardinality_model = pipeline()
        self.one_to_one_model = pipeline()

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "HierarchicalLifecycleModel":
        labels = np.asarray(labels)
        structural = np.asarray([CARDINALITY[str(label)] for label in labels])
        self.cardinality_model.fit(features, structural)
        mask = structural == "1->1"
        self.one_to_one_model.fit(features[mask], labels[mask])
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        structural_values = self.cardinality_model.predict_proba(features)
        structural = {name: structural_values[:, index] for index, name in enumerate(self.cardinality_model.classes_)}
        one_values = self.one_to_one_model.predict_proba(features)
        one = {name: one_values[:, index] for index, name in enumerate(self.one_to_one_model.classes_)}
        output = np.zeros((len(features), len(CLASS_ORDER)), dtype=float)
        for column, name in enumerate(CLASS_ORDER):
            cardinality = CARDINALITY[name]
            output[:, column] = structural[cardinality] * one.get(name, np.ones(len(features))) if cardinality == "1->1" else structural[cardinality]
        output /= output.sum(axis=1, keepdims=True)
        return output

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return np.asarray([CLASS_ORDER[index] for index in np.argmax(probabilities, axis=1)])
