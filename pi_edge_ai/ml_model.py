"""Small dependency-free Gaussian classifier for Raspberry Pi inference.

The model is stored as JSON so the live server can run without scikit-learn.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from features import FEATURE_NAMES, vectorise


CLASSES = ["normal", "burst", "attack_like"]


@dataclass
class Prediction:
    """Classifier output returned to the dashboard and evaluation scripts."""
    label: str
    confidence: float
    probabilities: dict[str, float]


class GaussianTrafficClassifier:
    """Custom Gaussian Naive Bayes implementation using raw feature values."""

    def __init__(self, model: dict | None = None) -> None:
        self.model = model or {}

    @property
    def ready(self) -> bool:
        return bool(self.model.get("classes"))

    def fit(self, rows: Iterable[tuple[str, dict[str, float]]]) -> None:
        """Learn class priors plus per-feature means and variances from training rows."""
        grouped: dict[str, list[list[float]]] = {label: [] for label in CLASSES}
        for label, features in rows:
            if label in grouped:
                grouped[label].append(vectorise(features))

        classes = {}
        total = sum(len(items) for items in grouped.values())
        if total == 0:
            raise ValueError("No labelled training rows were supplied.")

        for label, vectors in grouped.items():
            if not vectors:
                continue
            cols = list(zip(*vectors))
            means = [sum(col) / len(col) for col in cols]
            variances = []
            for col, mu in zip(cols, means):
                # A small variance floor avoids divide-by-zero on compact edge datasets.
                var = sum((x - mu) ** 2 for x in col) / max(len(col), 1)
                variances.append(max(var, 1e-6))
            classes[label] = {
                "prior": len(vectors) / total,
                "mean": means,
                "variance": variances,
                "count": len(vectors),
            }

        self.model = {
            "type": "gaussian_naive_bayes",
            "feature_names": FEATURE_NAMES,
            "classes": classes,
        }

    def predict(self, features: dict[str, float]) -> Prediction:
        """Classify one feature window using Gaussian log likelihood and softmax."""
        if not self.ready:
            return rule_fallback(features)

        x = vectorise(features)
        scores: dict[str, float] = {}
        for label, params in self.model["classes"].items():
            prior = max(float(params.get("prior", 1e-6)), 1e-6)
            score = math.log(prior)
            for value, mu, var in zip(x, params["mean"], params["variance"]):
                var = max(float(var), 1e-6)
                score += -0.5 * math.log(2.0 * math.pi * var) - ((value - float(mu)) ** 2) / (2.0 * var)
            scores[label] = score

        return softmax_prediction(scores)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GaussianTrafficClassifier":
        if not path.exists():
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8")))


def softmax_prediction(scores: dict[str, float]) -> Prediction:
    """Convert unnormalised log scores into probabilities for the dashboard."""
    if not scores:
        return Prediction("unknown", 0.0, {})
    max_score = max(scores.values())
    exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
    total = sum(exp_scores.values()) or 1.0
    probabilities = {label: value / total for label, value in exp_scores.items()}
    label = max(probabilities, key=probabilities.get)
    return Prediction(label, probabilities[label], probabilities)


def rule_fallback(features: dict[str, float]) -> Prediction:
    """Transparent hand-written rule used as a baseline and safe fallback."""
    rate = float(features.get("messages_per_sec", 0.0))
    jitter = float(features.get("std_interarrival_ms", 0.0))
    active = float(features.get("active_clients", 0.0))
    if rate >= 24 or (rate >= 16 and active >= 2):
        scores = {"normal": -4.0, "burst": -1.0, "attack_like": 2.5 + rate / 50.0}
    elif rate >= 8 or jitter > 120:
        scores = {"normal": -1.0, "burst": 1.8 + rate / 25.0, "attack_like": 0.2}
    else:
        scores = {"normal": 2.0, "burst": -0.5, "attack_like": -3.0}
    return softmax_prediction(scores)

