#!/usr/bin/env python3
"""Evaluate a validation-tuned rate-only baseline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


CLASSES = ["normal", "burst", "attack_like"]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def predict_rate_only(rate: float, normal_max: float, burst_max: float) -> str:
    if rate < normal_max:
        return "normal"
    if rate < burst_max:
        return "burst"
    return "attack_like"


def class_metrics(actual: list[str], predicted: list[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in CLASSES:
        tp = sum(1 for a, p in zip(actual, predicted) if a == label and p == label)
        fp = sum(1 for a, p in zip(actual, predicted) if a != label and p == label)
        fn = sum(1 for a, p in zip(actual, predicted) if a == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for a in actual if a == label),
        }
    return metrics


def evaluate(rows: Iterable[dict[str, str]], normal_max: float, burst_max: float) -> dict[str, object]:
    actual = [row["label"] for row in rows]
    predicted = [
        predict_rate_only(float(row["messages_per_sec"]), normal_max, burst_max)
        for row in rows
    ]
    per_class = class_metrics(actual, predicted)
    confusion = {
        actual_label: {predicted_label: 0 for predicted_label in CLASSES}
        for actual_label in CLASSES
    }
    for a, p in zip(actual, predicted):
        confusion[a][p] += 1
    accuracy = sum(1 for a, p in zip(actual, predicted) if a == p) / len(actual)
    macro_f1 = sum(per_class[label]["f1"] for label in CLASSES) / len(CLASSES)
    burst_false_alarm_den = sum(1 for a in actual if a == "burst")
    burst_false_alarm = (
        sum(1 for a, p in zip(actual, predicted) if a == "burst" and p == "attack_like")
        / burst_false_alarm_den
        if burst_false_alarm_den
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "attack_like_recall": per_class["attack_like"]["recall"],
        "burst_false_alarm_rate": burst_false_alarm,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "prediction_counts": dict(Counter(predicted)),
        "n": len(actual),
    }


def candidate_thresholds(rows: list[dict[str, str]]) -> list[float]:
    rates = sorted({float(row["messages_per_sec"]) for row in rows})
    if not rates:
        return [0.0]
    mids = [(left + right) / 2 for left, right in zip(rates, rates[1:])]
    return [0.0, *mids, rates[-1] + 1.0]


def tune(validation_rows: list[dict[str, str]]) -> tuple[float, float, dict[str, object]]:
    best: tuple[float, float, float, float, dict[str, object]] | None = None
    thresholds = candidate_thresholds(validation_rows)
    for normal_max in thresholds:
        for burst_max in thresholds:
            if burst_max <= normal_max:
                continue
            metrics = evaluate(validation_rows, normal_max, burst_max)
            score = (
                float(metrics["macro_f1"]),
                float(metrics["attack_like_recall"]),
                -abs(burst_max - normal_max),
            )
            candidate = (score[0], score[1], normal_max, burst_max, metrics)
            if best is None or candidate[:4] > best[:4]:
                best = candidate
    if best is None:
        raise ValueError("No threshold pair could be selected.")
    _, _, normal_max, burst_max, metrics = best
    return normal_max, burst_max, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a tuned rate-only baseline.")
    parser.add_argument("--splits-dir", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/metrics/tuned_rate_baseline.json"))
    parser.add_argument("--normal-max", type=float, default=5.75)
    parser.add_argument("--burst-max", type=float, default=9.65)
    parser.add_argument("--auto-tune", action="store_true", help="Search validation thresholds instead of using the reported thresholds.")
    args = parser.parse_args()

    train = load_rows(args.splits_dir / "train_windows.csv")
    validation = load_rows(args.splits_dir / "validation_windows.csv")
    test = load_rows(args.splits_dir / "test_windows.csv")

    if args.auto_tune:
        normal_max, burst_max, validation_metrics = tune(validation)
        selection_data = "validation split grid search"
    else:
        normal_max, burst_max = args.normal_max, args.burst_max
        validation_metrics = evaluate(validation, normal_max, burst_max)
        selection_data = "reported validation-selected thresholds"
    result = {
        "model": "tuned_rate_only_baseline",
        "feature_used": "messages_per_sec",
        "selection_data": selection_data,
        "normal_max_messages_per_sec": normal_max,
        "burst_max_messages_per_sec": burst_max,
        "rule": (
            "normal if messages_per_sec < normal_max; "
            "burst if messages_per_sec < burst_max; otherwise attack_like"
        ),
        "train_window_count": len(train),
        "validation": validation_metrics,
        "test": evaluate(test, normal_max, burst_max),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
