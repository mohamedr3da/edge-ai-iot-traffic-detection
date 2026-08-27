#!/usr/bin/env python3
"""Grouped evaluation, model comparison and figures for the local traffic experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import statistics
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import Event, FEATURE_NAMES, extract_features
from ml_model import CLASSES, GaussianTrafficClassifier, Prediction, rule_fallback


COLORS = {"normal": "#2fbf8f", "burst": "#f5c542", "attack_like": "#ff5c77"}
SPLITS = ("train", "validation", "test")


class CustomGaussianWrapper:
    name = "custom_gaussian_nb"

    def __init__(self) -> None:
        self.model = GaussianTrafficClassifier()

    def fit(self, x_df: pd.DataFrame, y: list[str]) -> "CustomGaussianWrapper":
        rows = []
        for label, (_, features) in zip(y, x_df.iterrows()):
            rows.append((label, {name: float(features[name]) for name in FEATURE_NAMES}))
        self.model.fit(rows)
        return self

    def predict(self, x_df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(row).label for _, row in x_df.iterrows()])

    def predict_one(self, row: pd.Series | dict[str, float]) -> Prediction:
        features = {name: float(row[name]) for name in FEATURE_NAMES}
        return self.model.predict(features)

    def predict_proba(self, x_df: pd.DataFrame) -> np.ndarray:
        out = []
        for _, row in x_df.iterrows():
            probs = self.predict_one(row).probabilities
            out.append([probs.get(label, 0.0) for label in CLASSES])
        return np.array(out)

    def save(self, path: Path) -> int:
        self.model.save(path)
        return path.stat().st_size

    def size_bytes(self) -> int:
        return len(json.dumps(self.model.model).encode("utf-8"))


class BaselineWrapper:
    name = "threshold_rule_baseline"

    def fit(self, x_df: pd.DataFrame, y: list[str]) -> "BaselineWrapper":
        return self

    def predict(self, x_df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(row).label for _, row in x_df.iterrows()])

    def predict_one(self, row: pd.Series | dict[str, float]) -> Prediction:
        features = {name: float(row[name]) for name in FEATURE_NAMES}
        return rule_fallback(features)

    def predict_proba(self, x_df: pd.DataFrame) -> np.ndarray:
        out = []
        for _, row in x_df.iterrows():
            probs = self.predict_one(row).probabilities
            out.append([probs.get(label, 0.0) for label in CLASSES])
        return np.array(out)

    def size_bytes(self) -> int:
        return 0


def sklearn_models(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight=None, random_state=seed),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=1,
        ),
    }


def event_from_csv(row: dict[str, str]) -> Event:
    sensor = {
        "ax": row.get("ax", 0.0),
        "ay": row.get("ay", 0.0),
        "az": row.get("az", 0.0),
        "gx": row.get("gx", 0.0),
        "gy": row.get("gy", 0.0),
        "gz": row.get("gz", 0.0),
    }
    metadata: dict[str, Any] = {}
    raw_metadata = row.get("metadata_json", "")
    if raw_metadata:
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                metadata = parsed
        except json.JSONDecodeError:
            metadata = {"metadata_parse_error": raw_metadata[:120]}
    return Event(
        server_ts=float(row["server_ts"]),
        source_ip=row.get("source_ip", "unknown"),
        device=row.get("device", "unknown"),
        client_role=row.get("client_role", "unknown"),
        label=row.get("label", "unlabelled"),
        payload_bytes=int(float(row.get("payload_bytes", "0") or 0)),
        sensor=sensor,
        scenario_id=row.get("scenario_id", "unknown"),
        request_seq=int(float(row.get("request_seq", "0") or 0)),
        metadata=metadata,
    )


def load_events(paths: list[Path]) -> list[Event]:
    events: list[Event] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("server_ts") or row.get("label") not in CLASSES:
                    continue
                events.append(event_from_csv(row))
    return events


def build_window_rows(events: list[Event], window_seconds: float) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    by_run: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        by_run[event.scenario_id].append(event)

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for scenario_id, run_events in sorted(by_run.items()):
        ordered = sorted(run_events, key=lambda item: item.server_ts)
        if not ordered:
            continue
        labels = Counter(event.label for event in ordered)
        label = labels.most_common(1)[0][0]
        if label not in CLASSES:
            exclusions.append({"scenario_id": scenario_id, "reason": "unsupported_label", "label": label})
            continue
        start = ordered[0].server_ts
        indexed: dict[int, list[Event]] = defaultdict(list)
        for event in ordered:
            window_index = int((event.server_ts - start) // window_seconds)
            indexed[window_index].append(event)
        for window_index, window_events in sorted(indexed.items()):
            features = extract_features(window_events, window_seconds)
            row = {
                "scenario_id": scenario_id,
                "window_index": window_index,
                "label": label,
                "event_count": len(window_events),
                "window_start_rel_s": window_index * window_seconds,
                "window_end_rel_s": (window_index + 1) * window_seconds,
                "run_start_ts": start,
                "source_ips": ";".join(sorted({event.source_ip for event in window_events})),
            }
            row.update(features)
            rows.append(row)

    return pd.DataFrame(rows), exclusions


def split_by_scenario(windows: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = random.Random(seed)
    split_for_scenario: dict[str, str] = {}
    groups_by_label: dict[str, list[str]] = {}
    for label in CLASSES:
        groups = sorted(windows.loc[windows["label"] == label, "scenario_id"].unique())
        rng.shuffle(groups)
        groups_by_label[label] = groups
        count = len(groups)
        if count >= 10:
            test_count = max(1, round(count * 0.10))
            validation_count = max(1, round(count * 0.10))
        elif count >= 3:
            test_count = 1
            validation_count = 1
        elif count == 2:
            test_count = 1
            validation_count = 0
        else:
            test_count = 0
            validation_count = 0
        train_count = max(0, count - validation_count - test_count)
        for scenario_id in groups[:train_count]:
            split_for_scenario[scenario_id] = "train"
        for scenario_id in groups[train_count : train_count + validation_count]:
            split_for_scenario[scenario_id] = "validation"
        for scenario_id in groups[train_count + validation_count :]:
            split_for_scenario[scenario_id] = "test"

    out = windows.copy()
    out["split"] = out["scenario_id"].map(split_for_scenario).fillna("excluded")
    leakage = {}
    for scenario_id, split_count in out.groupby("scenario_id")["split"].nunique().items():
        if split_count > 1:
            leakage[scenario_id] = int(split_count)
    manifest = {
        "method": "grouped_by_complete_scenario_id",
        "seed": seed,
        "requested_ratio": {"train": 0.80, "validation": 0.10, "test": 0.10},
        "groups_by_label": groups_by_label,
        "groups_by_split": {
            split: sorted(out.loc[out["split"] == split, "scenario_id"].unique().tolist())
            for split in SPLITS
        },
        "window_counts_by_split": {
            split: int((out["split"] == split).sum()) for split in SPLITS
        },
        "class_counts_by_split": {
            split: out.loc[out["split"] == split, "label"].value_counts().reindex(CLASSES, fill_value=0).to_dict()
            for split in SPLITS
        },
        "leakage_detected": bool(leakage),
        "leakage_details": leakage,
    }
    return out, manifest


def metric_report(y_true: list[str], y_pred: list[str], confidences: list[float] | None = None) -> dict[str, Any]:
    matrix = {true: {pred: 0 for pred in CLASSES} for true in CLASSES}
    for true, pred in zip(y_true, y_pred):
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1
    per_class = {}
    f1_values = []
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in CLASSES)
    for label in CLASSES:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in CLASSES if other != label)
        fn = sum(matrix[label][other] for other in CLASSES if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }
    burst_total = sum(matrix["burst"].values())
    report = {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "attack_like_recall": per_class["attack_like"]["recall"],
        "burst_false_alarm_rate": matrix["burst"]["attack_like"] / burst_total if burst_total else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "n": len(y_true),
    }
    if confidences:
        report["confidence"] = {
            "mean": float(statistics.mean(confidences)),
            "median": float(statistics.median(confidences)),
            "min": float(min(confidences)),
        }
    return report


def predict_with_confidence(model: Any, x_df: pd.DataFrame) -> tuple[list[str], list[float]]:
    preds = list(model.predict(x_df))
    confidences = [0.0 for _ in preds]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_df)
        confidences = [float(max(row)) for row in probs]
    return preds, confidences


def fit_model(name: str, x_train: pd.DataFrame, y_train: list[str], seed: int) -> Any:
    if name == "threshold_rule_baseline":
        return BaselineWrapper().fit(x_train, y_train)
    if name == "custom_gaussian_nb":
        return CustomGaussianWrapper().fit(x_train, y_train)
    model = sklearn_models(seed)[name]
    return model.fit(x_train, y_train)


def model_size_bytes(name: str, model: Any, model_dir: Path) -> int:
    model_dir.mkdir(parents=True, exist_ok=True)
    if name == "threshold_rule_baseline":
        return 0
    if name == "custom_gaussian_nb":
        return model.save(model_dir / "custom_gaussian_nb_selected.json")
    data = pickle.dumps(model)
    (model_dir / f"{name}.pkl").write_bytes(data)
    return len(data)


def latency_report(model: Any, x_df: pd.DataFrame, repeats: int) -> dict[str, float]:
    if x_df.empty:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    rows = [x_df.iloc[[i % len(x_df)]] for i in range(repeats)]
    latencies = []
    tracemalloc.start()
    cpu_start = time.process_time()
    for row in rows:
        start = time.perf_counter()
        model.predict(row)
        latencies.append((time.perf_counter() - start) * 1000.0)
    cpu_seconds = time.process_time() - cpu_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    latencies_sorted = sorted(latencies)
    p95 = latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))]
    return {
        "mean_ms": float(statistics.mean(latencies)),
        "median_ms": float(statistics.median(latencies)),
        "p95_ms": float(p95),
        "max_ms": float(max(latencies)),
        "cpu_seconds": float(cpu_seconds),
        "peak_python_alloc_kb": float(peak_bytes / 1024.0),
        "repeated_predictions": repeats,
    }


def make_models(seed: int) -> list[str]:
    return [
        "threshold_rule_baseline",
        "custom_gaussian_nb",
        "logistic_regression",
        "decision_tree",
        "random_forest",
    ]


def evaluate_models(windows: pd.DataFrame, seed: int, out_dir: Path, repeats: int) -> dict[str, Any]:
    train = windows[windows["split"] == "train"]
    validation = windows[windows["split"] == "validation"]
    test = windows[windows["split"] == "test"]
    x_train = train[FEATURE_NAMES]
    y_train = train["label"].tolist()
    x_test = test[FEATURE_NAMES]
    y_test = test["label"].tolist()
    x_val = validation[FEATURE_NAMES]
    y_val = validation["label"].tolist()

    metrics: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for name in make_models(seed):
        model = fit_model(name, x_train, y_train, seed)
        fitted[name] = model
        test_pred, test_conf = predict_with_confidence(model, x_test)
        val_pred, val_conf = predict_with_confidence(model, x_val) if not x_val.empty else ([], [])
        metrics[name] = {
            "validation": metric_report(y_val, val_pred, val_conf) if y_val else None,
            "test": metric_report(y_test, test_pred, test_conf),
            "model_size_bytes": model_size_bytes(name, model, out_dir / "models"),
            "latency": latency_report(model, x_test, repeats),
        }

    choice_order = {
        "custom_gaussian_nb": 0,
        "logistic_regression": 1,
        "decision_tree": 2,
        "random_forest": 3,
        "threshold_rule_baseline": 4,
    }
    # Select the deployed model using validation performance only; the test split
    # remains reserved for final reporting.
    ranked = sorted(
        metrics,
        key=lambda name: (
            metrics[name]["validation"]["macro_f1"] if metrics[name]["validation"] else -1.0,
            metrics[name]["validation"]["attack_like_recall"] if metrics[name]["validation"] else -1.0,
            -(metrics[name]["validation"]["burst_false_alarm_rate"] if metrics[name]["validation"] else 1.0),
            -choice_order[name],
        ),
        reverse=True,
    )
    selected = ranked[0]
    if "custom_gaussian_nb" in metrics:
        best = metrics[ranked[0]]["validation"]["macro_f1"]
        custom = metrics["custom_gaussian_nb"]["validation"]["macro_f1"]
        if custom >= best - 0.02:
            selected = "custom_gaussian_nb"

    (out_dir / "models" / "selected_model_summary.json").write_text(
        json.dumps(
            {
                "selected": selected,
                "selection_reason": "Selected using validation performance; when candidates are within 0.02 macro-F1, prefer the simplest interpretable deployment before final test evaluation.",
                "feature_names": FEATURE_NAMES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics["selected_model"] = selected
    return metrics


def grouped_folds(windows: pd.DataFrame, k: int, seed: int) -> list[set[str]]:
    rng = random.Random(seed)
    by_label = {
        label: sorted(windows.loc[windows["label"] == label, "scenario_id"].unique().tolist())
        for label in CLASSES
    }
    max_k = max(2, min(k, *(len(groups) for groups in by_label.values() if groups)))
    folds = [set() for _ in range(max_k)]
    for label, groups in by_label.items():
        rng.shuffle(groups)
        for index, scenario_id in enumerate(groups):
            folds[index % max_k].add(scenario_id)
    return folds


def cross_validation(windows: pd.DataFrame, seed: int, k: int) -> dict[str, Any]:
    folds = grouped_folds(windows, k, seed)
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in make_models(seed)}
    for fold_index, test_groups in enumerate(folds):
        train = windows[~windows["scenario_id"].isin(test_groups)]
        test = windows[windows["scenario_id"].isin(test_groups)]
        x_train = train[FEATURE_NAMES]
        y_train = train["label"].tolist()
        x_test = test[FEATURE_NAMES]
        y_test = test["label"].tolist()
        for name in make_models(seed):
            model = fit_model(name, x_train, y_train, seed + fold_index)
            pred, conf = predict_with_confidence(model, x_test)
            report = metric_report(y_test, pred, conf)
            report["fold"] = fold_index + 1
            report["test_groups"] = sorted(test_groups)
            results[name].append(report)

    summary: dict[str, Any] = {}
    for name, fold_reports in results.items():
        macro = [item["macro_f1"] for item in fold_reports]
        attack = [item["attack_like_recall"] for item in fold_reports]
        summary[name] = {
            "folds": fold_reports,
            "mean_macro_f1": float(statistics.mean(macro)) if macro else 0.0,
            "std_macro_f1": float(statistics.pstdev(macro)) if len(macro) > 1 else 0.0,
            "mean_attack_like_recall": float(statistics.mean(attack)) if attack else 0.0,
            "std_attack_like_recall": float(statistics.pstdev(attack)) if len(attack) > 1 else 0.0,
        }
    return {"k": len(folds), "results": summary}


def learning_curve(windows: pd.DataFrame, selected_model: str, seed: int) -> list[dict[str, Any]]:
    train_groups_by_label = {
        label: sorted(windows.loc[(windows["split"] == "train") & (windows["label"] == label), "scenario_id"].unique())
        for label in CLASSES
    }
    test = windows[windows["split"] == "test"]
    x_test = test[FEATURE_NAMES]
    y_test = test["label"].tolist()
    rows = []
    for fraction in [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]:
        selected_groups: set[str] = set()
        rng = random.Random(seed + int(fraction * 1000))
        for label, groups in train_groups_by_label.items():
            shuffled = list(groups)
            rng.shuffle(shuffled)
            n = max(1, math.ceil(len(shuffled) * fraction)) if shuffled else 0
            selected_groups.update(shuffled[:n])
        train = windows[windows["scenario_id"].isin(selected_groups)]
        if train.empty:
            continue
        model = fit_model(selected_model, train[FEATURE_NAMES], train["label"].tolist(), seed)
        pred, conf = predict_with_confidence(model, x_test)
        report = metric_report(y_test, pred, conf)
        rows.append(
            {
                "train_fraction": fraction,
                "train_groups": len(selected_groups),
                "train_windows": int(len(train)),
                "macro_f1": report["macro_f1"],
                "attack_like_recall": report["attack_like_recall"],
                "burst_false_alarm_rate": report["burst_false_alarm_rate"],
            }
        )
    return rows


def error_analysis(windows: pd.DataFrame, model_name: str, seed: int, out_dir: Path) -> dict[str, Any]:
    train = windows[windows["split"] == "train"]
    test = windows[windows["split"] == "test"]
    model = fit_model(model_name, train[FEATURE_NAMES], train["label"].tolist(), seed)
    preds, confs = predict_with_confidence(model, test[FEATURE_NAMES])
    rows = []
    for (_, row), pred, conf in zip(test.iterrows(), preds, confs):
        item = {
            "scenario_id": row["scenario_id"],
            "window_index": int(row["window_index"]),
            "actual": row["label"],
            "predicted": pred,
            "confidence": conf,
            "correct": row["label"] == pred,
        }
        for feature in FEATURE_NAMES:
            item[feature] = float(row[feature])
        rows.append(item)
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / "evaluation" / "test_predictions_with_features.csv", index=False)
    pred_df[~pred_df["correct"]].to_csv(out_dir / "evaluation" / "misclassified_test_windows.csv", index=False)
    pred_df.sort_values("confidence").head(20).to_csv(out_dir / "evaluation" / "lowest_confidence_correct_or_error.csv", index=False)
    return {
        "selected_model": model_name,
        "test_predictions": len(pred_df),
        "misclassified": int((~pred_df["correct"]).sum()),
        "lowest_confidence_examples": pred_df.sort_values("confidence").head(10).to_dict(orient="records"),
    }


def save_confusion_matrix(matrix: dict[str, dict[str, int]], title: str, path: Path) -> None:
    data = np.array([[matrix[true][pred] for pred in CLASSES] for true in CLASSES])
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=180)
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), [label.replace("_", "-") for label in CLASSES], rotation=25, ha="right")
    ax.set_yticks(range(len(CLASSES)), [label.replace("_", "-") for label in CLASSES])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(data[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_figures(windows: pd.DataFrame, metrics: dict[str, Any], cv: dict[str, Any], learning: list[dict[str, Any]], out_dir: Path) -> None:
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    run_counts = windows.groupby("label")["scenario_id"].nunique().reindex(CLASSES, fill_value=0)
    window_counts = windows.groupby("label").size().reindex(CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    x = np.arange(len(CLASSES))
    ax.bar(x - 0.18, run_counts.values, width=0.36, label="runs", color="#6fa8ff")
    ax.bar(x + 0.18, window_counts.values, width=0.36, label="windows", color="#30c8f2")
    ax.set_xticks(x, [label.replace("_", "-") for label in CLASSES])
    ax.set_ylabel("Count")
    ax.set_title("Class/sample distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "01_class_sample_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    data = [windows.loc[windows["label"] == label, "messages_per_sec"].values for label in CLASSES]
    ax.boxplot(data, tick_labels=[label.replace("_", "-") for label in CLASSES], showmeans=True)
    for i, label in enumerate(CLASSES, start=1):
        y = windows.loc[windows["label"] == label, "messages_per_sec"].values
        ax.scatter(np.full_like(y, i, dtype=float) + np.random.default_rng(7).normal(0, 0.035, len(y)), y, s=18, alpha=0.55, color=COLORS[label])
    ax.set_ylabel("Messages per second")
    ax.set_title("Traffic rate distribution by class")
    fig.tight_layout()
    fig.savefig(figure_dir / "02_traffic_rate_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    for label in CLASSES:
        subset = windows[windows["label"] == label]
        ax.scatter(subset["messages_per_sec"], subset["std_interarrival_ms"], s=26, alpha=0.75, label=label.replace("_", "-"), color=COLORS[label])
    ax.set_xlabel("Messages per second")
    ax.set_ylabel("Inter-arrival jitter (ms)")
    ax.set_title("Feature-space: rate vs timing variability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "03_feature_space_rate_jitter.png", bbox_inches="tight")
    plt.close(fig)

    scaled = StandardScaler().fit_transform(windows[FEATURE_NAMES])
    pca = PCA(n_components=2, random_state=7)
    coords = pca.fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    for label in CLASSES:
        mask = windows["label"] == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=26, alpha=0.75, label=label.replace("_", "-"), color=COLORS[label])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("PCA feature-space visualisation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "04_pca_feature_space.png", bbox_inches="tight")
    plt.close(fig)

    save_confusion_matrix(metrics["threshold_rule_baseline"]["test"]["confusion_matrix"], "Threshold baseline confusion matrix", figure_dir / "05_baseline_confusion_matrix.png")
    selected = metrics["selected_model"]
    save_confusion_matrix(metrics[selected]["test"]["confusion_matrix"], f"{selected.replace('_', ' ')} confusion matrix", figure_dir / "06_selected_model_confusion_matrix.png")

    selected_per_class = metrics[selected]["test"]["per_class"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    width = 0.25
    x = np.arange(len(CLASSES))
    for offset, metric_name in [(-width, "precision"), (0, "recall"), (width, "f1")]:
        ax.bar(x + offset, [selected_per_class[label][metric_name] for label in CLASSES], width=width, label=metric_name)
    ax.set_xticks(x, [label.replace("_", "-") for label in CLASSES])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision, recall and F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "07_per_class_metrics.png", bbox_inches="tight")
    plt.close(fig)

    model_names = [name for name in make_models(7)]
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=180)
    x = np.arange(len(model_names))
    ax.bar(x - 0.18, [metrics[name]["test"]["macro_f1"] for name in model_names], width=0.36, label="macro-F1")
    ax.bar(x + 0.18, [metrics[name]["test"]["attack_like_recall"] for name in model_names], width=0.36, label="attack recall")
    ax.set_xticks(x, [name.replace("_", "\n") for name in model_names])
    ax.set_ylim(0, 1.05)
    ax.set_title("Baseline vs ML model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "08_baseline_vs_models.png", bbox_inches="tight")
    plt.close(fig)

    lc = pd.DataFrame(learning)
    if not lc.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
        ax.plot(lc["train_fraction"] * 100, lc["macro_f1"], marker="o", label="macro-F1")
        ax.plot(lc["train_fraction"] * 100, lc["attack_like_recall"], marker="s", label="attack-like recall")
        ax.set_xlabel("Training groups used (%)")
        ax.set_ylabel("Score on fixed test split")
        ax.set_ylim(0, 1.05)
        ax.set_title("Learning curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "09_learning_curve.png", bbox_inches="tight")
        plt.close(fig)

    pred_path = out_dir / "evaluation" / "test_predictions_with_features.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
        for label in CLASSES:
            vals = pred_df.loc[pred_df["actual"] == label, "confidence"].values
            if len(vals):
                ax.hist(vals, bins=10, alpha=0.55, label=label.replace("_", "-"), color=COLORS[label])
        ax.set_xlabel("Prediction confidence")
        ax.set_ylabel("Test windows")
        ax.set_title("Confidence distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "10_confidence_distribution.png", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=180)
    ax.bar(x, [metrics[name]["latency"]["mean_ms"] for name in model_names], color="#30c8f2")
    ax.set_xticks(x, [name.replace("_", "\n") for name in model_names])
    ax.set_ylabel("Mean single-window prediction latency (ms)")
    ax.set_title("Classifier latency comparison")
    fig.tight_layout()
    fig.savefig(figure_dir / "11_classifier_latency.png", bbox_inches="tight")
    plt.close(fig)


def save_feature_reports(windows: pd.DataFrame, out_dir: Path) -> None:
    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    windows.groupby("label")[FEATURE_NAMES].describe().to_csv(eval_dir / "feature_distributions_by_class.csv")
    windows[FEATURE_NAMES].corr().to_csv(eval_dir / "feature_correlation.csv")


def write_markdown_reports(out_dir: Path, metrics: dict[str, Any]) -> None:
    """Write concise method and benchmark notes alongside the evaluation outputs."""
    docs = out_dir / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "benchmark_feature_mapping.md").write_text(
        """# Benchmark feature mapping

CICIoT2023 and N-BaIoT are used as benchmark context, not as external validation for this project.

| Project feature | Benchmark overlap | Comment |
|---|---|---|
| messages_per_sec, message_count, bytes_per_sec | Partial | Comparable to flow/count/rate-derived network features after preprocessing. |
| mean/std inter-arrival, interarrival_cv | Partial | Comparable in principle if packet timestamps or flow durations are available. |
| active_clients | Partial | Requires source grouping; direct IP shortcuts must be avoided. |
| sensor_accel_mag_mean/std, sensor_gyro_mag_mean | No direct overlap | Specific to this Arduino sensor-node experiment. |

Direct validation would require a preprocessing adapter that rebuilds the same configured feature-window definitions from each benchmark. That was not claimed here.
""",
        encoding="utf-8",
    )
    (docs / "development_history.md").write_text(
        """# Development history evidence

An early pre-final attack-like pilot run was discarded before the final 30-run collection because blocking Arduino serial reads throttled the laptop traffic generator. The gateway was updated to keep the last valid IMU sample and use short non-blocking serial reads, allowing HTTP request timing to remain controlled.

This matters because the experiment compares traffic patterns. A run where the serial reader accidentally limits request rate is not valid evidence of attack-like sustained pressure.
""",
        encoding="utf-8",
    )
    selected = metrics.get("selected_model", "unknown")
    (docs / "model_selection_summary.md").write_text(
        f"""# Model selection summary

Selected model: `{selected}`.

Selection policy: compare candidate models on the validation split, then prefer the simplest interpretable deployment when performance is within 0.02 macro-F1. The final test split is reserved for reporting after selection.

The fixed threshold baseline shows whether ML adds value beyond simple rate rules. The tuned rate-only baseline gives a stronger comparison against request rate alone.
""",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-log", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=683)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=1000)
    args = parser.parse_args(argv)

    out_dir = args.out_dir.resolve()
    for sub in [
        "dataset/raw",
        "dataset/windowed",
        "dataset/splits",
        "models",
        "evaluation/metrics",
        "evaluation/errors",
        "evaluation/resource_tests",
        "figures",
        "docs",
        "evidence",
    ]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    events = load_events(args.raw_log)
    windows, exclusions = build_window_rows(events, args.window_seconds)
    windows, split_manifest = split_by_scenario(windows, args.seed)
    windows.to_csv(out_dir / "dataset" / "windowed" / "windowed_features.csv", index=False)
    for split in SPLITS:
        windows[windows["split"] == split].to_csv(out_dir / "dataset" / "splits" / f"{split}_windows.csv", index=False)

    raw_manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topology": "Arduino USB serial -> Windows laptop -> local Wi-Fi/router -> Raspberry Pi 5",
        "raw_logs": [str(path) for path in args.raw_log],
        "event_count": len(events),
        "window_seconds": args.window_seconds,
        "window_count": int(len(windows)),
        "scenario_count": int(windows["scenario_id"].nunique()) if not windows.empty else 0,
        "class_counts_windows": windows["label"].value_counts().reindex(CLASSES, fill_value=0).to_dict(),
        "class_counts_runs": windows.groupby("label")["scenario_id"].nunique().reindex(CLASSES, fill_value=0).to_dict(),
        "exclusions": exclusions,
    }
    (out_dir / "dataset" / "raw" / "raw_manifest.json").write_text(json.dumps(raw_manifest, indent=2), encoding="utf-8")
    (out_dir / "dataset" / "splits" / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    (out_dir / "dataset" / "splits" / "leakage_check.json").write_text(
        json.dumps(
            {
                "leakage_detected": split_manifest["leakage_detected"],
                "leakage_details": split_manifest["leakage_details"],
                "scenario_ids_checked": int(windows["scenario_id"].nunique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    save_feature_reports(windows, out_dir)
    metrics = evaluate_models(windows, args.seed, out_dir, args.latency_repeats)
    cv = cross_validation(windows, args.seed, args.cv_folds)
    learning = learning_curve(windows, metrics["selected_model"], args.seed)
    errors = error_analysis(windows, metrics["selected_model"], args.seed, out_dir)

    (out_dir / "evaluation" / "metrics" / "model_comparison.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "evaluation" / "metrics" / "grouped_cross_validation.json").write_text(json.dumps(cv, indent=2), encoding="utf-8")
    pd.DataFrame(learning).to_csv(out_dir / "evaluation" / "metrics" / "learning_curve.csv", index=False)
    (out_dir / "evaluation" / "errors" / "error_analysis_summary.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    make_figures(windows, metrics, cv, learning, out_dir)
    write_markdown_reports(out_dir, metrics)

    summary = {
        "raw_manifest": raw_manifest,
        "split_manifest": split_manifest,
        "selected_model": metrics["selected_model"],
        "selected_test_metrics": metrics[metrics["selected_model"]]["test"],
        "baseline_test_metrics": metrics["threshold_rule_baseline"]["test"],
        "cv_summary": {
            name: {
                "mean_macro_f1": value["mean_macro_f1"],
                "std_macro_f1": value["std_macro_f1"],
                "mean_attack_like_recall": value["mean_attack_like_recall"],
                "std_attack_like_recall": value["std_attack_like_recall"],
            }
            for name, value in cv["results"].items()
        },
        "method_limitations": [
            "Local DoS-like/attack-like detection only; not internet-scale DDoS.",
            "Raspberry Pi used the normal local Wi-Fi/router rather than providing the network.",
            "No measured power draw unless an external wattmeter is used.",
            "External CICIoT2023/N-BaIoT validation not claimed without a preprocessing adapter.",
        ],
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
