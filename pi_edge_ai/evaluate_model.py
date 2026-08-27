#!/usr/bin/env python3
"""Create an evaluation report for the local traffic classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import Event, extract_features
from ml_model import CLASSES, GaussianTrafficClassifier, Prediction, rule_fallback
from train_model import synthetic_rows


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "reports" / "evaluation_report.json"
VALID_LABELS = set(CLASSES)


@dataclass(frozen=True)
class WindowRow:
    label: str
    features: dict[str, float]
    group: str


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def window_rows_from_csv(paths: Iterable[Path], window_seconds: float) -> list[WindowRow]:
    grouped_events: dict[str, list[Event]] = defaultdict(list)

    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label = row.get("label", "unlabelled")
                if label not in VALID_LABELS:
                    continue

                scenario_id = row.get("scenario_id") or f"{path.stem}_{label}"
                sensor = {
                    "ax": _safe_float(row.get("ax")),
                    "ay": _safe_float(row.get("ay")),
                    "az": _safe_float(row.get("az")),
                    "gx": _safe_float(row.get("gx")),
                    "gy": _safe_float(row.get("gy")),
                    "gz": _safe_float(row.get("gz")),
                }
                event = Event(
                    server_ts=_safe_float(row.get("server_ts")),
                    source_ip=row.get("source_ip") or "unknown",
                    device=row.get("device") or "unknown",
                    client_role=row.get("client_role") or "unknown",
                    label=label,
                    payload_bytes=_safe_int(row.get("payload_bytes")),
                    sensor=sensor,
                    scenario_id=scenario_id,
                    request_seq=_safe_int(row.get("request_seq")),
                )
                grouped_events[f"{path.stem}:{scenario_id}"].append(event)

    rows: list[WindowRow] = []
    for group, events in grouped_events.items():
        events.sort(key=lambda item: item.server_ts)
        if not events:
            continue

        cursor = events[0].server_ts
        end = events[-1].server_ts
        while cursor + window_seconds <= end:
            window = [event for event in events if cursor <= event.server_ts < cursor + window_seconds]
            labels = [event.label for event in window if event.label in VALID_LABELS]
            if len(window) >= 3 and labels:
                label = Counter(labels).most_common(1)[0][0]
                rows.append(WindowRow(label, extract_features(window, window_seconds), group))
            cursor += window_seconds

    return rows


def synthetic_window_rows(samples_per_class: int, window_seconds: float, group_size: int = 12) -> list[WindowRow]:
    counters: dict[str, int] = Counter()
    rows: list[WindowRow] = []

    for label, features in synthetic_rows(samples_per_class, window_seconds):
        index = counters[label]
        counters[label] += 1
        group = f"synthetic_{label}_{index // group_size:02d}"
        rows.append(WindowRow(label, features, group))

    return rows


def grouped_split(
    rows: list[WindowRow],
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[list[WindowRow], list[WindowRow], list[WindowRow], dict[str, object]]:
    rng = random.Random(seed)
    groups: dict[str, list[WindowRow]] = defaultdict(list)
    for row in rows:
        groups[row.group].append(row)

    by_label: dict[str, list[str]] = defaultdict(list)
    for group, items in groups.items():
        label = Counter(item.label for item in items).most_common(1)[0][0]
        by_label[label].append(group)

    split_groups = {"train": set(), "validation": set(), "test": set()}
    warnings: list[str] = []

    for label in CLASSES:
        label_groups = by_label.get(label, [])
        rng.shuffle(label_groups)
        count = len(label_groups)
        if count >= 5:
            test_count = max(1, round(count * test_fraction))
            validation_count = max(1, round(count * validation_fraction))
            train_count = max(1, count - validation_count - test_count)
            validation_count = min(validation_count, count - train_count)
            test_count = min(test_count, count - train_count - validation_count)
            split_groups["train"].update(label_groups[:train_count])
            split_groups["validation"].update(label_groups[train_count : train_count + validation_count])
            split_groups["test"].update(label_groups[train_count + validation_count : train_count + validation_count + test_count])
        elif count >= 3:
            split_groups["train"].update(label_groups[:-2])
            split_groups["validation"].add(label_groups[-2])
            split_groups["test"].add(label_groups[-1])
        elif count == 2:
            split_groups["train"].add(label_groups[0])
            split_groups["test"].add(label_groups[1])
            warnings.append(f"Only two {label} groups: validation has no separate {label} group.")
        elif count == 1:
            split_groups["train"].add(label_groups[0])
            warnings.append(f"Only one {label} group: collect more runs for grouped validation/test evidence.")
        else:
            warnings.append(f"No {label} groups found.")

    train_rows = [row for row in rows if row.group in split_groups["train"]]
    validation_rows = [row for row in rows if row.group in split_groups["validation"]]
    test_rows = [row for row in rows if row.group in split_groups["test"]]

    if not test_rows and len(rows) >= 9:
        warnings.append("Falling back to window-level split because there were too few scenario groups.")
        return window_level_split(rows, seed, validation_fraction, test_fraction, warnings)

    meta = {
        "method": "grouped_by_scenario_id",
        "warnings": warnings,
        "groups": {name: sorted(values) for name, values in split_groups.items()},
    }
    return train_rows, validation_rows, test_rows, meta


def window_level_split(
    rows: list[WindowRow],
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    warnings: list[str] | None = None,
) -> tuple[list[WindowRow], list[WindowRow], list[WindowRow], dict[str, object]]:
    rng = random.Random(seed)
    by_label: dict[str, list[WindowRow]] = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)

    train_rows: list[WindowRow] = []
    validation_rows: list[WindowRow] = []
    test_rows: list[WindowRow] = []

    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        count = len(label_rows)
        test_count = max(1, round(count * test_fraction)) if count >= 3 else 0
        validation_count = max(1, round(count * validation_fraction)) if count >= 5 else 0
        train_count = max(1, count - test_count - validation_count)
        train_rows.extend(label_rows[:train_count])
        validation_rows.extend(label_rows[train_count : train_count + validation_count])
        test_rows.extend(label_rows[train_count + validation_count :])

    meta = {
        "method": "window_level_fallback",
        "warnings": warnings or [],
        "groups": {},
    }
    return train_rows, validation_rows, test_rows, meta


def class_counts(rows: Iterable[WindowRow]) -> dict[str, int]:
    counts = Counter(row.label for row in rows)
    return {label: counts.get(label, 0) for label in CLASSES}


def metric_report(
    name: str,
    rows: list[WindowRow],
    predictor: Callable[[dict[str, float]], Prediction],
) -> dict[str, object]:
    matrix = {true: {pred: 0 for pred in CLASSES} for true in CLASSES}
    latencies_ms: list[float] = []

    tracemalloc.start()
    cpu_start = time.process_time()
    for row in rows:
        start = time.perf_counter()
        prediction = predictor(row.features)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        if row.label in matrix and prediction.label in matrix[row.label]:
            matrix[row.label][prediction.label] += 1
    cpu_seconds = time.process_time() - cpu_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

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
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }

    sorted_latencies = sorted(latencies_ms)
    p95_index = int(0.95 * (len(sorted_latencies) - 1)) if sorted_latencies else 0
    burst_total = sum(matrix["burst"].values())

    return {
        "name": name,
        "windows": len(rows),
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "attack_like_recall": per_class["attack_like"]["recall"],
        "burst_false_alarm_rate": matrix["burst"]["attack_like"] / burst_total if burst_total else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "latency_ms": {
            "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
            "p95": sorted_latencies[p95_index] if sorted_latencies else 0.0,
            "max": max(latencies_ms) if latencies_ms else 0.0,
        },
        "cpu_seconds": cpu_seconds,
        "peak_memory_kb": peak_bytes / 1024.0,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="*", type=Path, help="CSV logs from pi_edge_ai/logs")
    parser.add_argument("--synthetic", action="store_true", help="Evaluate with generated starter data")
    parser.add_argument("--samples-per-class", type=int, default=180)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-out", type=Path, help="Optionally save the model trained from the train split")
    args = parser.parse_args(argv)

    rows: list[WindowRow] = []
    if args.synthetic:
        rows.extend(synthetic_window_rows(args.samples_per_class, args.window_seconds))
    if args.logs:
        rows.extend(window_rows_from_csv(args.logs, args.window_seconds))
    if not rows:
        raise SystemExit("No evaluation rows. Use --synthetic or pass --logs CSV files.")

    train_rows, validation_rows, test_rows, split_meta = grouped_split(
        rows,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    if not train_rows:
        raise SystemExit("No training rows after split. Collect more labelled runs.")

    model = GaussianTrafficClassifier()
    model.fit((row.label, row.features) for row in train_rows)
    if args.model_out:
        model.save(args.model_out)

    validation_metrics = metric_report("validation_gaussian_model", validation_rows, model.predict) if validation_rows else None
    test_metrics = metric_report("test_gaussian_model", test_rows, model.predict) if test_rows else None
    baseline_metrics = metric_report("test_rule_baseline", test_rows, rule_fallback) if test_rows else None

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window_seconds": args.window_seconds,
        "total_windows": len(rows),
        "class_counts": class_counts(rows),
        "split": {
            "train_windows": len(train_rows),
            "validation_windows": len(validation_rows),
            "test_windows": len(test_rows),
            "train_class_counts": class_counts(train_rows),
            "validation_class_counts": class_counts(validation_rows),
            "test_class_counts": class_counts(test_rows),
            **split_meta,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "baseline": baseline_metrics,
        "model": {
            "type": "gaussian_naive_bayes",
            "classes": CLASSES,
            "model_path": str(args.model_out) if args.model_out else None,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved evaluation report to {args.out}")
    print(f"Split method: {report['split']['method']}")
    if test_metrics:
        print(f"Test macro-F1: {test_metrics['macro_f1']:.3f}")
        print(f"Attack-like recall: {test_metrics['attack_like_recall']:.3f}")
        print(f"Burst false alarm rate: {test_metrics['burst_false_alarm_rate']:.3f}")
        print(f"Mean inference latency: {test_metrics['latency_ms']['mean']:.4f} ms")
    for warning in report["split"]["warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
