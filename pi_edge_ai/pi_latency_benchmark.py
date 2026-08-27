#!/usr/bin/env python3
"""Measure feature extraction and classifier latency on the Raspberry Pi."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from features import Event, extract_features
from ml_model import GaussianTrafficClassifier


def f(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def i(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except ValueError:
        return default


def load_events(path: Path) -> list[Event]:
    events: list[Event] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            events.append(
                Event(
                    server_ts=f(row.get("server_ts")),
                    source_ip=row.get("source_ip", ""),
                    device=row.get("device", ""),
                    client_role=row.get("client_role", ""),
                    label=row.get("label", ""),
                    scenario_id=row.get("scenario_id", ""),
                    request_seq=i(row.get("request_seq")),
                    payload_bytes=i(row.get("payload_bytes")),
                    sensor={
                        "ax": f(row.get("ax")),
                        "ay": f(row.get("ay")),
                        "az": f(row.get("az")),
                        "gx": f(row.get("gx")),
                        "gy": f(row.get("gy")),
                        "gz": f(row.get("gz")),
                    },
                )
            )
    return events


def build_windows(events: list[Event], window_seconds: float) -> list[list[Event]]:
    by_scenario: dict[str, list[Event]] = {}
    for event in events:
        if not event.scenario_id or not event.label:
            continue
        by_scenario.setdefault(event.scenario_id, []).append(event)

    windows: list[list[Event]] = []
    for scenario_events in by_scenario.values():
        ordered = sorted(scenario_events, key=lambda item: item.server_ts)
        if not ordered:
            continue
        start = ordered[0].server_ts
        indexed: dict[int, list[Event]] = {}
        for event in ordered:
            window_index = int((event.server_ts - start) // window_seconds)
            indexed.setdefault(window_index, []).append(event)
        windows.extend(window for _, window in sorted(indexed.items()))
    return windows


def stats(values_ms: list[float]) -> dict[str, float]:
    values = sorted(values_ms)
    if not values:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "samples": 0}
    p95_index = min(len(values) - 1, int(round((len(values) - 1) * 0.95)))
    return {
        "mean_ms": sum(values) / len(values),
        "median_ms": values[len(values) // 2],
        "p95_ms": values[p95_index],
        "max_ms": values[-1],
        "samples": len(values),
    }


def time_calls(callable_obj, repeats: int) -> dict[str, float]:
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_obj()
        times.append((time.perf_counter() - start) * 1000.0)
    return stats(times)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-log", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    events = load_events(args.raw_log)
    windows = build_windows(events, args.window_seconds)
    if not windows:
        raise ValueError("No windows built from raw log.")

    model = GaussianTrafficClassifier.load(args.model)
    if not model.ready:
        raise ValueError(f"Model was not ready: {args.model}")

    feature_rows = [extract_features(window, args.window_seconds) for window in windows]
    sample_window = windows[len(windows) // 2]
    sample_features = feature_rows[len(feature_rows) // 2]

    feature_stats = time_calls(lambda: extract_features(sample_window, args.window_seconds), args.repeats)
    classifier_stats = time_calls(lambda: model.predict(sample_features), args.repeats)
    feature_plus_classifier_stats = time_calls(
        lambda: model.predict(extract_features(sample_window, args.window_seconds)),
        args.repeats,
    )

    result = {
        "device": "Raspberry Pi 5 Model B Rev 1.1",
        "python_runtime": "Python 3.13.5 on Raspberry Pi OS",
        "raw_log": str(args.raw_log),
        "model": str(args.model),
        "window_seconds": args.window_seconds,
        "event_count": len(events),
        "window_count": len(windows),
        "repeats": args.repeats,
        "feature_extraction": feature_stats,
        "classifier_only": classifier_stats,
        "feature_plus_classifier": feature_plus_classifier_stats,
        "dashboard_refresh_seconds": 0.7,
        "expected_visible_detection_delay_seconds": args.window_seconds + 0.7,
        "note": "Timing uses time.perf_counter around local Pi Python calls. Classifier-only excludes feature extraction and dashboard refresh.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
