#!/usr/bin/env python3
"""Train the local traffic classifier.

Use --synthetic first to create a starter model. Later, train from logged CSV
after collecting real labelled windows.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import Event, extract_features
from ml_model import GaussianTrafficClassifier


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "traffic_classifier.json"


def synthetic_window(label: str, window_seconds: float, index: int) -> tuple[str, dict[str, float]]:
    if label == "normal":
        count = random.randint(6, 12)
        clients = ["10.42.0.20"]
        interval_jitter = 0.030
    elif label == "burst":
        count = random.randint(16, 30)
        clients = ["10.42.0.20", "10.42.0.31"] if random.random() < 0.35 else ["10.42.0.20"]
        interval_jitter = 0.050
    else:
        count = random.randint(42, 78)
        clients = ["10.42.0.20", "10.42.0.31"] if random.random() < 0.7 else ["10.42.0.20"]
        interval_jitter = 0.012

    events = []
    t = 1000.0 + index * 10.0
    base_interval = window_seconds / max(count, 1)
    for i in range(count):
        t += max(0.004, random.gauss(base_interval, interval_jitter))
        source_ip = random.choice(clients)
        events.append(
            Event(
                server_ts=t,
                source_ip=source_ip,
                device="synthetic",
                client_role="trainer",
                label=label,
                payload_bytes=random.randint(210, 380),
                sensor={
                    "ax": random.gauss(0.0, 0.04),
                    "ay": random.gauss(0.0, 0.04),
                    "az": random.gauss(1.0, 0.04),
                    "gx": random.gauss(0.0, 1.2),
                    "gy": random.gauss(0.0, 1.2),
                    "gz": random.gauss(0.0, 1.2),
                },
            )
        )
    return label, extract_features(events, window_seconds)


def synthetic_rows(samples_per_class: int, window_seconds: float) -> list[tuple[str, dict[str, float]]]:
    rows = []
    for label in ["normal", "burst", "attack_like"]:
        for index in range(samples_per_class):
            rows.append(synthetic_window(label, window_seconds, index))
    random.shuffle(rows)
    return rows


def rows_from_csv(paths: Iterable[Path], window_seconds: float) -> list[tuple[str, dict[str, float]]]:
    rows = []
    for path in paths:
        events = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    sensor = {
                        "ax": float(row.get("ax") or 0.0),
                        "ay": float(row.get("ay") or 0.0),
                        "az": float(row.get("az") or 0.0),
                        "gx": float(row.get("gx") or 0.0),
                        "gy": float(row.get("gy") or 0.0),
                        "gz": float(row.get("gz") or 0.0),
                    }
                    events.append(
                        Event(
                            server_ts=float(row["server_ts"]),
                            source_ip=row.get("source_ip", "unknown"),
                            device=row.get("device", "unknown"),
                            client_role=row.get("client_role", "unknown"),
                            label=row.get("label", "unlabelled"),
                            payload_bytes=int(float(row.get("payload_bytes") or 0)),
                            sensor=sensor,
                            scenario_id=row.get("scenario_id") or f"{path.stem}_{row.get('label', 'unlabelled')}",
                            request_seq=int(float(row.get("request_seq") or 0)),
                        )
                    )
                except (KeyError, ValueError):
                    continue

        events.sort(key=lambda item: item.server_ts)
        start = events[0].server_ts if events else 0.0
        end = events[-1].server_ts if events else 0.0
        cursor = start
        while cursor + window_seconds <= end:
            window = [event for event in events if cursor <= event.server_ts < cursor + window_seconds]
            labels = [event.label for event in window if event.label in {"normal", "burst", "attack_like"}]
            if len(window) >= 3 and labels:
                label = max(set(labels), key=labels.count)
                rows.append((label, extract_features(window, window_seconds)))
            cursor += window_seconds
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Train a starter model from generated labelled windows")
    parser.add_argument("--logs", nargs="*", type=Path, help="CSV logs from pi_edge_ai/logs")
    parser.add_argument("--samples-per-class", type=int, default=180)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    rows: list[tuple[str, dict[str, float]]] = []
    if args.synthetic:
        rows.extend(synthetic_rows(args.samples_per_class, args.window_seconds))
    if args.logs:
        rows.extend(rows_from_csv(args.logs, args.window_seconds))
    if not rows:
        raise SystemExit("No training rows. Use --synthetic or pass --logs CSV files.")

    model = GaussianTrafficClassifier()
    model.fit(rows)
    model.save(args.out)
    print(f"Saved model to {args.out}")
    print(f"Training windows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
