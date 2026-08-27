#!/usr/bin/env python3
"""Summarise Raspberry Pi resource monitoring evidence for the local traffic experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def pct(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if not series.empty else 0.0


def summarise_numeric(df: pd.DataFrame) -> dict[str, float]:
    return {
        "samples": int(len(df)),
        "duration_s": float(df["elapsed_s"].max() - df["elapsed_s"].min()) if len(df) > 1 else 0.0,
        "process_cpu_mean_percent_one_core": float(df["process_cpu_percent_one_core"].mean()),
        "process_cpu_median_percent_one_core": float(df["process_cpu_percent_one_core"].median()),
        "process_cpu_p95_percent_one_core": pct(df["process_cpu_percent_one_core"], 0.95),
        "process_cpu_max_percent_one_core": float(df["process_cpu_percent_one_core"].max()),
        "system_cpu_mean_percent": float(df["system_cpu_percent"].mean()),
        "system_cpu_p95_percent": pct(df["system_cpu_percent"], 0.95),
        "rss_mean_mb": float((df["rss_kb"] / 1024.0).mean()),
        "rss_p95_mb": pct(df["rss_kb"] / 1024.0, 0.95),
        "rss_max_mb": float((df["rss_kb"] / 1024.0).max()),
        "messages_per_sec_mean": float(df["messages_per_sec"].mean()),
        "messages_per_sec_p95": pct(df["messages_per_sec"], 0.95),
        "messages_per_sec_max": float(df["messages_per_sec"].max()),
    }


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.resource_csv)
    if df.empty:
        raise ValueError("Resource CSV is empty.")

    df["elapsed_s"] = df["ts"] - df["ts"].min()
    active = df[df["log_bytes"] > 0].copy()
    if active.empty:
        active = df.copy()

    summary = {
        "resource_csv": str(Path(args.resource_csv).resolve()),
        "all_samples": summarise_numeric(df),
        "active_collection_samples": summarise_numeric(active),
        "prediction_label_counts": active["prediction_label"].fillna("unknown").value_counts().to_dict(),
    }
    (out_dir / "resource_usage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    by_label = []
    for label, part in active.groupby(active["prediction_label"].fillna("unknown")):
        row = {"prediction_label": label}
        row.update(summarise_numeric(part))
        by_label.append(row)
    pd.DataFrame(by_label).sort_values("prediction_label").to_csv(out_dir / "resource_usage_by_prediction.csv", index=False)

    slim = active[
        [
            "elapsed_s",
            "prediction_label",
            "process_cpu_percent_one_core",
            "system_cpu_percent",
            "rss_kb",
            "messages_per_sec",
            "prediction_confidence",
        ]
    ].copy()
    slim["rss_mb"] = slim["rss_kb"] / 1024.0
    slim.to_csv(out_dir / "resource_usage_samples_trimmed.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8), sharex=True)
    axes[0].plot(active["elapsed_s"], active["process_cpu_percent_one_core"], color="#0b7fab", linewidth=1.2)
    axes[0].set_ylabel("Server CPU %\n(one core)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(active["elapsed_s"], active["rss_kb"] / 1024.0, color="#295f48", linewidth=1.2)
    axes[1].set_ylabel("RSS MB")
    axes[1].grid(alpha=0.25)

    axes[2].plot(active["elapsed_s"], active["messages_per_sec"], color="#bf7d18", linewidth=1.2)
    axes[2].set_ylabel("messages/s")
    axes[2].set_xlabel("elapsed seconds")
    axes[2].grid(alpha=0.25)

    fig.suptitle("Raspberry Pi resource usage during local edge inference")
    fig.tight_layout()
    fig.savefig(out_dir / "resource_usage_timeseries.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
