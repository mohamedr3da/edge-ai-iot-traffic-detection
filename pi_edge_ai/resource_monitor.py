#!/usr/bin/env python3
"""Record Raspberry Pi process/system resource usage to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from urllib.request import urlopen


def read_system_cpu() -> tuple[int, int]:
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        parts = handle.readline().split()[1:]
    values = [int(value) for value in parts]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def read_process_ticks(pid: int) -> int:
    with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
        parts = handle.read().split()
    return int(parts[13]) + int(parts[14])


def read_rss_kb(pid: int) -> int:
    with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def find_pid(match: str) -> int:
    own_pid = os.getpid()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == own_pid:
            continue
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_text(encoding="utf-8", errors="ignore").replace("\x00", " ")
        except OSError:
            continue
        if match in cmdline:
            return pid
    raise SystemExit(f"No process found containing: {match}")


def dashboard_state(url: str) -> dict[str, object]:
    try:
        with urlopen(url, timeout=0.5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample process CPU/RSS on Raspberry Pi")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--match", default="pi_server.py")
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("pi_edge_ai/reports/resource_usage.csv"))
    parser.add_argument("--state-url", default="http://127.0.0.1:8000/api/state")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pid = args.pid or find_pid(args.match)
    ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    args.out.parent.mkdir(parents=True, exist_ok=True)

    exists = args.out.exists() and args.out.stat().st_size > 0
    with args.out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ts",
                "label",
                "pid",
                "process_cpu_percent_one_core",
                "system_cpu_percent",
                "rss_kb",
                "prediction_label",
                "prediction_confidence",
                "messages_per_sec",
                "active_clients",
                "log_bytes",
            ],
        )
        if not exists:
            writer.writeheader()

        prev_total, prev_idle = read_system_cpu()
        prev_proc = read_process_ticks(pid)
        prev_wall = time.time()
        end = prev_wall + args.duration
        while time.time() < end:
            time.sleep(args.interval)
            now = time.time()
            total, idle = read_system_cpu()
            proc = read_process_ticks(pid)
            total_delta = max(total - prev_total, 1)
            idle_delta = max(idle - prev_idle, 0)
            proc_delta = max(proc - prev_proc, 0)
            wall_delta = max(now - prev_wall, 0.001)
            system_cpu = 100.0 * (1.0 - (idle_delta / total_delta))
            proc_cpu = 100.0 * (proc_delta / ticks_per_second) / wall_delta
            state = dashboard_state(args.state_url)
            prediction = state.get("prediction", {}) if isinstance(state.get("prediction"), dict) else {}
            features = state.get("features", {}) if isinstance(state.get("features"), dict) else {}
            writer.writerow(
                {
                    "ts": f"{now:.6f}",
                    "label": args.label,
                    "pid": pid,
                    "process_cpu_percent_one_core": f"{proc_cpu:.3f}",
                    "system_cpu_percent": f"{system_cpu:.3f}",
                    "rss_kb": read_rss_kb(pid),
                    "prediction_label": prediction.get("label", "unknown"),
                    "prediction_confidence": prediction.get("confidence", 0.0),
                    "messages_per_sec": features.get("messages_per_sec", 0.0),
                    "active_clients": features.get("active_clients", 0.0),
                    "log_bytes": state.get("log_bytes", 0),
                }
            )
            handle.flush()
            prev_total, prev_idle, prev_proc, prev_wall = total, idle, proc, now
    print(f"Resource samples written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
