#!/usr/bin/env python3
"""Collect a larger real-hardware DoS-like traffic dataset.

Topology used by this project:
Arduino Nano 33 BLE/Sense -> USB serial -> Windows laptop -> local Wi-Fi/router
-> Raspberry Pi 5 edge inference server.

This script never targets public systems. It sends low-rate HTTP requests only
to the configured Raspberry Pi /ingest endpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gateway import ArduinoReader, MockSensor, SensorSample


CLASSES = ("normal", "burst", "attack_like")
TOPOLOGY = "Arduino USB serial -> Windows laptop -> local Wi-Fi/router -> Raspberry Pi 5"
MOVEMENT_CYCLE = (
    "leave the Arduino completely still on the desk",
    "gently rotate/reposition the Arduino once, then leave it still",
    "give the Arduino two or three light taps, then leave it still",
    "gently lift/tilt the Arduino for a moment, then place it back down",
    "leave the Arduino still, but change its orientation slightly during the first few seconds",
)


@dataclass
class RunConfig:
    experiment_id: str
    label: str
    run_index: int
    scenario_id: str
    duration_s: float
    profile: str
    base_rate: float
    peak_rate: float
    jitter_fraction: float
    pause_probability: float
    max_pause_s: float
    burst_period_s: float = 0.0
    burst_length_s: float = 0.0
    note: str = ""
    movement_instruction: str = ""


def now_stamp() -> str:
    """Timestamp used to identify a collection run."""
    return time.strftime("%Y%m%d_%H%M%S")


def local_ip_for(url_host: str) -> str:
    """Resolve the laptop IP address used on the local route to the Pi."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((url_host, 80))
            return sock.getsockname()[0]
    except OSError:
        return "unknown"


def post_json(url: str, payload: dict[str, Any], timeout: float = 2.0) -> tuple[bool, str]:
    """Send one HTTP POST and report whether the Pi accepted it."""
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, str(exc)


def get_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def create_config(experiment_id: str, label: str, run_index: int, duration_s: float, rng: random.Random) -> RunConfig:
    """Create one labelled scenario with controlled but varied timing parameters."""
    scenario_id = f"{experiment_id}_{label}_{run_index:03d}"
    movement_instruction = MOVEMENT_CYCLE[(run_index - 1) % len(MOVEMENT_CYCLE)]

    if label == "normal":
        base = rng.uniform(3.0, 8.0)
        return RunConfig(
            experiment_id=experiment_id,
            label=label,
            run_index=run_index,
            scenario_id=scenario_id,
            duration_s=duration_s,
            profile=rng.choice(["steady_low_jitter", "steady_with_pauses", "upper_normal"]),
            base_rate=base,
            peak_rate=base,
            jitter_fraction=rng.uniform(0.04, 0.12),
            pause_probability=rng.uniform(0.002, 0.018),
            max_pause_s=rng.uniform(0.25, 1.10),
            note="Normal client traffic: steady telemetry with occasional pauses.",
            movement_instruction=movement_instruction,
        )

    if label == "burst":
        base = rng.uniform(3.0, 7.0)
        peak = rng.uniform(11.0, 18.0)
        period = rng.uniform(7.0, 13.0)
        length = rng.uniform(2.0, min(6.0, period - 1.0))
        return RunConfig(
            experiment_id=experiment_id,
            label=label,
            run_index=run_index,
            scenario_id=scenario_id,
            duration_s=duration_s,
            profile=rng.choice(["short_peak", "repeated_burst", "bursty_with_gaps"]),
            base_rate=base,
            peak_rate=peak,
            jitter_fraction=rng.uniform(0.08, 0.20),
            pause_probability=rng.uniform(0.0, 0.010),
            max_pause_s=rng.uniform(0.10, 0.45),
            burst_period_s=period,
            burst_length_s=length,
            note="Legitimate temporary spike: high rate appears in short bursts, not sustained pressure.",
            movement_instruction=movement_instruction,
        )

    if label == "attack_like":
        base = rng.uniform(14.0, 22.0)
        return RunConfig(
            experiment_id=experiment_id,
            label=label,
            run_index=run_index,
            scenario_id=scenario_id,
            duration_s=duration_s,
            profile=rng.choice(["sustained_pressure", "near_burst_boundary", "sustained_jittered"]),
            base_rate=base,
            peak_rate=base,
            jitter_fraction=rng.uniform(0.03, 0.15),
            pause_probability=rng.uniform(0.0, 0.006),
            max_pause_s=rng.uniform(0.05, 0.25),
            note="Safe local attack-like pressure: sustained repetitive requests to the Pi only.",
            movement_instruction=movement_instruction,
        )

    raise ValueError(f"Unsupported label: {label}")


def effective_rate(config: RunConfig, elapsed_s: float) -> tuple[float, str]:
    """Return the active request rate and phase for the current scenario."""
    if config.label != "burst":
        return config.base_rate, "sustained"
    in_burst = (elapsed_s % config.burst_period_s) < config.burst_length_s
    return (config.peak_rate, "burst_peak") if in_burst else (config.base_rate, "burst_gap")


def sleep_for_next_request(config: RunConfig, elapsed_s: float, rng: random.Random) -> None:
    """Apply per-scenario jitter and pauses before sending the next request."""
    rate, _phase = effective_rate(config, elapsed_s)
    base_delay = 1.0 / max(rate, 0.1)
    jitter = base_delay * config.jitter_fraction
    delay = max(0.008, base_delay + rng.uniform(-jitter, jitter))
    if rng.random() < config.pause_probability:
        delay += rng.uniform(0.05, config.max_pause_s)
    time.sleep(delay)


def sensor_payload(sample: SensorSample) -> dict[str, Any]:
    """Serialise the latest Arduino IMU sample into the gateway request."""
    return sample.to_dict()


def run_one(
    config: RunConfig,
    reader: Any,
    ingest_url: str,
    laptop_ip: str,
    output_dir: Path,
    rng: random.Random,
) -> dict[str, Any]:
    """Execute one labelled recording run and append its metadata summary."""
    started_at = time.time()
    sent = 0
    failed = 0
    first_error = ""
    min_interval = 1.0
    max_interval = 0.0
    last_send = None

    metadata = asdict(config)
    metadata.update(
        {
            "started_at_unix": started_at,
            "started_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "laptop_ip": laptop_ip,
            "pi_ingest_url": ingest_url,
            "network_topology": TOPOLOGY,
        }
    )

    print(f"\n[{config.scenario_id}] {config.label} for {config.duration_s:.1f}s")
    print(
        f"  profile={config.profile} base={config.base_rate:.2f}/s peak={config.peak_rate:.2f}/s "
        f"jitter={config.jitter_fraction:.2f}"
    )
    print(f"  USER ACTION: {config.movement_instruction}", flush=True)

    while time.time() - started_at < config.duration_s:
        sample = reader.read_latest()
        if sample is None:
            time.sleep(0.02)
            continue

        elapsed = time.time() - started_at
        rate_now, phase = effective_rate(config, elapsed)
        now = time.time()
        if last_send is not None:
            gap = now - last_send
            min_interval = min(min_interval, gap)
            max_interval = max(max_interval, gap)
        last_send = now
        sent += 1
        payload = {
            "device": "laptop_gateway",
            "client_role": "laptop_gateway",
            "label": config.label,
            "scenario_id": config.scenario_id,
            "request_seq": sent,
            "client_ts": now,
            "client_send_ts": now,
            "profile": config.profile,
            "target_rate": rate_now,
            "base_rate": config.base_rate,
            "peak_rate": config.peak_rate,
            "planned_duration_s": config.duration_s,
            "phase": phase,
            "network_topology": TOPOLOGY,
            "metadata": metadata,
            "sensor": sensor_payload(sample),
        }
        ok, message = post_json(ingest_url, payload)
        if not ok:
            failed += 1
            if not first_error:
                first_error = message
            if failed <= 3:
                print(f"  request failed: {message}")

        sleep_for_next_request(config, time.time() - started_at, rng)

    finished_at = time.time()
    elapsed_total = finished_at - started_at
    expected_low = config.duration_s * min(config.base_rate, config.peak_rate) * 0.45
    failure_rate = failed / sent if sent else 1.0
    exclusion_reason = ""
    if sent < expected_low:
        exclusion_reason = f"very_low_sent_count: sent {sent}, expected at least {expected_low:.1f}"
    elif failure_rate > 0.05:
        exclusion_reason = f"high_failure_rate: {failure_rate:.3f}"

    summary = {
        **metadata,
        "finished_at_unix": finished_at,
        "finished_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": elapsed_total,
        "sent": sent,
        "failed": failed,
        "failure_rate": failure_rate,
        "actual_send_rate": sent / max(elapsed_total, 0.001),
        "min_send_interval_s": min_interval if sent > 1 else None,
        "max_send_interval_s": max_interval if sent > 1 else None,
        "excluded": bool(exclusion_reason),
        "exclusion_reason": exclusion_reason,
        "first_error": first_error,
    }

    with (output_dir / "run_metadata.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")

    print(
        f"  finished sent={sent} failed={failed} actual={summary['actual_send_rate']:.2f}/s "
        f"excluded={summary['excluded']}"
    )
    return summary


def write_plan(configs: list[RunConfig], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(config) for config in configs]
    (output_dir / "run_plan.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_dir / "run_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def build_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Create the shuffled list of planned scenarios for the collection session."""
    rng = random.Random(args.seed)
    experiment_id = args.experiment_id or f"cw2_real_{now_stamp()}"
    configs: list[RunConfig] = []
    labels = CLASSES if not args.labels else tuple(args.labels)
    for label in labels:
        for run_index in range(args.start_index, args.start_index + args.runs_per_class):
            configs.append(create_config(experiment_id, label, run_index, args.duration, rng))
    rng.shuffle(configs)
    return configs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect varied local real-hardware traffic runs")
    parser.add_argument("--pi", default="http://192.168.1.187:8000", help="Pi server base URL")
    parser.add_argument("--serial", default="COM4", help="Arduino serial port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mock-sensor", action="store_true")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--runs-per-class", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--labels", nargs="*", choices=CLASSES)
    parser.add_argument("--pause-between", type=float, default=2.5)
    parser.add_argument("--prompt-hold", type=float, default=5.0, help="Seconds to wait after printing movement prompt")
    parser.add_argument("--seed", type=int, default=683)
    parser.add_argument("--experiment-id")
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/raw_collection"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ingest_url = args.pi.rstrip("/") + "/ingest"
    health_url = args.pi.rstrip("/") + "/api/health"
    host = args.pi.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    laptop_ip = local_ip_for(host)
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args)
    write_plan(configs, output_dir)
    health = get_json(health_url)
    (output_dir / "preflight_health.json").write_text(json.dumps(health, indent=2), encoding="utf-8")

    print(f"Experiment output: {output_dir}")
    print(f"Topology: {TOPOLOGY}")
    print(f"Pi health: {health}")
    print(f"Planned runs: {len(configs)}")

    if args.dry_run:
        return 0

    if args.mock_sensor:
        reader: Any = MockSensor()
    else:
        reader = ArduinoReader(args.serial, args.baud)

    summaries: list[dict[str, Any]] = []
    for idx, config in enumerate(configs, start=1):
        print(f"\nRun {idx}/{len(configs)}", flush=True)
        print(f"PREPARE: next {config.label} run, {config.duration_s:.0f}s. When START appears, {config.movement_instruction}.", flush=True)
        if args.prompt_hold > 0:
            time.sleep(args.prompt_hold)
        print(f"START MOVE NOW: {config.movement_instruction}.", flush=True)
        summaries.append(run_one(config, reader, ingest_url, laptop_ip, output_dir, random.Random(args.seed + idx)))
        if idx < len(configs):
            time.sleep(args.pause_between)

    with (output_dir / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    post_health = get_json(health_url)
    (output_dir / "postflight_health.json").write_text(json.dumps(post_health, indent=2), encoding="utf-8")
    print(f"\nDone. Metadata written under {output_dir}")
    print(f"Pi log path: {post_health.get('log_path', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
