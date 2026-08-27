"""Feature extraction for fixed-length traffic windows.

The Raspberry Pi server and the offline evaluation use the same functions so
that live predictions and reported metrics are based on identical features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from statistics import mean, pstdev
from typing import Any, Iterable


FEATURE_NAMES = [
    "messages_per_sec",
    "message_count",
    "active_clients",
    "mean_interarrival_ms",
    "std_interarrival_ms",
    "interarrival_cv",
    "bytes_per_sec",
    "sensor_accel_mag_mean",
    "sensor_accel_mag_std",
    "sensor_gyro_mag_mean",
]


@dataclass
class Event:
    """One request received by the Raspberry Pi, including its sensor payload."""
    server_ts: float
    source_ip: str
    device: str
    client_role: str
    label: str
    payload_bytes: int
    sensor: dict[str, Any]
    scenario_id: str = "unknown"
    request_seq: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert sensor fields to floats while tolerating missing or invalid values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def event_from_payload(server_ts: float, source_ip: str, payload: dict[str, Any], payload_bytes: int) -> Event:
    """Build an Event from the JSON body received by the Pi /ingest endpoint."""
    try:
        request_seq = int(payload.get("request_seq", 0) or 0)
    except (TypeError, ValueError):
        request_seq = 0
    metadata: dict[str, Any] = {}
    if isinstance(payload.get("metadata"), dict):
        metadata.update(payload["metadata"])
    for key in (
        "experiment_id",
        "run_id",
        "run_index",
        "profile",
        "target_rate",
        "base_rate",
        "peak_rate",
        "planned_duration_s",
        "phase",
        "network_topology",
        "client_send_ts",
    ):
        if key in payload:
            metadata[key] = payload[key]

    return Event(
        server_ts=server_ts,
        source_ip=source_ip,
        device=str(payload.get("device", "unknown")),
        client_role=str(payload.get("client_role", "unknown")),
        label=str(payload.get("label", "unlabelled")),
        payload_bytes=payload_bytes,
        sensor=payload.get("sensor") if isinstance(payload.get("sensor"), dict) else {},
        scenario_id=str(payload.get("scenario_id", "unknown")),
        request_seq=request_seq,
        metadata=metadata,
    )


def extract_features(events: Iterable[Event], window_seconds: float) -> dict[str, float]:
    """Convert a window of request events into the 10 numeric model features."""
    ordered = sorted(events, key=lambda item: item.server_ts)
    if not ordered:
        return {name: 0.0 for name in FEATURE_NAMES}

    # Use the configured window length so sparse and dense windows are comparable.
    duration = max(window_seconds, ordered[-1].server_ts - ordered[0].server_ts, 0.001)
    times = [event.server_ts for event in ordered]
    intervals_ms = [(b - a) * 1000.0 for a, b in zip(times, times[1:]) if b >= a]
    accel_mags = []
    gyro_mags = []

    for event in ordered:
        # IMU magnitudes provide lightweight embedded-sensor context without payload inspection.
        sensor = event.sensor
        ax = _safe_float(sensor.get("ax"))
        ay = _safe_float(sensor.get("ay"))
        az = _safe_float(sensor.get("az"))
        gx = _safe_float(sensor.get("gx"))
        gy = _safe_float(sensor.get("gy"))
        gz = _safe_float(sensor.get("gz"))
        accel_mags.append(math.sqrt(ax * ax + ay * ay + az * az))
        gyro_mags.append(math.sqrt(gx * gx + gy * gy + gz * gz))

    interval_mean = mean(intervals_ms) if intervals_ms else window_seconds * 1000.0
    interval_std = pstdev(intervals_ms) if len(intervals_ms) > 1 else 0.0

    return {
        "messages_per_sec": len(ordered) / duration,
        "message_count": float(len(ordered)),
        "active_clients": float(len({event.source_ip for event in ordered})),
        "mean_interarrival_ms": interval_mean,
        "std_interarrival_ms": interval_std,
        "interarrival_cv": interval_std / max(interval_mean, 1.0),
        "bytes_per_sec": sum(event.payload_bytes for event in ordered) / duration,
        "sensor_accel_mag_mean": mean(accel_mags) if accel_mags else 0.0,
        "sensor_accel_mag_std": pstdev(accel_mags) if len(accel_mags) > 1 else 0.0,
        "sensor_gyro_mag_mean": mean(gyro_mags) if gyro_mags else 0.0,
    }


def vectorise(features: dict[str, float]) -> list[float]:
    """Return features in the stable order expected by the saved JSON model."""
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
