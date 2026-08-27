#!/usr/bin/env python3
"""Raspberry Pi receiver, feature extractor, ML inference engine and dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import Event, event_from_payload, extract_features
from ml_model import GaussianTrafficClassifier


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "traffic_classifier_10s_real.json"
LOG_DIR = ROOT / "logs"
WINDOW_SECONDS = 10.0
RETENTION_SECONDS = 120.0


DASHBOARD_HTML = """<!doctype html>
<html>
<head>
  <title>Edge AI Traffic Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #08111f; color: #eef6ff; }
    header { padding: 22px 28px; border-bottom: 1px solid #17324a; }
    h1 { margin: 0 0 6px; font-size: 26px; }
    main { padding: 24px 28px; display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
    section { background: #101d2d; border: 1px solid #1d3b55; border-radius: 8px; padding: 18px; }
    .prediction { font-size: 32px; font-weight: 700; margin-top: 8px; }
    .normal { color: #62d394; }
    .burst { color: #ffd166; }
    .attack_like { color: #ff6b6b; }
    .metric { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #20364d; }
    .bar { height: 12px; background: #14263a; border-radius: 999px; overflow: hidden; margin: 7px 0 12px; }
    .fill { height: 100%; background: #2ec4ff; width: 0%; transition: width .2s; }
    code { color: #94d2ff; }
    small { color: #9db1c4; }
  </style>
</head>
<body>
<header>
  <h1>Edge AI IoT Traffic Dashboard</h1>
  <small>Raspberry Pi local inference.</small>
</header>
<main>
  <section>
    <h2>Prediction</h2>
    <div id="prediction" class="prediction">waiting...</div>
    <div id="confidence"></div>
    <div id="probs"></div>
  </section>
  <section>
    <h2>Traffic Window</h2>
    <div id="metrics"></div>
  </section>
  <section>
    <h2>Clients</h2>
    <div id="clients"></div>
  </section>
  <section>
    <h2>Recent Events</h2>
    <div id="events"></div>
  </section>
</main>
<script>
function pct(x) { return `${Math.round((x || 0) * 100)}%`; }
function fixed(x, n=2) { return Number(x || 0).toFixed(n); }
async function refresh() {
  const response = await fetch('/api/state', {cache: 'no-store'});
  const state = await response.json();
  const pred = state.prediction || {};
  const label = pred.label || 'waiting';
  const predEl = document.getElementById('prediction');
  predEl.className = `prediction ${label}`;
  predEl.textContent = label.replace('_', '-');
  document.getElementById('confidence').textContent = `confidence: ${pct(pred.confidence)}`;
  document.getElementById('probs').innerHTML = Object.entries(pred.probabilities || {}).map(([k,v]) =>
    `<div>${k.replace('_','-')} <div class="bar"><div class="fill" style="width:${pct(v)}"></div></div></div>`
  ).join('');
  const f = state.features || {};
  const metrics = [
    ['messages/sec', fixed(f.messages_per_sec)],
    ['message count', fixed(f.message_count, 0)],
    ['active clients', fixed(f.active_clients, 0)],
    ['mean inter-arrival ms', fixed(f.mean_interarrival_ms, 1)],
    ['jitter ms', fixed(f.std_interarrival_ms, 1)],
    ['bytes/sec', fixed(f.bytes_per_sec, 0)]
  ];
  document.getElementById('metrics').innerHTML = metrics.map(([k,v]) => `<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join('');
  document.getElementById('clients').innerHTML = (state.clients || []).map(c => `<div class="metric"><span>${c.ip}</span><strong>${c.count}</strong></div>`).join('');
  document.getElementById('events').innerHTML = (state.recent_events || []).map(e => `<div><code>${e.source_ip}</code> ${e.label} ${e.device}</div>`).join('');
}
setInterval(refresh, 700);
refresh();
</script>
</body>
</html>"""


class Monitor:
    """Thread-safe event buffer, CSV logger and live prediction state."""

    def __init__(self, model_path: Path) -> None:
        self.events: deque[Event] = deque()
        self.lock = threading.Lock()
        self.model = GaussianTrafficClassifier.load(model_path)
        self.log_path = self._new_log_path()
        self.csv_file = self.log_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "server_ts",
                "source_ip",
                "device",
                "client_role",
                "label",
                "scenario_id",
                "request_seq",
                "payload_bytes",
                "ax",
                "ay",
                "az",
                "gx",
                "gy",
                "gz",
                "metadata_json",
            ],
        )
        if self.log_path.stat().st_size == 0:
            self.writer.writeheader()

    def _new_log_path(self) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return LOG_DIR / f"events_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    def add(self, event: Event) -> None:
        """Store an incoming request and append the raw event to the CSV log."""
        with self.lock:
            self.events.append(event)
            cutoff = time.time() - RETENTION_SECONDS
            while self.events and self.events[0].server_ts < cutoff:
                self.events.popleft()
            sensor = event.sensor
            self.writer.writerow(
                {
                    "server_ts": f"{event.server_ts:.6f}",
                    "source_ip": event.source_ip,
                    "device": event.device,
                    "client_role": event.client_role,
                    "label": event.label,
                    "scenario_id": event.scenario_id,
                    "request_seq": event.request_seq,
                    "payload_bytes": event.payload_bytes,
                    "ax": sensor.get("ax", 0.0),
                    "ay": sensor.get("ay", 0.0),
                    "az": sensor.get("az", 0.0),
                    "gx": sensor.get("gx", 0.0),
                    "gy": sensor.get("gy", 0.0),
                    "gz": sensor.get("gz", 0.0),
                    "metadata_json": json.dumps(event.metadata, sort_keys=True),
                }
            )
            self.csv_file.flush()

    def state(self) -> dict[str, Any]:
        """Build the current dashboard state from the latest fixed-length window."""
        now = time.time()
        with self.lock:
            events = list(self.events)
        window = [event for event in events if event.server_ts >= now - WINDOW_SECONDS]
        features = extract_features(window, WINDOW_SECONDS)
        prediction = None if not window else self.model.predict(features)
        client_counts: dict[str, int] = {}
        for event in window:
            client_counts[event.source_ip] = client_counts.get(event.source_ip, 0) + 1
        return {
            "window_seconds": WINDOW_SECONDS,
            "features": features,
            "prediction": {
                "label": prediction.label if prediction else "waiting",
                "confidence": prediction.confidence if prediction else 0.0,
                "probabilities": prediction.probabilities if prediction else {},
            },
            "clients": [{"ip": ip, "count": count} for ip, count in sorted(client_counts.items())],
            "recent_events": [
                {
                    "source_ip": event.source_ip,
                    "device": event.device,
                    "client_role": event.client_role,
                    "label": event.label,
                    "scenario_id": event.scenario_id,
                    "request_seq": event.request_seq,
                }
                for event in events[-8:]
            ],
            "model_ready": self.model.ready,
            "log_path": str(self.log_path),
            "pid": os.getpid(),
            "log_bytes": self.log_path.stat().st_size if self.log_path.exists() else 0,
        }


def response(handler: BaseHTTPRequestHandler, status: int, content_type: str, body: bytes) -> None:
    """Send a no-cache HTTP response used by the dashboard and API endpoints."""
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    monitor: Monitor

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            response(self, HTTPStatus.OK, "text/html; charset=utf-8", DASHBOARD_HTML.encode("utf-8"))
        elif parsed.path == "/api/state":
            body = json.dumps(self.monitor.state()).encode("utf-8")
            response(self, HTTPStatus.OK, "application/json", body)
        elif parsed.path == "/api/health":
            state = self.monitor.state()
            body = json.dumps(
                {
                    "ok": True,
                    "model_ready": state["model_ready"],
                    "log_path": state["log_path"],
                    "window_seconds": state["window_seconds"],
                }
            ).encode("utf-8")
            response(self, HTTPStatus.OK, "application/json", body)
        elif parsed.path == "/ingest":
            query = parse_qs(parsed.query)
            payload = {key: values[-1] for key, values in query.items()}
            payload.setdefault("device", "query_client")
            payload.setdefault("client_role", "query_client")
            payload.setdefault("label", payload.get("mode", "unlabelled"))
            self._ingest(payload, payload_bytes=len(parsed.query.encode("utf-8")))
        else:
            response(self, HTTPStatus.NOT_FOUND, "text/plain", b"not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/ingest":
            response(self, HTTPStatus.NOT_FOUND, "text/plain", b"not found")
            return
        size = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(size)
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            response(self, HTTPStatus.BAD_REQUEST, "application/json", b'{"ok":false,"error":"invalid_json"}')
            return
        self._ingest(payload, payload_bytes=len(raw))

    def _ingest(self, payload: dict[str, Any], payload_bytes: int) -> None:
        """Convert a received payload into an Event and return the current prediction."""
        event = event_from_payload(time.time(), self.client_address[0], payload, payload_bytes)
        self.monitor.add(event)
        body = json.dumps({"ok": True, "state": self.monitor.state()["prediction"]}).encode("utf-8")
        response(self, HTTPStatus.OK, "application/json", body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main(argv: list[str]) -> int:
    global WINDOW_SECONDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    args = parser.parse_args(argv)

    WINDOW_SECONDS = args.window_seconds
    Handler.monitor = Monitor(args.model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving dashboard on http://{args.host}:{args.port}")
    print(f"Model ready: {Handler.monitor.model.ready}")
    print(f"Logging to: {Handler.monitor.log_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
