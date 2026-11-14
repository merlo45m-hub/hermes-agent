"""``hermes metrics summarize`` — compact summary of gateway health + benchmark metrics.

Issue #56. Prints a one-page summary of the monitoring events written by
:mod:`agent.monitoring.gateway_health` and the benchmark timeline
(:mod:`tests.hermes_cli.benchmark_session_timeline`), intended for inclusion in
postmortem dispatches, cron health notices, and the Hermes dashboard.

Usage:
  hermes metrics summarize --events <events.json> --timeline <timeline.json> [--json]

Events JSON: JSON array of objects each starting with ``event``; parsed via
:mod:`agent.monitoring.events` (``GatewayHealthEvent`` / ``GatewayDiagnosticEvent``).
Timeline JSON: JSON array of ``{metric, value, unit, timestamp}`` entries.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from agent.monitoring.events import (
    GatewayDiagnosticEvent,
    GatewayHealthEvent,
    _MonitoringEvent,
)


def load_events(path: Path) -> list[_MonitoringEvent]:
    """Load a JSON array of monitoring events into typed objects.

    A missing, unparseable, or empty file yields an empty list (fail-safe for cron
    jobs that want a summary even when no events were recorded).
    """
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    events: list[_MonitoringEvent] = []
    for item in raw:
        kind = str(item.get("event"))
        payload = {k: v for k, v in item.items() if k != "event"}
        if kind == "gateway_health":
            events.append(GatewayHealthEvent(**payload))
        elif kind == "gateway_diagnostic":
            events.append(GatewayDiagnosticEvent(**payload))
        else:
            # Unknown event kinds are carried as-is so nothing is dropped.
            events.append(_MonitoringEventWrapper(item))
    return events


class _MonitoringEventWrapper:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def event(self) -> str:
        return str(self._data.get("event", ""))


def summarize_events(events: list[_MonitoringEvent]) -> dict[str, Any]:
    health = [e for e in events if isinstance(e, GatewayHealthEvent)]
    diagnostic = [e for e in events if isinstance(e, GatewayDiagnosticEvent)]

    states = Counter()
    active = 0
    for e in health:
        s = str(getattr(e, "gateway_state", "") or getattr(e, "new_state", "") or "")
        if s:
            states[s] += 1
        if getattr(e, "active_agents", 0):
            active = max(active, int(getattr(e, "active_agents", 0)))

    fatal = sum(
        1
        for e in health
        if int(getattr(e, "fatal_platform_count", 0) or 0) > 0
    )
    pid = next((int(e.pid) for e in health if getattr(e, "pid", None)), None)
    version = next((str(e.version) for e in health if getattr(e, "version", None)), None)

    # Benchmark timeline: aggregation by unit/metric.
    timeline = []
    for e in events:
        if not isinstance(e, _MonitoringEventWrapper):
            continue
        payload = e._data
        if "metric" in payload:
            timeline.append(payload)

    agg: dict[str, Any] = {}
    for t in timeline:
        metric = str(t.get("metric"))
        unit = str(t.get("unit") or "1")
        key = (metric, unit)
        agg.setdefault(key, {"metric": metric, "unit": unit, "count": 0, "total": 0.0, "max": None, "min": None})
        val = t.get("value", 0)
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        agg[key]["count"] += 1
        agg[key]["total"] += fval
        agg[key]["max"] = fval if agg[key]["max"] is None else max(agg[key]["max"], fval)
        agg[key]["min"] = fval if agg[key]["min"] is None else min(agg[key]["min"], fval)

    # Drop metrics that only saw invalid (non-numeric) values: they carry no data.
    agg = {k: v for k, v in agg.items() if v["count"] > 0}

    return {
        "event_count": len(events),
        "health_events": len(health),
        "diagnostic_events": len(diagnostic),
        "gateway_states": dict(states),
        "max_active_agents": active,
        "fatal_windows": fatal,
        "pid": pid,
        "version": version,
        "timeline": list(agg.values()),
    }


def format_summary(doc: dict[str, Any]) -> str:
    lines = []
    lines.append("=== Hermes Metrics Summary ===")
    lines.append(f"events: {doc['event_count']} (health {doc['health_events']}, diagnostic {doc['diagnostic_events']})")
    if doc["gateway_states"]:
        lines.append("gateway states: " + ", ".join(f"{k}={v}" for k, v in doc["gateway_states"].items()))
    lines.append(f"max_active_agents: {doc['max_active_agents']}")
    lines.append(f"fatal_windows: {doc['fatal_windows']}")
    if doc["version"]:
        lines.append(f"version: {doc['version']}")
    if doc["pid"]:
        lines.append(f"pid: {doc['pid']}")
    if doc["timeline"]:
        lines.append("timeline:")
        for t in doc["timeline"]:
            lines.append(
                f"  {t['metric']} ({t['unit']}): count={t['count']}, "
                f"min={t['min']}, max={t['max']}, total={t['total']:.3f}"
            )
    return "\n".join(lines)


def cmd_metrics_summarize(args: argparse.Namespace) -> int:
    events_path = Path(args.events) if args.events else Path("events.json")
    timeline_path = Path(args.timeline) if args.timeline else None
    json_out = args.json

    events = load_events(events_path)
    doc = summarize_events(events)

    raw = json.dumps(doc, indent=2) if json_out else format_summary(doc)
    print(raw)
    return 0


def build_metrics_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "metrics",
        help="Gateway / agent metrics (summarize, export)",
        description="Summarize gateway health and benchmark metrics.",
    )
    sub = parser.add_subparsers(dest="metrics_command")

    sum_parser = sub.add_parser("summarize", help="Print a one-page metrics summary")
    sum_parser.add_argument("--events", default="events.json", help="Path to monitoring events JSON")
    sum_parser.add_argument("--timeline", help="Path to benchmark timeline JSON")
    sum_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    sum_parser.set_defaults(func=cmd_metrics_summarize)