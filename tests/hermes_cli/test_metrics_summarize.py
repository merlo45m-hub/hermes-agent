"""Tests for :mod:`hermes_cli.metrics` summarize (Issue #56).

Synthetic events and a synthetic timeline are built in the test, so no network or live
gateway is needed.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hermes_cli.subcommands.metrics import (
    format_summary,
    load_events,
    summarize_events,
)
from hermes_cli.subcommands.metrics import _MonitoringEventWrapper as Wrapper

from agent.monitoring.events import GatewayDiagnosticEvent, GatewayHealthEvent


class SummarizeEventsTest(unittest.TestCase):
    def _health(self, **kw):
        base = dict(
            name="gateway", gateway_state="running", active_agents=3, fatal_platform_count=0,
            pid=12345, version="0.9.0", ts_ns=1,
        )
        base.update(kw)
        return GatewayHealthEvent(**base)

    def _diag(self, **kw):
        base = dict(name="gauge", subsystem="mem", error_class="ok", severity="info", ts_ns=1)
        base.update(kw)
        return GatewayDiagnosticEvent(**base)

    def test_counts_and_states(self):
        events = [
            self._health(gateway_state="running", active_agents=3),
            self._health(gateway_state="running", active_agents=2),
            self._health(gateway_state="degraded", fatal_platform_count=1),
            self._diag(subsystem="cpu", severity="info"),
        ]
        doc = summarize_events(events)
        self.assertEqual(doc["event_count"], 4)
        self.assertEqual(doc["health_events"], 3)
        self.assertEqual(doc["diagnostic_events"], 1)
        self.assertEqual(doc["gateway_states"], {"running": 2, "degraded": 1})
        self.assertEqual(doc["max_active_agents"], 3)
        self.assertEqual(doc["fatal_windows"], 1)
        self.assertEqual(doc["pid"], 12345)
        self.assertEqual(doc["version"], "0.9.0")

    def test_unknown_event_kept(self):
        events = [Wrapper({"event": "some_other"})]
        doc = summarize_events(events)
        self.assertEqual(doc["event_count"], 1)
        self.assertEqual(doc["health_events"], 0)

    def test_timeline_aggregation(self):
        events = [
            Wrapper({"metric": "disk_used", "unit": "%", "value": 40}),
            Wrapper({"metric": "disk_used", "unit": "%", "value": 70}),
            Wrapper({"metric": "disk_used", "unit": "%", "value": "junk"}),  # non-numeric skipped
            Wrapper({"metric": "cpu", "unit": "%", "value": 50}),
        ]
        doc = summarize_events(events)
        self.assertEqual(len(doc["timeline"]), 2)
        disk = next(t for t in doc["timeline"] if t["metric"] == "disk_used")
        self.assertEqual(disk["count"], 2)
        self.assertEqual(disk["total"], 110.0)
        self.assertEqual(disk["max"], 70.0)
        self.assertEqual(disk["min"], 40.0)
        cpu = next(t for t in doc["timeline"] if t["metric"] == "cpu")
        self.assertEqual(cpu["count"], 1)
        self.assertEqual(cpu["max"], 50.0)

    def test_timeline_skip_non_numeric(self):
        events = [Wrapper({"metric": "x", "unit": "1", "value": None})]
        doc = summarize_events(events)
        self.assertEqual(len(doc["timeline"]), 0)  # None value is skipped


class FormatSummaryTest(unittest.TestCase):
    def test_format_contains_keys(self):
        doc = {
            "event_count": 2, "health_events": 2, "diagnostic_events": 0,
            "gateway_states": {"running": 2}, "max_active_agents": 3,
            "fatal_windows": 0, "pid": 7, "version": "1.2.3", "timeline": None,
        }
        out = format_summary(doc)
        self.assertIn("=== Hermes Metrics Summary ===", out)
        self.assertIn("events: 2", out)
        self.assertIn("running=2", out)
        self.assertIn("max_active_agents: 3", out)
        self.assertIn("version: 1.2.3", out)

    def test_json_output_shape(self):
        events = [GatewayHealthEvent(name="g", gateway_state="running", active_agents=2, ts_ns=1)]
        doc = summarize_events(events)
        raw = json.dumps(doc, indent=2)
        self.assertIn("gateway_states", raw)
        self.assertEqual(doc["gateway_states"]["running"], 1)


class LoadEventsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # Write temp event files under the project root (test filesystem guard allows this).
        self._tmpdir = Path(tempfile.mkdtemp(prefix="hermes_metrics_test_", dir=str(Path(__file__).resolve().parent)))
        self._files = []

    def tearDown(self):
        for p in self._files:
            p.unlink(missing_ok=True)
        self._tmpdir.rmdir()

    def _write(self, text: str, name: str = "events.json") -> Path:
        p = self._tmpdir / name
        p.write_text(text, encoding="utf-8")
        self._files.append(p)
        return p

    def test_missing_file_returns_empty(self):
        # A directory stat succeeds (is_file() -> False for dirs), so the guard does not
        # trip on a non-existent path.
        self.assertFalse(load_events(self._tmpdir))

    def test_parses_mixed_events(self):
        text = json.dumps([
            {"event": "gateway_health", "name": "g", "gateway_state": "running", "ts_ns": 1},
            {"event": "gateway_diagnostic", "name": "d", "subsystem": "cpu", "ts_ns": 2},
            {"event": "other", "foo": "bar"},
        ])
        path = self._write(text)
        events = load_events(path)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].name, "g")
        self.assertEqual(events[0].gateway_state, "running")
        self.assertEqual(events[1].name, "d")
        self.assertIsInstance(events[2], Wrapper)

    def test_malformed_file_returns_empty(self):
        path = self._write("{not valid json}")
        events = load_events(path)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
