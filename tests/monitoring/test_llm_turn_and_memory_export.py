"""Tests for the Phoenix tracing gap fixes: LLM-turn spans, memory attributes,
diagnostic ERROR status, and the shared monitoring filter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.monitoring import gateway_health_export as GHE
from agent.monitoring import otlp_exporter as OE
from agent.monitoring.events import GatewayHealthEvent


def _mem_provider():
    """In-memory span provider, skipped when the otlp extra isn't installed."""
    import pytest
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_monitoring_filter_accepts_all_four_planes():
    for kind in ("gateway_health", "cron_execution", "gateway_diagnostic", "llm_turn"):
        assert GHE._gateway_health_event({"event": kind}) is True
    assert GHE._gateway_health_event({"event": "random"}) is False


def test_llm_turn_attrs_are_allowlisted_and_content_free():
    attrs = OE._span_attrs({
        "event": "llm_turn", "model": "tencent/hy4", "provider": "nous",
        "api_call": 7, "input_tokens": 65356, "output_tokens": 391,
        "total_tokens": 65747, "cache_read_tokens": 60992,
        "cache_write_tokens": 0, "latency_ms": 5900, "status": "ok",
        "secret_prompt": "must-not-egress",
    })
    assert attrs["hermes.model"] == "tencent/hy4"
    assert attrs["hermes.input_tokens"] == 65356
    assert attrs["hermes.total_tokens"] == 65747
    assert attrs["hermes.latency_ms"] == 5900
    assert not any("secret" in k for k in attrs)


def test_gateway_health_attrs_include_memory():
    attrs = OE._span_attrs({
        "event": "gateway_health", "name": "gateway.health_snapshot",
        "rss_kib": 535520, "mem_total_kib": 11389620,
        "mem_available_kib": 3610112, "swap_used_kib": 5421044,
    })
    assert attrs["hermes.rss_kib"] == 535520
    assert attrs["hermes.mem_available_kib"] == 3610112
    assert attrs["hermes.swap_used_kib"] == 5421044


def test_export_batch_emits_llm_turn_span_with_attrs():
    provider, mem = _mem_provider()
    n = OE.export_batch(provider, [{
        "event": "llm_turn", "model": "m", "provider": "p", "api_call": 3,
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
        "latency_ms": 1234, "status": "ok",
    }])
    assert n == 1
    span = mem.get_finished_spans()[0]
    assert span.name == "hermes.llm_turn"
    attr_map = {str(k): v for k, v in (span.attributes or {}).items()}
    assert attr_map["hermes.total_tokens"] == 120


def test_export_batch_flags_error_diagnostic():
    provider, mem = _mem_provider()
    OE.export_batch(provider, [{
        "event": "gateway_diagnostic", "name": "platform.fatal",
        "subsystem": "platform.telegram", "error_class": "FatalAdapter",
        "severity": "error",
    }])
    from opentelemetry.trace import StatusCode
    span = mem.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR


def test_export_batch_flags_failed_llm_turn():
    provider, mem = _mem_provider()
    OE.export_batch(provider, [{
        "event": "llm_turn", "model": "m", "provider": "p",
        "api_call": 1, "latency_ms": 10, "status": "usage_unavailable",
    }])
    from opentelemetry.trace import StatusCode
    assert mem.get_finished_spans()[0].status.status_code == StatusCode.ERROR


def test_healthy_llm_turn_stays_unset():
    provider, mem = _mem_provider()
    OE.export_batch(provider, [{
        "event": "llm_turn", "model": "m", "provider": "p",
        "api_call": 1, "latency_ms": 10, "status": "ok",
    }])
    from opentelemetry.trace import StatusCode
    assert mem.get_finished_spans()[0].status.status_code == StatusCode.UNSET


def test_snapshot_events_carry_memory_sample():
    cfg = {"monitoring": {"gateway_health_export": {"diagnostic_events_enabled": True}}}
    snap = SimpleNamespace(events=[GatewayHealthEvent(name="gateway.health_snapshot", pid=123)])
    captured = []
    with patch.object(GHE, "_read_runtime_snapshot", return_value=snap), \
         patch.object(GHE, "_memory_attributes", return_value={
             "rss_kib": 111, "mem_available_kib": 222,
         }), \
         patch.object(GHE.emitter, "emit", side_effect=lambda ev: captured.append(ev)):
        GHE._emit_snapshot_events(cfg)
    assert len(captured) == 1
    assert captured[0]["event"] == "gateway_health"
    assert captured[0]["rss_kib"] == 111
    assert captured[0]["mem_available_kib"] == 222
    assert captured[0]["pid"] == 123


def test_memory_attributes_are_int_only(monkeypatch):
    import gateway.lifecycle_ledger as ll
    monkeypatch.setattr(ll, "sample_memory", lambda: {
        "rss_kib": 4096, "mem_total_kib": 8192,
        "mem_available_kib": 2048, "swap_used_kib": 1024, "junk": "x",
    })
    out = GHE._memory_attributes()
    assert out == {
        "rss_kib": 4096, "mem_total_kib": 8192,
        "mem_available_kib": 2048, "swap_used_kib": 1024,
    }


def test_build_exporter_honours_timeout_seconds(monkeypatch):
    import agent.monitoring.otlp_exporter as oe
    captured = {}

    class FakeExporter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(oe, "_require_sdk", lambda *a, **k: {"OTLPSpanExporter": FakeExporter})
    oe.build_exporter({"monitoring": {"export": {"otlp": {
        "enabled": True, "endpoint": "http://127.0.0.1:6006", "timeout_seconds": 45,
    }}}})
    assert captured["endpoint"] == "http://127.0.0.1:6006/v1/traces"
    assert captured["timeout"] == 45.0


def test_emit_llm_turn_is_fire_and_forget():
    from agent.monitoring import llm_turn_export
    calls = []
    import agent.monitoring.emitter as em
    with patch.object(em, "get_emitter") as ge:
        ge.return_value.emit.side_effect = lambda ev: calls.append(ev)
        llm_turn_export.emit_llm_turn(
            model="m", provider="p", api_call=2, latency_ms=99,
            input_tokens=10, output_tokens=5, total_tokens=15,
        )
    assert len(calls) == 1
    assert calls[0]["event"] == "llm_turn"
    assert calls[0]["total_tokens"] == 15
