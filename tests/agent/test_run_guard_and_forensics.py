"""Tests for the crash-prevention work: context ceiling, forensic span for
unclean exits, and memory on lifecycle spans."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.iteration_budget import IterationBudget
from agent.run_guard import enforce_context_ceiling, CONTEXT_CEILING_WRAPUP_NOTICE


def _agent(max_context_tokens=1000):
    return SimpleNamespace(
        max_context_tokens=max_context_tokens,
        iteration_budget=IterationBudget(50),
        session_api_calls=7,
        _budget_grace_call=False,
        _context_ceiling_hit=False,
    )


def test_iteration_budget_exhaust():
    b = IterationBudget(10)
    b.consume()
    assert b.remaining == 9
    b.exhaust()
    assert b.remaining == 0
    assert b.consume() is False


def test_below_ceiling_does_nothing():
    a = _agent()
    msgs = [{"role": "tool", "content": "ok"}]
    assert enforce_context_ceiling(a, msgs, 999) is False
    assert a.iteration_budget.remaining == 50
    assert a._budget_grace_call is False
    assert msgs[0]["content"] == "ok"


def test_above_ceiling_stops_loop_and_emits_span():
    a = _agent()
    msgs = [{"role": "user", "content": "go"}, {"role": "tool", "content": "result"}]
    captured = []
    import agent.monitoring.emitter as em
    with patch.object(em, "get_emitter") as ge:
        ge.return_value.emit.side_effect = lambda ev: captured.append(ev)
        assert enforce_context_ceiling(a, msgs, 850000) is True
    assert a._context_ceiling_hit is True
    assert a._budget_grace_call is True
    assert a.iteration_budget.remaining == 0
    assert CONTEXT_CEILING_WRAPUP_NOTICE in msgs[1]["content"]
    assert len(captured) == 1
    ev = captured[0]
    assert ev["event"] == "gateway_diagnostic"
    assert ev["severity"] == "error"
    assert ev["prompt_tokens"] == 850000
    assert ev["max_context_tokens"] == 1000
    assert ev["error_class"] == "ContextCeilingExceeded"


def test_ceiling_is_one_shot():
    a = _agent()
    assert enforce_context_ceiling(a, [{"role": "tool", "content": "x"}], 5000) is True
    assert enforce_context_ceiling(a, [{"role": "tool", "content": "x"}], 9000) is False


def test_guard_never_raises_on_broken_agent():
    # Contract: observability/guard code must never break the turn loop, even on
    # a partially-constructed agent. The return value is not the guarantee.
    broken = SimpleNamespace()
    enforce_context_ceiling(broken, [], 10**9)


def test_ceiling_reads_config_when_agent_has_none(monkeypatch):
    a = SimpleNamespace(iteration_budget=IterationBudget(50), session_api_calls=1,
                        _budget_grace_call=False, _context_ceiling_hit=False)
    monkeypatch.setenv("HERMES_MAX_CONTEXT_TOKENS", "2000")
    # Force the config lookup to miss so the env var wins.
    with patch("hermes_cli.config.load_config", side_effect=RuntimeError("no config")):
        assert enforce_context_ceiling(a, [{"role": "tool", "content": "x"}], 1999) is False
        assert enforce_context_ceiling(a, [{"role": "tool", "content": "x"}], 2001) is True


def test_stage_and_flush_unclean_exit_span(tmp_path, monkeypatch):
    import gateway.lifecycle_ledger as ll
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "state").mkdir()
    evidence = {
        "prior_pid": 30613, "suspected_oom": True,
        "last_heartbeat_mem": {"rss_kib": 535520, "mem_available_kib": 3610112,
                               "swap_used_kib": 5421044, "mem_total_kib": 11389620},
    }
    ll._stage_unclean_exit_span(evidence, tmp_path)
    path = ll.get_pending_unclean_exit_path(tmp_path)
    assert path.exists()
    staged = json.loads(path.read_text())
    assert staged["name"] == "gateway.unclean_exit"
    assert staged["error_class"] == "SuspectedOOM"
    assert staged["rss_kib"] == 535520

    captured = []
    import agent.monitoring.emitter as em
    with patch.object(em, "get_emitter") as ge:
        ge.return_value.emit.side_effect = lambda ev: captured.append(ev)
        assert ll.flush_unclean_exit_span(tmp_path) is True
    assert not path.exists()
    assert captured[0]["event"] == "gateway_diagnostic"
    assert captured[0]["severity"] == "error"
    assert captured[0]["prior_pid"] == 30613
    assert ll.flush_unclean_exit_span(tmp_path) is False


def test_lifecycle_transition_carries_memory(monkeypatch):
    import agent.monitoring.gateway_health as gh
    import agent.monitoring.emitter as em
    captured = []
    monkeypatch.setattr(gh, "_safe_profile", lambda: "default")
    monkeypatch.setattr(gh, "_safe_version", lambda: "0.22.2")
    with patch("gateway.lifecycle_ledger.sample_memory", return_value={
        "rss_kib": 111, "mem_available_kib": 222, "swap_used_kib": 333, "junk": "x",
    }), patch.object(em, "get_emitter") as ge:
        ge.return_value.emit.side_effect = lambda ev: captured.append(ev)
        gh.emit_runtime_status_transition(
            {"gateway_state": "starting"},
            {"gateway_state": "running", "pid": 42, "active_agents": 1},
        )
    assert captured, "expected a lifecycle event"
    life = [ev for ev in captured if ev.get("name") == "gateway.lifecycle"]
    assert life, captured
    assert life[0]["rss_kib"] == 111
    assert life[0]["mem_available_kib"] == 222
    assert life[0]["swap_used_kib"] == 333
    assert "junk" not in life[0]


def test_diagnostic_allowlist_carries_forensic_numbers():
    from agent.monitoring.otlp_exporter import _span_attrs
    attrs = _span_attrs({
        "event": "gateway_diagnostic", "name": "gateway.unclean_exit",
        "error_class": "SuspectedOOM", "severity": "error", "prior_pid": 30613,
        "suspected_oom": True, "rss_kib": 535520, "prompt_tokens": 850000,
    })
    assert attrs["hermes.rss_kib"] == 535520
    assert attrs["hermes.prompt_tokens"] == 850000
    assert attrs["hermes.prior_pid"] == 30613
