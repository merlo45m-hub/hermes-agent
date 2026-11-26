"""Negative controls for MRTR cancellation, retry budget, and envelope helpers."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types

from tools.mcp_tool_handlers import (
    _MRTR_ATTEMPTS_PER_TRIP,
    _StdioChildExited,
    _call_tool_with_mrtr,
    _mrtr_cancel_notification,
    _mrtr_final_envelope,
    _mrtr_input_required_envelope,
    _mrtr_is_call_cancelled,
    _mrtr_single_shot_envelope,
)


def _ok():
    return types.CallToolResult(content=[types.TextContent(type="text", text="ok")], is_error=False)


def _server(returns, *, protocol="stateless"):
    s = MagicMock()
    s.session = MagicMock()
    s.session.call_tool = AsyncMock(side_effect=list(returns))
    s.session.send_notification = AsyncMock()
    s._config = {"protocol": protocol}
    s._stdio_children_dead = lambda: False
    s._watch_stdio_children = None
    s._elicitation = None
    s._pending_call_context = None
    return s


def test_cancel_notification_shape_and_idempotency_key():
    n = _mrtr_cancel_notification("key-1", "user cancelled")
    assert n.method == "notifications/cancelled"
    assert n.params.request_id == "key-1"
    assert n.params.reason == "user cancelled"
    assert n.params.meta["dev.hermes/idempotencyKey"] == "key-1"


def test_call_cancelled_detection_all_shapes():
    assert _mrtr_is_call_cancelled(types.ErrorData(code=-32800, message="call_cancelled: user stopped"))
    assert _mrtr_is_call_cancelled({"error": {"code": -32800, "message": "call cancelled"}})
    assert not _mrtr_is_call_cancelled(_ok())


def test_call_cancelled_result_maps_to_error_and_sends_cancel():
    s = _server([types.ErrorData(code=-32800, message="call_cancelled")])
    result = asyncio.run(_call_tool_with_mrtr(s, "srv", "tool", {}))
    assert result.is_error is True
    assert "call_cancelled" in result.content[0].text
    s.session.send_notification.assert_awaited_once()


def test_retry_budget_and_backoff():
    s = _server([TimeoutError("transient"), TimeoutError("transient"), _ok()])
    with patch("tools.mcp_tool_handlers.asyncio.sleep", new=AsyncMock()) as sleep:
        result = asyncio.run(_call_tool_with_mrtr(s, "srv", "tool", {}))
    assert result.content[0].text == "ok"
    assert s.session.call_tool.call_count == _MRTR_ATTEMPTS_PER_TRIP
    assert sleep.await_count == 2


def test_retry_cap_exhaustion_fails_closed():
    s = _server([TimeoutError("t1"), TimeoutError("t2"), TimeoutError("t3")])
    result = asyncio.run(_call_tool_with_mrtr(s, "srv", "tool", {}))
    assert result.is_error is True
    assert "retry budget exhausted" in result.content[0].text
    assert s.session.call_tool.call_count == _MRTR_ATTEMPTS_PER_TRIP


def test_outcome_uncertain_readonly_retries_once():
    s = _server([RuntimeError("outcome_uncertain"), _ok()])
    with patch("tools.mcp_tool_handlers._tool_is_read_only", return_value=True), \
         patch("tools.mcp_tool_handlers.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(_call_tool_with_mrtr(s, "srv", "tool", {}))
    assert result.content[0].text == "ok"
    assert s.session.call_tool.call_count == 2


def test_outcome_uncertain_non_readonly_retry_unsafe():
    s = _server([RuntimeError("outcome_uncertain")])
    with patch("tools.mcp_tool_handlers._tool_is_read_only", return_value=False):
        result = asyncio.run(_call_tool_with_mrtr(s, "srv", "tool", {}))
    assert result.is_error is True
    assert "retry_unsafe" in result.content[0].text
    assert s.session.call_tool.call_count == 1


def test_stdio_child_death_not_retried():
    s = _server([_StdioChildExited("dead", in_flight=True)])
    with pytest.raises(_StdioChildExited):
        asyncio.run(_call_tool_with_mrtr(s, "srv", "tool", {}))
    assert s.session.call_tool.call_count == 1


def test_envelope_helpers():
    single = _mrtr_single_shot_envelope("t", {"a": 1}, "id-1")
    assert single["params"]["_meta"]["dev.hermes/idempotencyKey"] == "id-1"
    inter = _mrtr_input_required_envelope({"r1": {}}, "S1")
    assert inter["resultType"] == "input_required" and inter["requestState"] == "S1"
    final = _mrtr_final_envelope([{"type": "text", "text": "ok"}], "S1")
    assert final["resultType"] == "complete" and final["requestState"] == "S1"
