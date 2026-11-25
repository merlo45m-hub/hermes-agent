"""Negative-control tests for bounded MCP 2026-07-28 MRTR round trips."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from mcp.types import CallToolResult, ElicitRequest, ElicitRequestFormParams, ElicitResult, InputRequiredResult, TextContent

from tools.mcp_tool_handlers import _MAX_MRTR_ROUND_TRIPS, _call_tool_with_mrtr


def _ok():
    return CallToolResult(content=[TextContent(type="text", text="ok")], is_error=False)


def _input_required(requests, request_state="S1"):
    return InputRequiredResult(input_requests=requests, request_state=request_state)


def _elicit_request():
    return ElicitRequest(
        method="elicitation/create",
        params=ElicitRequestFormParams(
            message="Need approval", requested_schema={"type": "object", "properties": {}},
        ),
    )


def _server(returns, *, protocol="stateless", elicitation=None):
    server = MagicMock()
    server.session = MagicMock()
    server.session.call_tool = AsyncMock(side_effect=list(returns))
    server._config = {"protocol": protocol}
    server._stdio_children_dead = lambda: False
    server._watch_stdio_children = None
    server._elicitation = elicitation
    server._pending_call_context = None
    return server


def test_ordinary_result_no_retry():
    result = _ok()
    server = _server([result], protocol="legacy")
    assert asyncio.run(_call_tool_with_mrtr(server, "srv", "tool", {})) is result
    assert server.session.call_tool.call_count == 1
    assert "allow_input_required" not in server.session.call_tool.call_args.kwargs


def test_stateless_first_call_opts_in():
    result = _ok()
    server = _server([result])
    asyncio.run(_call_tool_with_mrtr(server, "srv", "tool", {}))
    assert server.session.call_tool.call_args.kwargs["allow_input_required"] is True


def test_input_required_resolves_elicit_then_succeeds():
    final = _ok()
    server = _server([_input_required({"r1": _elicit_request()}), final])
    answer = ElicitResult(action="accept", content={})
    handler = AsyncMock(return_value=answer)
    server._elicitation = handler
    assert asyncio.run(_call_tool_with_mrtr(server, "srv", "tool", {})) is final
    calls = [c.kwargs for c in server.session.call_tool.call_args_list]
    assert calls[0].get("allow_input_required") is True
    assert calls[1]["input_responses"] == {"r1": answer}
    assert calls[1]["request_state"] == "S1"
    handler.assert_awaited_once()


def test_unsupported_input_request_declines_explicitly():
    request = __import__("mcp").types.CreateMessageRequest(
        method="sampling/createMessage", params={"messages": [], "maxTokens": 10})
    server = _server([_input_required({"r1": request})])
    result = asyncio.run(_call_tool_with_mrtr(server, "srv", "tool", {}))
    assert result.is_error is True
    assert "sampling/createMessage" in result.content[0].text
    assert server.session.call_tool.call_count == 1


def test_round_trip_cap_enforced():
    server = _server([_input_required({"r1": _elicit_request()}), _input_required({"r1": _elicit_request()})])
    server._elicitation = AsyncMock(return_value=ElicitResult(action="accept", content={}))
    result = asyncio.run(_call_tool_with_mrtr(server, "srv", "tool", {}))
    assert result.is_error is True
    assert "cap" in result.content[0].text
    assert server.session.call_tool.call_count == _MAX_MRTR_ROUND_TRIPS


def test_declined_elicit_fails_closed():
    server = _server([_input_required({"r1": _elicit_request()})])
    server._elicitation = AsyncMock(return_value=ElicitResult(action="decline"))
    result = asyncio.run(_call_tool_with_mrtr(server, "srv", "tool", {}))
    assert result.is_error is True
    assert "declined" in result.content[0].text
    assert server.session.call_tool.call_count == 1


def test_legacy_mode_no_opt_in():
    result = _ok()
    server = _server([result], protocol="legacy")
    asyncio.run(_call_tool_with_mrtr(server, "srv", "tool", {}))
    assert "allow_input_required" not in server.session.call_tool.call_args.kwargs
