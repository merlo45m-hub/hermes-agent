"""Negative-control tests for per-request 2026-07-28 stateless ``meta``.

Asserts on the kwargs forwarded to ``server.session.call_tool`` — no real
MCP server, subprocess, or network is involved.
"""

import asyncio
from unittest.mock import MagicMock

from tools.mcp_tool_handlers import (
    _call_tool_racing_stdio_death,
    _stateless_request_meta,
)

PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"


def _server(protocol: str = "auto") -> MagicMock:
    server = MagicMock()
    server._config = {"protocol": protocol}
    server._stdio_children_dead = lambda: False
    server._watch_stdio_children = None
    server.session.call_tool = MagicMock(return_value=object())
    return server


def _run(server, protocol: str = "auto", progress_callback=None):
    server._config = {"protocol": protocol}
    return asyncio.run(
        _call_tool_racing_stdio_death(
            server, "srv", "echo", {"text": "hi"}, progress_callback
        )
    )


def _meta_kw(server, protocol: str = "auto", progress_callback=None):
    _run(server, protocol, progress_callback)
    kwargs = server.session.call_tool.call_args.kwargs
    return kwargs


# --- helper unit behavior -------------------------------------------------

def test_stateless_request_meta_returns_none_for_legacy_and_auto():
    assert _stateless_request_meta(_server("legacy")) is None
    assert _stateless_request_meta(_server("auto")) is None
    assert _stateless_request_meta(_server("handshake")) is None


def test_stateless_request_meta_attaches_required_fields_for_stateless_modes():
    for mode in ("stateless", "modern", "2026-07-28"):
        meta = _stateless_request_meta(_server(mode))
        assert meta is not None, mode
        assert meta[PROTOCOL_VERSION] == "2026-07-28", mode
        assert CLIENT_CAPABILITIES in meta, mode
        assert isinstance(meta[CLIENT_CAPABILITIES], dict), mode


# --- call_tool attachment / legacy preservation --------------------------

def test_stateless_mode_attaches_required_meta():
    server = _server("stateless")
    kwargs = _meta_kw(server, "stateless")
    assert "meta" in kwargs
    meta = kwargs["meta"]
    assert meta[PROTOCOL_VERSION] == "2026-07-28"
    assert CLIENT_CAPABILITIES in meta
    # progress_token is a separate opt-in; the helper does not set it.
    assert "progress_token" not in meta


def test_modern_and_2026_07_28_also_attach_meta():
    for mode in ("modern", "2026-07-28"):
        server = _server(mode)
        kwargs = _meta_kw(server, mode)
        assert "meta" in kwargs, mode
        assert kwargs["meta"][PROTOCOL_VERSION] == "2026-07-28", mode


def test_legacy_mode_attaches_no_meta():
    server = _server("legacy")
    kwargs = _meta_kw(server, "legacy")
    assert "meta" not in kwargs
    # Legacy: no ``meta`` key and no ``progress_callback`` key (callback is None).
    assert kwargs == {"arguments": {"text": "hi"}}


def test_auto_mode_does_not_attach_meta():
    server = _server("auto")
    kwargs = _meta_kw(server, "auto")
    assert "meta" not in kwargs


def test_handshake_mode_attaches_no_meta():
    server = _server("handshake")
    kwargs = _meta_kw(server, "handshake")
    assert "meta" not in kwargs


def test_stateless_meta_coexists_with_progress_callback():
    server = _server("stateless")
    kwargs = _meta_kw(server, "stateless", progress_callback=lambda p, t, m: None)
    assert "meta" in kwargs
    assert kwargs["meta"][PROTOCOL_VERSION] == "2026-07-28"
    assert kwargs["progress_callback"] is not None


def test_legacy_stays_byte_compatible_with_progress_callback():
    server = _server("legacy")
    kwargs = _meta_kw(server, "legacy", progress_callback=lambda p, t, m: None)
    assert "meta" not in kwargs
    assert kwargs["progress_callback"] is not None
