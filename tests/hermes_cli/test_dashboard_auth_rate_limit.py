"""Rate limiter: per-IP sliding window, 429 + Retry-After, no handler bypass."""
from __future__ import annotations

import pytest

from hermes_cli.dashboard_auth.rate_limit import RateLimitMiddleware


async def _call(mw: RateLimitMiddleware, ip: str = "127.0.0.1") -> tuple[list[dict], int]:
    messages: list[dict] = []
    calls = 0

    async def app(scope, receive, send):
        nonlocal calls
        calls += 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw.app = app
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": "/", "raw_path": b"/",
        "query_string": b"", "headers": [(b"host", b"testserver")],
        "client": (ip, 1234), "server": ("testserver", 80),
    }

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    await mw(scope, receive, send)
    return messages, calls


@pytest.mark.asyncio
async def test_rate_limit_allows_then_rejects() -> None:
    mw = RateLimitMiddleware(app=None, limit=2, window_s=60.0)
    first, calls1 = await _call(mw)
    second, calls2 = await _call(mw)
    third, calls3 = await _call(mw)
    assert first[0]["status"] == 200
    assert second[0]["status"] == 200
    assert third[0]["status"] == 429
    assert any(k.lower() == b"retry-after" for k, _ in third[0]["headers"])
    assert (calls1, calls2, calls3) == (1, 1, 0)


@pytest.mark.asyncio
async def test_rate_limit_is_per_ip() -> None:
    mw = RateLimitMiddleware(app=None, limit=1, window_s=60.0)
    first, _ = await _call(mw, "10.0.0.1")
    second, _ = await _call(mw, "10.0.0.2")
    assert first[0]["status"] == 200
    assert second[0]["status"] == 200
