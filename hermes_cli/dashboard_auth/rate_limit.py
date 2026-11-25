"""Per-IP sliding-window rate limiting for the Hermes web dashboard."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Awaitable, Callable, Deque, Dict, Tuple

from fastapi.responses import JSONResponse

Scope = dict
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]


class RateLimitMiddleware:
    """Bound requests per client IP with an in-memory sliding window.

    State is process-local by design: the dashboard is a single-process local
    service. Shared-store limiting would add a dependency to the auth path.
    """

    def __init__(self, app: Callable[..., Awaitable[None]], limit: int = 120, window_s: float = 60.0):
        self.app = app
        self.limit = max(1, int(limit))
        self.window_s = float(window_s)
        self._state: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        client = scope.get("client") or ("unknown", 0)
        ip = str(client[0]) if client else "unknown"
        now = time.monotonic()
        with self._lock:
            hits = self._state.get(ip, deque())
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = max(1, int(self.window_s - (now - hits[0])))
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "rate_limited"},
                    headers={"Retry-After": str(retry_after)},
                )
                allowed = False
            else:
                hits.append(now)
                self._state[ip] = hits
                allowed = True
        if not allowed:
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
