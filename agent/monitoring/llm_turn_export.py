"""Content-free per-LLM-call telemetry for Phoenix.

One monitoring event per model API attempt -> one ``hermes.llm_turn`` OTLP span.
This is what makes a runaway agent loop (e.g. an 850k-token context) visible in
Phoenix: token buckets, cache hit/write and latency are recorded on every call,
not just in a local log line. Emitted fire-and-forget through the shared
monitoring emitter; if no OTLP streamer is subscribed the event is dropped
locally and never leaves the host. Never raises into the caller.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["emit_llm_turn"]

_INGEST_URL_DEFAULT = "http://10.9.0.1:8090/api/telemetry-ingest"


def _push_ttft_to_backend(ttft_ms: Optional[int]) -> None:
    """Fire-and-forget POST of time-to-first-token to the :8090 telemetry ingest.

    TTFT is measurable only in the gateway's streaming path, so the source pushes
    it to the dashboard rather than the dashboard guessing it. A daemon thread
    with a short timeout keeps the hot path non-blocking; nothing here raises.
    """
    if ttft_ms is None:
        return
    try:
        import json
        import os
        import threading
        import urllib.request

        url = os.environ.get("HERMES_TELEMETRY_INGEST_URL", _INGEST_URL_DEFAULT)
        body = json.dumps({"time_to_first_token": int(ttft_ms)}).encode()

        def _send() -> None:
            try:
                req = urllib.request.Request(
                    url, data=body, method="POST",
                    headers={"content-type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    resp.read()
            except Exception:
                pass

        threading.Thread(target=_send, daemon=True, name="ttft-ingest").start()
    except Exception:
        pass


def emit_llm_turn(
    *,
    model: str,
    provider: str,
    api_call: int,
    latency_ms: int,
    status: str = "ok",
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cache_read_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    ttft_ms: Optional[int] = None,
) -> None:
    """Emit one llm_turn event. All values are primitives; no prompts or IDs."""
    payload: dict[str, Any] = {
        "event": "llm_turn",
        "model": str(model or "unknown")[:128],
        "provider": str(provider or "unknown")[:64],
        "api_call": int(api_call),
        "latency_ms": int(latency_ms),
        "status": str(status)[:32],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_hit_rate": (
            round(cache_read_tokens / input_tokens, 4)
            if (input_tokens and cache_read_tokens is not None) else None
        ),
        "ttft_ms": ttft_ms,
    }
    try:
        from agent.monitoring.emitter import get_emitter
        get_emitter().emit(payload)
    except Exception:
        # Hot-path invariant: observability must never break the turn loop.
        pass
    # :8090 cannot measure TTFT itself; push it from the source.
    _push_ttft_to_backend(ttft_ms)
