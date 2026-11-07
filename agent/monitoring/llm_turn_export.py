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
    }
    try:
        from agent.monitoring.emitter import get_emitter
        get_emitter().emit(payload)
    except Exception:
        # Hot-path invariant: observability must never break the turn loop.
        pass
