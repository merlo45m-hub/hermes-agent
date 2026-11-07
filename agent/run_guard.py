"""Runaway-turn guard: hard context ceiling for autonomous agent loops.

The 2026-11-07 crash was an ~850k-token autonomous loop that drove the phone into
memory pressure until Android's low-memory killer took the gateway. Observability
now records that growth; this module STOPS it.

One-shot per agent: on the first response whose prompt exceeds
``agent.max_context_tokens`` (config ``agent.max_context_tokens``, env
``HERMES_MAX_CONTEXT_TOKENS``, default 400000) it

  * emits an ERROR gateway_diagnostic span to Phoenix with the token + memory
    numbers (so the stop is visible, not silent),
  * appends a cache-safe wrap-up notice to the newest tool result,
  * exhausts the IterationBudget and grants exactly one ``_budget_grace_call``,
    so the model gets one final call to summarise and then the loop exits.

Never raises into the turn loop.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("agent.conversation_loop")

DEFAULT_MAX_CONTEXT_TOKENS = 400_000

CONTEXT_CEILING_WRAPUP_NOTICE = (
    "[SYSTEM NOTICE — context ceiling reached] The conversation has exceeded the "
    "configured maximum context size. Stop expanding the task now: summarise what "
    "is done, state what remains, and finish this turn. Do not start new tool "
    "rounds or repeat earlier work; the loop will end after this response."
)

__all__ = ["DEFAULT_MAX_CONTEXT_TOKENS", "enforce_context_ceiling"]


def _ceiling(agent: Any) -> int:
    """Resolve the ceiling once per agent; config > env > built-in default."""
    current = getattr(agent, "max_context_tokens", None)
    if isinstance(current, int) and not isinstance(current, bool) and current > 0:
        return current
    raw: Any = None
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        raw = (cfg.get("agent") or {}).get("max_context_tokens")
    except Exception:
        raw = None
    if raw is None:
        raw = os.getenv("HERMES_MAX_CONTEXT_TOKENS")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_CONTEXT_TOKENS
    if value <= 0:
        value = DEFAULT_MAX_CONTEXT_TOKENS
    try:
        agent.max_context_tokens = value
    except Exception:
        pass
    return value


def _inject_wrapup(messages: list[dict[str, Any]]) -> bool:
    """Append to the newest non-persisted tool result (cache-safe, mirrors run budget)."""
    try:
        from agent.context_compressor import _DB_PERSISTED_MARKER
    except Exception:
        _DB_PERSISTED_MARKER = "_db_persisted"  # pragma: no cover - defensive
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            if msg.get(_DB_PERSISTED_MARKER):
                return False
            existing = msg.get("content", "")
            if isinstance(existing, str):
                msg["content"] = existing + f"\n\n{CONTEXT_CEILING_WRAPUP_NOTICE}"
            else:
                try:
                    msg["content"] = [*(existing or []), {"type": "text", "text": CONTEXT_CEILING_WRAPUP_NOTICE}]
                except Exception:
                    return False
            return True
    return False


def _memory_numbers() -> dict[str, int]:
    try:
        from gateway.lifecycle_ledger import sample_memory
        mem = sample_memory()
    except Exception:
        return {}
    keep = ("rss_kib", "mem_total_kib", "mem_available_kib", "swap_used_kib")
    return {k: v for k, v in mem.items() if k in keep and isinstance(v, int)}


def enforce_context_ceiling(agent: Any, messages: list[dict[str, Any]], prompt_tokens: int) -> bool:
    """Return True (once) when the ceiling was breached and the loop was stopped."""
    try:
        ceiling = _ceiling(agent)
        if not isinstance(prompt_tokens, int) or prompt_tokens <= ceiling:
            return False
        if getattr(agent, "_context_ceiling_hit", False):
            return False
        agent._context_ceiling_hit = True

        payload: dict[str, Any] = {
            "event": "gateway_diagnostic",
            "name": "context.ceiling_exceeded",
            "subsystem": "agent.turn",
            "error_class": "ContextCeilingExceeded",
            "error_code": "E_CONTEXT_CEILING",
            "severity": "error",
            "prompt_tokens": prompt_tokens,
            "max_context_tokens": ceiling,
            "api_call": int(getattr(agent, "session_api_calls", 0) or 0),
        }
        payload.update(_memory_numbers())
        try:
            from agent.monitoring import emitter
            emitter.get_emitter().emit(payload)
        except Exception:
            pass

        injected = _inject_wrapup(messages)
        try:
            budget = agent.iteration_budget
            if hasattr(budget, "exhaust"):
                budget.exhaust()
            else:  # pragma: no cover - older budget object
                budget._used = budget.max_total
        except Exception:
            logger.debug("context ceiling: could not exhaust iteration budget", exc_info=True)
        agent._budget_grace_call = True
        logger.warning(
            "Context ceiling exceeded (prompt=%d > max=%d); stopping the loop after one wrap-up call "
            "(wrapup_injected=%s)",
            prompt_tokens, ceiling, injected,
        )
        return True
    except Exception:
        logger.debug("context ceiling guard failed", exc_info=True)
        return False
