"""Hermes pre_tool_call hook for voice input sanitization.

Intercepts tool calls with voice-dictated input, runs VoiceInputProcessor
sanitization, and optionally rewrites phonetic errors via LLM rewriter.

Integration: Voice dictation → Hermes pre_tool_call hook → VoiceInputProcessor → sanitized command → LLM
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Add VoiceInputProcessor path
_VIP_PATHS = [
    "/storage/emulated/0/Games/Muakbank_zombie/tools",
    "/root/mcp-servers",
]
for p in _VIP_PATHS:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

# Import VoiceInputProcessor (may not be available in all environments)
try:
    from VoiceInputProcessor import VoiceInputProcessor, process_input
    _VIP_AVAILABLE = True
except ImportError:
    _VIP_AVAILABLE = False
    logger.warning("VoiceInputProcessor not available, hook disabled")

# Import context engine
try:
    from voice_context_engine import get_context, set_context, enhance_sanitization, build_context_prompt
    _CONTEXT_AVAILABLE = True
except ImportError:
    try:
        from agent.voice_context_engine import get_context, set_context, enhance_sanitization, build_context_prompt
        _CONTEXT_AVAILABLE = True
    except ImportError:
        _CONTEXT_AVAILABLE = False
        logger.warning("Voice context engine not available")

# Import rewriter
try:
    from voice_rewriter import rewrite as llm_rewrite
    _REWRITER_AVAILABLE = True
except ImportError:
    try:
        from agent.voice_rewriter import rewrite as llm_rewrite
        _REWRITER_AVAILABLE = True
    except ImportError:
        _REWRITER_AVAILABLE = False
        logger.warning("Voice rewriter not available")

# Confidence threshold for triggering LLM rewrite
CONFIDENCE_THRESHOLD = 0.65

# Destructive commands that should never run from voice input
DESTRUCTIVE_COMMANDS = {"rm", "rmdir", "chmod", "sudo", "mkfs", "dd", "format"}


def _is_destructive(command: str) -> bool:
    """Check if a command contains destructive operations."""
    cmd_lower = command.lower().strip()
    for dc in DESTRUCTIVE_COMMANDS:
        if cmd_lower.startswith(dc) or f" {dc} " in cmd_lower or f" {dc}\t" in cmd_lower:
            return True
    return False


def _reject_empty(text: str) -> bool:
    """Check if input is empty or whitespace-only."""
    return not text or not text.strip()


def sanitize_voice_input(raw: str) -> dict[str, Any]:
    """Sanitize voice input using VoiceInputProcessor + optional LLM rewrite.

    Returns dict with:
        - sanitized: the sanitized text
        - rewritten: the LLM-rewritten text (if applied)
        - confidence: VoiceInputProcessor confidence score
        - blocked: True if input was blocked (destructive/empty)
        - reason: reason for blocking (if blocked)
    """
    result = {
        "sanitized": raw,
        "rewritten": None,
        "confidence": 1.0,
        "blocked": False,
        "reason": None,
    }

    # Reject empty input
    if _reject_empty(raw):
        result["blocked"] = True
        result["reason"] = "Empty input rejected"
        return result

    # File system guardrail: block destructive commands
    if _is_destructive(raw):
        result["blocked"] = True
        result["reason"] = f"Destructive command blocked: {raw[:50]}"
        logger.warning("Blocked destructive voice command: %s", raw[:100])
        return result

    # Run VoiceInputProcessor if available
    if _VIP_AVAILABLE:
        try:
            vip_result = process_input(raw)
            result["sanitized"] = vip_result.get("sanitized", raw)
            result["confidence"] = vip_result.get("confidence", 1.0)
        except Exception as e:
            logger.warning("VoiceInputProcessor failed: %s", e)
            result["sanitized"] = raw

    # LLM rewrite if confidence is low
    if result["confidence"] < CONFIDENCE_THRESHOLD and _REWRITER_AVAILABLE:
        try:
            rewritten = llm_rewrite(result["sanitized"])
            result["rewritten"] = rewritten
            result["sanitized"] = rewritten
        except Exception as e:
            logger.warning("LLM rewrite failed: %s", e)

    return result


def pre_tool_call_hook(tool_name: str, args: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Hermes pre_tool_call hook for voice input sanitization.

    Called before tool execution. If the tool accepts voice-dictated input,
    sanitize it before passing to the tool.

    Returns modified args dict, or None if no changes needed.
    """
    # Check if this is a voice input tool call
    voice_tools = {"voice_input", "dictate", "voice_command"}
    if tool_name not in voice_tools:
        return None

    # Extract voice input from args
    voice_text = args.get("text") or args.get("input") or args.get("command")
    if not voice_text:
        return None

    # Sanitize
    result = sanitize_voice_input(voice_text)

    if result["blocked"]:
        logger.warning("Voice input blocked: %s", result["reason"])
        return {"blocked": True, "reason": result["reason"]}

    # Update args with sanitized text
    new_args = dict(args)
    if "text" in new_args:
        new_args["text"] = result["sanitized"]
    elif "input" in new_args:
        new_args["input"] = result["sanitized"]
    elif "command" in new_args:
        new_args["command"] = result["sanitized"]

    return new_args


def _voice_pre_tool_call(tool_name: str = "", args: Optional[dict[str, Any]] = None, **_: Any) -> Optional[dict[str, Any]]:
    """pre_tool_call adapter for ``pre_tool_call_hook``.

    Runs VoiceInputProcessor sanitization and returns a plugin directive:
    ``None`` to pass through, ``{"action": "block", "message"}`` to veto a
    destructive/empty voice input, or ``{"action": "modify", "args": {...}}``
    to rewrite voice-dictated tool input before execution.
    """
    outcome = pre_tool_call_hook(tool_name, args if isinstance(args, dict) else {})
    if outcome is None:
        return None
    if outcome.get("blocked"):
        return {"action": "block", "message": outcome.get("reason") or "voice input blocked"}
    if isinstance(outcome, dict) and outcome:
        return {"action": "modify", "args": outcome}
    return None


# Self-register as a Hermes ``pre_tool_call`` hook so voice-dictated tool calls
# are sanitized before execution. Mirrors plugin ``ctx.register_hook(...)``,
# which ultimately appends to the plugin manager's ``_hooks`` registry.
# The plugin manager is a process singleton, so guard against duplicate
# registration on re-import.
_voice_hook_registered = False
try:
    from hermes_cli.plugins import get_plugin_manager

    hooks = get_plugin_manager()._hooks.setdefault("pre_tool_call", [])
    if (
        not _voice_hook_registered
        and not any(
            getattr(existing, "__name__", None) == _voice_pre_tool_call.__name__
            for existing in hooks
        )
    ):
        hooks.append(_voice_pre_tool_call)
        _voice_hook_registered = True
except Exception:  # pragma: no cover - best-effort; hook remains directly callable
    logger.warning("voice_input_hook: pre_tool_call registration failed", exc_info=True)


if __name__ == "__main__":
    # Test the hook
    test_inputs = [
        "switch to shot gun",
        "rm -rf /",
        "",
        "attack zombie",
        "mk deer",
    ]
    for inp in test_inputs:
        result = sanitize_voice_input(inp)
        print(f"Input: {inp!r}")
        print(f"  Sanitized: {result['sanitized']!r}")
        print(f"  Rewritten: {result['rewritten']!r}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Blocked: {result['blocked']} ({result['reason']})")
        print()