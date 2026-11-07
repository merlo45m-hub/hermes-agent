"""Layer 1 — Prompt & Context hooks (single merged implementation).

pre_llm_call   : typed / transcribed text -> Prompt Cache Engine -> context block
                 (articulation + profile/state/params/memory). Returns None when
                 there is nothing to add, so clean messages stay untouched.
pre_tool_call  : voice tools -> VoiceInputProcessor game pipeline + destructive
                 guardrail. Scoped to voice tool calls.

This file merges the two earlier conflicting implementations — the repo's
pre_tool_call-only hook and the plugin's pre_llm_call game classifier — into one
coherent layer. The game parser no longer runs on typed chat.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from prompt_cache_engine import (
    build_context_block,
    cache_stats,
    extract_params,
    process as engine_process,
)
from voice_context_engine import VoiceContextEngine

try:
    from VoiceInputProcessor import VoiceInputProcessor
except Exception:  # pragma: no cover - voice pipeline optional
    VoiceInputProcessor = None

_engine = VoiceContextEngine()
_processor = VoiceInputProcessor(enable_rate_limiting=False) if VoiceInputProcessor else None

_AUDIT = os.path.join(_DIR, "layer1_audit.jsonl")
_CONFIG_PATH = os.path.join(_DIR, "voice_context_config.yaml")

_DEFAULTS: Dict[str, Any] = {
    "articulate_mode": "auto",   # auto | always | off
    "include_context": True,
    "memory_mode": "first_turn",  # first_turn | always | off
    "profile": "general",         # profile label for typed-chat context
    "voice_tool_names": ["voice", "voice_input", "voice_command"],
}


def _load_config() -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    try:
        import yaml
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        engine_cfg = data.get("engine", {})
        if isinstance(engine_cfg, dict):
            for key, value in engine_cfg.items():
                if key in cfg:
                    cfg[key] = value
    except Exception:
        pass
    cfg["voice_tool_names"] = set(cfg.get("voice_tool_names") or [])
    return cfg


_CONFIG = _load_config()

# Mnemosyne memory is pulled at most once per TTL, not on every turn.
_MEMORY_CACHE: Dict[str, Any] = {"ts": 0.0, "value": []}
_MEMORY_TTL = 60.0


def _memory_context() -> List[str]:
    now = time.time()
    if now - _MEMORY_CACHE["ts"] > _MEMORY_TTL:
        _MEMORY_CACHE["value"] = _engine.memory_context()
        _MEMORY_CACHE["ts"] = now
    return _MEMORY_CACHE["value"]

_RE_DESTRUCTIVE = re.compile(
    r"(?i)(\brm\s+-rf\b|\bdrop\s+table\b|\bdelete\s+from\b|\bformat\s+[a-z]:|"
    r"\bmkfs\b|\bshutdown\b|\bdestroy\b|\bkill\s+-9\b)"
)
_DESTRUCTIVE_ACTIONS = {"DELETE", "DROP", "DESTROY", "FORMAT", "SHUTDOWN", "WIPE", "KILL"}


_RE_INTERNAL = re.compile(
    r"^\s*(\[IMPORTANT:|\[INTERNAL NOTIFICATION|\[USER INITIATED SKILLS|"
    r"\[Context from|\[Bundle context:|\[System:|📝 Articulated prompt)",
    re.IGNORECASE,
)


def _is_internal_traffic(text: str) -> bool:
    """Gateway notifications, curator prompts and skill dumps are NOT user dictation.

    Articulating them wastes an LLM call and can mangle structured payloads.
    """
    if _RE_INTERNAL.match(text or ""):
        return True
    return "You are running as Hermes' background skill CURATOR" in (text or "")[:200]


def _extract_text(user_message: Any) -> str:
    if isinstance(user_message, list):
        return " ".join(
            p.get("text", "") for p in user_message if isinstance(p, dict)
        ).strip()
    return str(user_message or "").strip()


def _audit(record: Dict[str, Any]) -> None:
    try:
        if os.path.exists(_AUDIT) and os.path.getsize(_AUDIT) > 2_000_000:
            os.replace(_AUDIT, _AUDIT + ".1")
        with open(_AUDIT, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def pre_llm_call_hook(
    session_id: str,
    user_message: Any,
    conversation_history: Optional[list] = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Typed / transcribed text -> articulation + context block.

    Returns None when there is nothing to add, so clean messages are untouched.
    """
    text = _extract_text(user_message)
    if not text:
        return None

    cfg = _CONFIG
    profile = cfg.get("profile") or _engine.current_profile
    # One engine for text AND voice: articulate every real user message. Skip
    # gateway/curator/skill-injection traffic, which is structured, not dictation.
    effective_mode = "off" if _is_internal_traffic(text) else cfg.get("articulate_mode", "auto")
    result = engine_process(text, profile=profile, mode=effective_mode)
    params = result.get("params") or extract_params(text)

    context = ""
    if cfg.get("include_context"):
        memory: List[str] = []
        if cfg.get("memory_mode") == "always" or (
            cfg.get("memory_mode") == "first_turn" and is_first_turn
        ):
            memory = _memory_context()
        context = build_context_block(
            profile, _engine.state, params, memory
        )

    parts: List[str] = []
    articulated = result.get("articulated")
    if articulated and articulated != text:
        parts.append(
            "📝 Articulated prompt — AUTHORITATIVE. Treat this as the user's actual "
            "request. Ignore transcription artifacts, phonetic typos and run-on "
            "joins in the raw text above; do not quote or correct the raw text.\n"
            + articulated
        )
    if context:
        parts.append("🧠 Context: " + context)

    _audit(
        {
            "ts": time.time(),
            "hook": "pre_llm_call",
            "platform": platform,
            "model": model,
            "input": text[:500],
            "needs_articulation": result.get("needs_articulation"),
            "articulated": bool(articulated and articulated != text),
            "provider": result.get("provider"),
            "cache_hit": result.get("cache_hit"),
            "context": bool(context),
            "cache_entries": cache_stats()["entries"],
        }
    )

    if not parts:
        return None
    return {"context": "\n\n".join(parts)}


def _is_destructive(command_dict: Dict[str, Any], raw_text: str) -> bool:
    action = str(command_dict.get("action", "")).upper()
    if action in _DESTRUCTIVE_ACTIONS:
        return True
    intent = str(command_dict.get("intent", "")).lower()
    if "destructive" in intent or "danger" in intent:
        return True
    sanitized = str(command_dict.get("sanitized_text", ""))
    if _RE_DESTRUCTIVE.search(sanitized) or _RE_DESTRUCTIVE.search(raw_text):
        return True
    return "DESTRUCTIVE" in (command_dict.get("flags") or [])


def pre_tool_call_hook(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    **tool_context: Any,
) -> Optional[Dict[str, Any]]:
    """Voice-tool guardrail: sanitize game/voice input, block destructive commands."""
    tool_args = args if isinstance(args, dict) else {}
    has_voice_input = (
        "raw_transcript" in tool_context
        or tool_name in _CONFIG["voice_tool_names"]
        or "raw_transcript" in tool_args
    )
    if not has_voice_input or _processor is None:
        return None

    raw_transcript = (
        tool_context.get("raw_transcript")
        or tool_args.get("raw_transcript")
        or tool_args.get("input")
        or ""
    )

    proc_result = _processor.process(raw_transcript)

    if proc_result.get("confidence", 0.0) < 0.65:
        from voice_rewriter import rewrite_voice_input

        rewrite_result = rewrite_voice_input(
            raw_transcript, proc_result.get("confidence", 0.0)
        )
        if rewrite_result.get("provider") != "none" and rewrite_result.get("rewritten"):
            rewritten_text = rewrite_result["rewritten"]
            if rewritten_text != raw_transcript:
                proc_result = _processor.process(rewritten_text)
                proc_result["flags"] = list(proc_result.get("flags", [])) + ["LLM_REWRITTEN"]
                proc_result["llm_rewrite"] = rewrite_result

    enriched_result = _engine.process(proc_result)

    if _is_destructive(enriched_result, raw_transcript):
        profile_name = _engine.current_profile
        detail = raw_transcript.strip()[:60] or enriched_result.get("action", "UNKNOWN")
        return {
            "action": "block",
            "message": f"Destructive command blocked by {profile_name} profile policy: {detail}",
        }

    merged_args = dict(tool_args)
    merged_args["voice_command"] = enriched_result
    merged_args["sanitized_text"] = enriched_result.get("sanitized_text", raw_transcript)
    merged_args["action"] = enriched_result.get("action")
    merged_args["parameters"] = enriched_result.get("parameters", {})
    merged_args["confidence"] = enriched_result.get("confidence")

    return {"action": "modify", "args": merged_args}
