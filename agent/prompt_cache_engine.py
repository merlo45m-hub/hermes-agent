"""Layer 1 — Prompt Cache Engine.

The single merged implementation of Layer 1 (Prompt & Context). It replaces the
two conflicting earlier versions — the repo's pre_tool_call-only hook and the
plugin's pre_llm_call game classifier — with one coherent pipeline:

    typed / transcribed text
        -> needs_articulation()   offline heuristic, no LLM, <1ms
        -> articulate()           dual-provider LLM (NVIDIA -> OpenRouter), cached
        -> context enrichment     profile + state + params + Mnemosyne memory
        -> context block          injected into the user message by the hook

Voice commands stay on the VoiceInputProcessor pipeline and are handled by the
pre_tool_call hook (voice tools), not by this engine.

Cache: in-memory LRU + on-disk JSON so articulated prompts survive restarts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_DISK_CACHE = os.environ.get(
    "LAYER1_CACHE_PATH", os.path.join(_PLUGIN_DIR, "prompt_cache.json")
)
_MAX_CACHE_ENTRIES = 1000

# --- offline heuristics ---------------------------------------------------

_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+[,.]")
_RE_MISSING_SPACE_AFTER = re.compile(r"[.!?][A-Za-z]")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")
_RE_TERMINATOR = re.compile(r"[.!?]")
# Dictation artifacts the original heuristic missed: run-on joins ("chanceAnd"),
# stuttered repeats ("chuck chuck"), and short unpunctuated garbled phrases.
# The lowercase-run guard (>=3) avoids firing on legit names like "GitHub"/"iPhone".
_RE_CAMEL_JOIN = re.compile(r"[a-z]{3,}[A-Z][a-z]")
_RE_REPEATED_WORD = re.compile(r"(?i)\b(\w+)\s+\1\b")
_RE_PATH = re.compile(r"(/(?:[\w.\-]+/)*[\w.\-]+)")
_RE_QUOTED = re.compile(r"[\"']([^\"']{2,80})[\"']")
_RE_COMMAND_VERB = re.compile(
    r"(?i)^(please\s+)?(run|execute|build|deploy|create|delete|remove|add|fix|"
    r"update|check|show|list|find|search|open|close|start|stop|restart|install|"
    r"test|compile|commit|push|pull)\b"
)


def needs_articulation(text: str) -> bool:
    """True when text looks dictated/run-on and would benefit from articulation.

    Offline and deterministic — no LLM call. Short messages and text that is
    already punctuated are left alone.
    """
    t = (text or "").strip()
    if len(t) < 20:
        return False
    if _RE_SPACE_BEFORE_PUNCT.search(t):
        return True
    if _RE_MISSING_SPACE_AFTER.search(t):
        return True
    if _RE_MULTI_SPACE.search(t):
        return True
    if _RE_CAMEL_JOIN.search(t) or _RE_REPEATED_WORD.search(t):
        return True
    if len(t.split()) >= 8 and not _RE_TERMINATOR.search(t):
        return True
    return False


def extract_params(text: str) -> Dict[str, Any]:
    """Pull structured params (intent, paths, quoted strings) out of the message."""
    params: Dict[str, Any] = {}
    t = text or ""

    if t.strip().endswith("?"):
        params["intent"] = "question"
    elif _RE_COMMAND_VERB.match(t.strip()):
        params["intent"] = "command"
    else:
        params["intent"] = "statement"

    paths = _RE_PATH.findall(t)
    quoted = _RE_QUOTED.findall(t)
    if paths:
        params["paths"] = paths[:5]
    if quoted:
        params["quoted"] = quoted[:5]
    return params


# --- cache ----------------------------------------------------------------

_cache: Dict[str, Dict[str, Any]] = {}
_cache_loaded = False


def _load_disk_cache() -> None:
    global _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    try:
        with open(_DISK_CACHE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key, value in data.items():
                if not (isinstance(value, dict) and "articulated" in value):
                    continue
                # A provider failure was cached as raw text by older versions.
                # Never resurrect one into memory: any long-lived process that
                # loaded the pre-purge file would otherwise rewrite the poisoned
                # entries back to disk on its next save.
                if value.get("provider") == "none":
                    continue
                _cache[key] = value
    except Exception:
        pass


def _save_disk_cache() -> None:
    try:
        # Defensive: drop any poisoned entry that predates this fix, even if it
        # is still held in a long-lived process's in-memory _cache.
        items = [
            (k, v) for k, v in _cache.items()
            if isinstance(v, dict) and v.get("provider") != "none"
        ][-_MAX_CACHE_ENTRIES:]
        with open(_DISK_CACHE, "w", encoding="utf-8") as f:
            json.dump(dict(items), f, ensure_ascii=False)
    except Exception:
        pass


def cache_stats() -> Dict[str, int]:
    _load_disk_cache()
    return {"entries": len(_cache), "max_entries": _MAX_CACHE_ENTRIES}


def clear_cache() -> None:
    _cache.clear()
    try:
        if os.path.exists(_DISK_CACHE):
            os.remove(_DISK_CACHE)
    except Exception:
        pass


# --- articulation ---------------------------------------------------------

def _prompt_fingerprint() -> str:
    """Hash of the articulation template, so editing the prompt invalidates the cache."""
    try:
        with open(os.path.join(_PLUGIN_DIR, "prompts", "articulate.j2"), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return "no-template"


def articulate(text: str) -> Dict[str, Any]:
    """Articulate typed text into a professional prompt. Cached by content hash."""
    _load_disk_cache()
    key = hashlib.sha256(
        (_prompt_fingerprint() + "|" + (text or "").strip().lower()).encode("utf-8")
    ).hexdigest()
    if key in _cache:
        hit = dict(_cache[key])
        hit["cache_hit"] = True
        return hit

    from voice_rewriter import articulate_prompt  # local import avoids a cycle

    result = articulate_prompt(text)
    candidate = str(result.get("articulated") or "")
    # A degenerate/truncated rewrite must never become an AUTHORITATIVE prompt.
    # Treat a long input whose output collapsed to a fraction of its length as a
    # provider miss: no cache, no injected authoritative block, retry next turn.
    if (
        result.get("provider", "none") != "none"
        and len(text) > 200
        and len(candidate) < len(text) * 0.6
    ):
        result = {
            "articulated": text,
            "provider": "none",
            "latency_ms": result.get("latency_ms", 0.0),
        }
    if result.get("provider", "none") == "none":
        # Both providers failed (or returned the input unchanged). NEVER cache
        # that: it would pin the raw garbled text as "articulated" forever, even
        # after the provider recovers. Return the miss so a later call retries.
        result["cache_hit"] = False
        return result
    entry = {
        "articulated": result.get("articulated", text),
        "provider": result.get("provider", "none"),
        "latency_ms": result.get("latency_ms", 0.0),
        "ts": time.time(),
    }
    _cache[key] = entry
    _save_disk_cache()
    entry["cache_hit"] = False
    return entry


# --- context --------------------------------------------------------------

def build_context_block(
    profile: str,
    state: Dict[str, Any],
    params: Dict[str, Any],
    memory: List[str],
) -> str:
    """Compact context line. Empty when there is nothing meaningful to add."""
    parts: List[str] = []
    task = (state or {}).get("task")
    if task:
        parts.append(f"task={task}")
    meaningful = {
        k: v for k, v in (params or {}).items() if k in ("paths", "quoted") and v
    }
    if meaningful:
        parts.append("params=" + json.dumps(meaningful, ensure_ascii=False, sort_keys=True))
    if memory:
        parts.append("memory=" + " | ".join(str(m)[:120] for m in memory[:3]))
    if not parts:
        return ""
    return f"profile={profile}; " + "; ".join(parts)


# --- pipeline -------------------------------------------------------------

def process(
    text: str,
    *,
    profile: Optional[str] = None,
    mode: str = "auto",
) -> Dict[str, Any]:
    """Full pipeline: articulate (if needed) + param extraction."""
    result: Dict[str, Any] = {
        "input": text,
        "needs_articulation": needs_articulation(text),
        "articulated": None,
        "provider": "none",
        "cache_hit": False,
        "latency_ms": 0.0,
        "profile": profile or "general",
        "params": extract_params(text),
    }
    if mode == "off":
        return result
    if mode == "always" or (mode == "auto" and result["needs_articulation"]):
        r = articulate(text)
        result["articulated"] = r.get("articulated")
        result["provider"] = r.get("provider", "none")
        result["cache_hit"] = r.get("cache_hit", False)
        result["latency_ms"] = r.get("latency_ms", 0.0)
    return result
