"""
voice_rewriter.py — Dual-provider LLM rewriter for low-confidence voice input.
Primary: NVIDIA NIM (nvidia/nemotron-3-ultra-550b-a55b)
Fallback: OpenRouter (nvidia/nemotron-3-ultra-550b-a55b:free)
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

# Attempt to load Phoenix tracing if available
try:
    if "/root/mcp-servers/servers" not in sys.path:
        sys.path.insert(0, "/root/mcp-servers/servers")
    from tracing_wrapper import init_tracing, flush_tracing
    _tracer = init_tracing()
except Exception:
    _tracer = None

    def flush_tracing() -> None:
        pass


# In-memory rewrite cache keyed by sha256(raw.lower())
_CACHE: Dict[str, Dict[str, Any]] = {}

# System prompt for the Prompt Cache Engine articulation stage (typed text).
ARTICULATE_SYSTEM_PROMPT = (
    "You are a prompt articulation engine. Rewrite the user's message into a clear, "
    "professional prompt.\n"
    "Rules:\n"
    "- Fix punctuation, grammar, spacing, and sentence structure.\n"
    "- Preserve the original meaning and intent exactly. Do not add or remove information.\n"
    "- Keep technical terms, names, file paths, and code identifiers unchanged.\n"
    "- Preserve the user's formatting style. Do not add markdown (bold, headers, bullets) unless the user used it.\n"
    "- If the message contains multiple requests, separate them into a numbered list.\n"
    "- If the input is too short or too vague to be a prompt (a greeting, a single word, a bare\n"
    "  acknowledgement), return it unchanged.\n"
    "- Output only the rewritten prompt, nothing else."
)


def _load_env_fallback() -> None:
    """Best-effort attempt to load environment variables from Hermes env files."""
    candidates = ["/root/.hermes/.env", os.path.expanduser("~/.hermes/.env")]
    for candidate in candidates:
        try:
            if os.path.isfile(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k not in os.environ:
                                os.environ[k] = v
        except Exception:
            pass


_load_env_fallback()


def _emit_trace_span(raw: str, output: str, provider: str, cache_hit: bool) -> None:
    """Emit a Phoenix tracing span named voice_rewrite if tracing is active."""
    if _tracer is not None:
        try:
            with _tracer.start_as_current_span("voice_rewrite") as span:
                span.set_attribute("input", raw)
                span.set_attribute("output", output)
                span.set_attribute("provider", provider)
                span.set_attribute("cache_hit", cache_hit)
            flush_tracing()
        except Exception:
            pass


def _clean_text(text: str) -> str:
    """Clean model response removing quotes and prefix noise."""
    res = text.strip()
    if (res.startswith('"') and res.endswith('"')) or (res.startswith("'") and res.endswith("'")):
        res = res[1:-1].strip()
    if res.lower().startswith("output:"):
        res = res[7:].strip()
    return res


def _call_nvidia_nim(
    raw: str, system_prompt: str, api_key: str,
    timeout: float = 10.0, max_tokens: int = 100,
) -> str:
    """Calls primary provider: NVIDIA NIM."""
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input: {raw}"}
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning": {"effort": "none"}
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content", "").strip()
    return ""


def _call_openrouter(
    raw: str, system_prompt: str, api_key: str,
    timeout: float = 10.0, max_tokens: int = 100,
) -> str:
    """Calls fallback provider: OpenRouter."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://layer1.vps",
        "X-Title": "Layer1 Voice Rewriter"
    }
    payload = {
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input: {raw}"}
        ],
        "temperature": 0,
        "max_tokens": max_tokens
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content", "").strip()
    return ""


def articulate_prompt(text: str, timeout: float = 10.0) -> Dict[str, Any]:
    """Articulate typed text into a professional prompt.

    Dual-provider: NVIDIA NIM primary -> OpenRouter fallback. The caller
    (prompt_cache_engine) owns caching; this function is a single uncached call.
    Returns {articulated, provider, latency_ms}.
    """
    start_time = time.perf_counter()
    raw = (text or "").strip()
    if not raw:
        return {"articulated": raw, "provider": "none", "latency_ms": 0.0}

    try:
        from prompts import load_prompt
        system_prompt = load_prompt("articulate.j2", raw_input=raw)
    except Exception:
        system_prompt = ARTICULATE_SYSTEM_PROMPT

    # articulate_mode=always means long pastes also get articulated. A fixed
    # 100-token cap would truncate them, and the AUTHORITATIVE block would then
    # tell the model to trust a truncated request. Size the budget to the input.
    max_tokens = min(2048, max(256, len(raw) // 2))

    nvidia_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if nvidia_key:
        for attempt in range(2):
            try:
                out = _call_nvidia_nim(
                    raw, system_prompt, nvidia_key, timeout=timeout, max_tokens=max_tokens
                )
                if out:
                    return {
                        "articulated": _clean_text(out),
                        "provider": "nvidia",
                        "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
                    }
            except Exception:
                if attempt == 0:
                    time.sleep(0.5)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        for attempt in range(2):
            try:
                out = _call_openrouter(
                    raw, system_prompt, openrouter_key, timeout=timeout, max_tokens=max_tokens
                )
                if out:
                    return {
                        "articulated": _clean_text(out),
                        "provider": "openrouter",
                        "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
                    }
            except Exception:
                if attempt == 0:
                    time.sleep(0.5)

    return {
        "articulated": raw,
        "provider": "none",
        "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
    }


def rewrite_voice_input(raw: str, current_confidence: float = 0.0) -> Dict[str, Any]:
    """
    Rewrites voice input to fix phonetic errors using NVIDIA NIM or OpenRouter fallback.
    ONLY invoked when VoiceInputProcessor result has confidence < 0.65.
    Returns:
      {"rewritten": str, "provider": "nvidia"|"openrouter"|"none", "cache_hit": bool, "latency_ms": float, "confidence": float}
    """
    start_time = time.perf_counter()
    raw_str = (raw or "").strip()

    # If confidence is already sufficient, or input is empty, return unchanged
    if current_confidence >= 0.65 or not raw_str:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return {
            "rewritten": raw_str,
            "provider": "none",
            "cache_hit": False,
            "latency_ms": round(elapsed, 2),
            "confidence": current_confidence
        }

    # Check in-memory cache
    cache_key = hashlib.sha256(raw_str.lower().encode("utf-8")).hexdigest()
    if cache_key in _CACHE:
        cached = _CACHE[cache_key]
        elapsed = (time.perf_counter() - start_time) * 1000.0
        _emit_trace_span(raw_str, cached["rewritten"], cached["provider"], cache_hit=True)
        return {
            "rewritten": cached["rewritten"],
            "provider": cached["provider"],
            "cache_hit": True,
            "latency_ms": round(elapsed, 2),
            "confidence": cached["confidence"]
        }

    # Build system prompt using prompts loader if available, or static fallback
    try:
        from prompts import load_prompt
        system_prompt = load_prompt("system.j2", raw_input=raw_str, role="voice transcription corrector")
    except Exception:
        system_prompt = (
            f"You are a voice transcription corrector. Fix only phonetic spelling errors. "
            f"Do not change words or meaning. Input: {raw_str}. Output only the corrected text, nothing else."
        )

    # 1. Primary: NVIDIA NIM (1 retry = 2 attempts)
    nvidia_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if nvidia_key:
        for attempt in range(2):
            try:
                rewritten_text = _call_nvidia_nim(raw_str, system_prompt, nvidia_key, timeout=10.0)
                if rewritten_text:
                    cleaned = _clean_text(rewritten_text)
                    boosted_conf = 0.85
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    _CACHE[cache_key] = {
                        "rewritten": cleaned,
                        "provider": "nvidia",
                        "confidence": boosted_conf
                    }
                    _emit_trace_span(raw_str, cleaned, "nvidia", cache_hit=False)
                    return {
                        "rewritten": cleaned,
                        "provider": "nvidia",
                        "cache_hit": False,
                        "latency_ms": round(elapsed, 2),
                        "confidence": boosted_conf
                    }
            except Exception:
                if attempt == 0:
                    time.sleep(0.5)

    # 2. Fallback: OpenRouter (1 retry = 2 attempts)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        for attempt in range(2):
            try:
                rewritten_text = _call_openrouter(raw_str, system_prompt, openrouter_key, timeout=10.0)
                if rewritten_text:
                    cleaned = _clean_text(rewritten_text)
                    boosted_conf = 0.85
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    _CACHE[cache_key] = {
                        "rewritten": cleaned,
                        "provider": "openrouter",
                        "confidence": boosted_conf
                    }
                    _emit_trace_span(raw_str, cleaned, "openrouter", cache_hit=False)
                    return {
                        "rewritten": cleaned,
                        "provider": "openrouter",
                        "cache_hit": False,
                        "latency_ms": round(elapsed, 2),
                        "confidence": boosted_conf
                    }
            except Exception:
                if attempt == 0:
                    time.sleep(0.5)

    # 3. Both failed or no keys: return original unchanged
    elapsed = (time.perf_counter() - start_time) * 1000.0
    _emit_trace_span(raw_str, raw_str, "none", cache_hit=False)
    return {
        "rewritten": raw_str,
        "provider": "none",
        "cache_hit": False,
        "latency_ms": round(elapsed, 2),
        "confidence": current_confidence
    }
