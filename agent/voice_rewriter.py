"""Dual-provider LLM rewriter for voice transcription correction.

Primary: NVIDIA NIM (nvidia/nemotron-3-ultra-550b-a55b)
Fallback: OpenRouter (nvidia/nemotron-3-ultra-550b-a55b:free)

Only invoked when VoiceInputProcessor confidence < 0.65.
Temperature=0, max_tokens=100, reasoning.effort=none.
HTTP timeout 10s, 1 retry, in-memory cache.

Phoenix tracing: spans exported to http://10.9.0.1:6006/v1/traces
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

# --- OpenTelemetry / Phoenix tracing ---
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode

    _PHOENIX_ENDPOINT = os.environ.get(
        "PHOENIX_ENDPOINT", "http://10.9.0.1:6006"
    )
    _resource = Resource.create({"service.name": "voice-rewriter", "service.version": "1.0"})
    _provider = TracerProvider(resource=_resource)
    _exporter = OTLPSpanExporter(endpoint=f"{_PHOENIX_ENDPOINT}/v1/traces")
    _provider.add_span_processor(SimpleSpanProcessor(_exporter))
    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer("voice-rewriter")
    _TRACING_AVAILABLE = True
except Exception as e:
    _TRACING_AVAILABLE = False
    _tracer = None
    logger.debug("Tracing not available: %s", e)

# Provider configuration
PRIMARY_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
PRIMARY_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
PRIMARY_KEY_ENV = "NVIDIA_API_KEY"

FALLBACK_URL = "https://openrouter.ai/api/v1/chat/completions"
FALLBACK_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_KEY_ENV = "OPENROUTER_API_KEY"

HTTP_TIMEOUT = 10  # seconds
MAX_RETRIES = 1
TEMPERATURE = 0
MAX_TOKENS = 100

PROMPT_TEMPLATE = (
    "You are a voice transcription corrector. Fix only phonetic spelling errors. "
    "Do not change words or meaning. Input: {raw}. Output only the corrected text, nothing else."
)

# In-memory cache: {input_hash: corrected_text}
_cache: dict[str, str] = {}


def _get_api_key(env_var: str) -> Optional[str]:
    """Get API key from environment or .env file."""
    key = os.environ.get(env_var)
    if key:
        return key
    # Try loading from ~/.hermes/.env
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{env_var}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _call_provider(url: str, model: str, api_key: str, raw: str) -> Optional[str]:
    """Call a single LLM provider. Returns corrected text or None on failure."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(raw=raw)}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "reasoning": {"effort": "none"},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            result = json.loads(resp.read())
            content = result.get("choices", [{}])[0].get("message", {}).get("content")
            if content and content.strip():
                return content.strip()
            return None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("Provider %s failed: %s", url, e)
        return None


def rewrite(raw: str) -> str:
    """Rewrite voice transcription using dual-provider LLM.

    Returns corrected text, or original input if all providers fail.
    Uses in-memory cache to avoid repeat calls.
    """
    if not raw or not raw.strip():
        return raw

    # Check cache
    cache_key = hashlib.sha256(raw.encode()).hexdigest()
    if cache_key in _cache:
        logger.debug("Cache hit for: %s", raw[:50])
        return _cache[cache_key]

    # Create tracing span
    if _TRACING_AVAILABLE and _tracer:
        with _tracer.start_as_current_span("voice_rewrite") as span:
            span.set_attribute("input", raw[:200])
            span.set_attribute("cache_hit", False)
            result = _do_rewrite(raw, cache_key)
            span.set_attribute("output", result[:200])
            span.set_attribute("provider", _last_provider)
            span.set_status(Status(StatusCode.OK))
            return result
    else:
        return _do_rewrite(raw, cache_key)


def _do_rewrite(raw: str, cache_key: str) -> str:
    """Internal rewrite implementation."""
    global _last_provider
    providers = [
        (PRIMARY_URL, PRIMARY_MODEL, PRIMARY_KEY_ENV, "NVIDIA NIM"),
        (FALLBACK_URL, FALLBACK_MODEL, FALLBACK_KEY_ENV, "OpenRouter"),
    ]

    for url, model, key_env, name in providers:
        api_key = _get_api_key(key_env)
        if not api_key:
            logger.warning("No API key for %s (%s), skipping", name, key_env)
            continue

        for attempt in range(MAX_RETRIES + 1):
            result = _call_provider(url, model, api_key, raw)
            if result is not None:
                logger.info("Rewrote via %s: '%s' -> '%s'", name, raw[:50], result[:50])
                _cache[cache_key] = result
                global _last_provider
                _last_provider = name
                return result
            if attempt < MAX_RETRIES:
                logger.info("Retrying %s (attempt %d)", name, attempt + 2)
                time.sleep(0.5)

    # All providers failed — return original
    logger.warning("All providers failed for: %s, returning original", raw[:50])
    _cache[cache_key] = raw
    _last_provider = "none"
    return raw


_last_provider: str = "none"


def clear_cache() -> None:
    """Clear the in-memory rewrite cache."""
    _cache.clear()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(rewrite(text))
    else:
        print("Usage: python voice_rewriter.py <text to rewrite>")