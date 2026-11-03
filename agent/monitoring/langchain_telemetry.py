"""LangChain telemetry integration for Hermes-Agent with Langfuse and Arize Phoenix.

Non-blocking async export with batch flush. Multi-step reasoning, tool latency,
constraint adherence tracking. OpenRouter/Claude primary, agy fallback.
"""

from __future__ import annotations

import logging
import os
import time
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from queue import Queue, Full

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tracers import BaseTracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "https://app.phoenix.arize.com/v1/traces")

PRIMARY_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "openrouter")
FALLBACK_PROVIDER = os.getenv("FALLBACK_LLM_PROVIDER", "agy")

BATCH_SIZE = int(os.getenv("TELEMETRY_BATCH_SIZE", "50"))
FLUSH_INTERVAL = float(os.getenv("TELEMETRY_FLUSH_INTERVAL", "5.0"))
MAX_QUEUE_SIZE = int(os.getenv("TELEMETRY_MAX_QUEUE_SIZE", "10000"))


# ---------------------------------------------------------------------------
# Async Export Queue
# ---------------------------------------------------------------------------

@dataclass
class TelemetryEvent:
    event_type: str  # "trace", "generation", "span"
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class AsyncTelemetryQueue:
    """Non-blocking async queue with batch flush for telemetry exports."""

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        flush_interval: float = FLUSH_INTERVAL,
        max_queue_size: int = MAX_QUEUE_SIZE,
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size
        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background flush thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._flush_loop, name="langchain-telemetry-flush", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background flush thread and flush remaining events."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.flush_all()

    def enqueue(self, event: TelemetryEvent) -> bool:
        """Add event to queue. Returns False if queue is full (non-blocking)."""
        try:
            self._queue.put_nowait(event)
            return True
        except Full:
            logger.warning("Telemetry queue full, dropping event")
            return False

    def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running or not self._queue.empty():
            self._stop_event.wait(self.flush_interval)
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Flush a batch of events."""
        events: List[TelemetryEvent] = []
        for _ in range(self.batch_size):
            try:
                events.append(self._queue.get_nowait())
            except Exception:
                break

        if not events:
            return

        # Export to Langfuse
        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
            self._export_langfuse(events)

        # Export to Phoenix
        if PHOENIX_API_KEY:
            self._export_phoenix(events)

    def _export_langfuse(self, events: List[TelemetryEvent]) -> None:
        """Export events to Langfuse."""
        try:
            from langfuse import Langfuse
            langfuse = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )
            for event in events:
                if event.event_type == "trace":
                    langfuse.trace(**event.data)
                elif event.event_type == "generation":
                    langfuse.generation(**event.data)
                elif event.event_type == "span":
                    langfuse.span(**event.data)
            langfuse.flush()
        except Exception as e:
            logger.warning(f"Langfuse export failed: {e}")

    def _export_phoenix(self, events: List[TelemetryEvent]) -> None:
        """Export events to Arize Phoenix."""
        try:
            import requests
            headers = {"Authorization": f"Bearer {PHOENIX_API_KEY}", "Content-Type": "application/json"}
            spans = [e.data for e in events if e.event_type == "span"]
            if spans:
                requests.post(PHOENIX_ENDPOINT, json={"spans": spans}, headers=headers, timeout=5)
        except Exception as e:
            logger.warning(f"Phoenix export failed: {e}")

    def flush_all(self) -> None:
        """Flush all queued events."""
        while not self._queue.empty():
            self._flush_batch()


# Global queue instance
_telemetry_queue = AsyncTelemetryQueue()


# ---------------------------------------------------------------------------
# LangChain Callback Handler
# ---------------------------------------------------------------------------

class TelemetryCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for telemetry with multi-step reasoning,
    tool latency, and constraint adherence tracking.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.tags = tags or []
        self.metadata = metadata or {}
        self._trace_id: Optional[str] = None
        self._span_stack: List[str] = []
        self._step_count = 0
        self._tool_latencies: Dict[str, float] = {}
        self._constraint_violations: List[Dict[str, Any]] = []

    @property
    def trace_id(self) -> str:
        if self._trace_id is None:
            import uuid
            self._trace_id = str(uuid.uuid4())
        return self._trace_id

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Track chain start - multi-step reasoning."""
        self._step_count += 1
        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "parent_span_id": parent_run_id,
            "name": serialized.get("name", "chain"),
            "start_time": self._now(),
            "metadata": {
                **self.metadata,
                **(metadata or {}),
                "step_number": self._step_count,
                "session_id": self.session_id,
                "user_id": self.user_id,
                "tags": self.tags + (tags or []),
            },
            "attributes": {
                "input": str(inputs)[:5000],
                "component_type": "chain",
            },
        }
        self._span_stack.append(run_id)
        _telemetry_queue.enqueue(TelemetryEvent("span", span_data))

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track chain end."""
        if run_id in self._span_stack:
            self._span_stack.remove(run_id)
        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "output": str(outputs)[:5000],
                "status": "success",
            },
        }
        _telemetry_queue.enqueue(TelemetryEvent("span", span_data))

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track chain error."""
        if run_id in self._span_stack:
            self._span_stack.remove(run_id)
        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "error": str(error),
                "status": "error",
            },
        }
        _telemetry_queue.enqueue(TelemetryEvent("span", span_data))

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Track LLM start - provider info, model config."""
        provider = metadata.get("provider", PRIMARY_PROVIDER) if metadata else PRIMARY_PROVIDER
        model = serialized.get("kwargs", {}).get("model_name", "unknown")

        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "parent_span_id": parent_run_id,
            "name": f"llm:{model}",
            "start_time": self._now(),
            "metadata": {
                **self.metadata,
                **(metadata or {}),
                "provider": provider,
                "model": model,
                "fallback_provider": FALLBACK_PROVIDER,
                "session_id": self.session_id,
                "user_id": self.user_id,
            },
            "attributes": {
                "input": "\n".join(prompts)[:10000],
                "component_type": "llm",
            },
        }
        self._span_stack.append(run_id)
        _telemetry_queue.enqueue(TelemetryEvent("span", span_data))

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track LLM end - token usage, latency, reasoning."""
        if run_id in self._span_stack:
            self._span_stack.remove(run_id)

        token_usage = {}
        if response.llm_output:
            token_usage = {
                "prompt_tokens": response.llm_output.get("token_usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.llm_output.get("token_usage", {}).get("completion_tokens", 0),
                "total_tokens": response.llm_output.get("token_usage", {}).get("total_tokens", 0),
            }

        reasoning = ""
        for gen in response.generations:
            for g in gen:
                if hasattr(g, "message") and hasattr(g.message, "additional_kwargs"):
                    reasoning = g.message.additional_kwargs.get("reasoning", "") or ""

        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "output": str(response)[:10000],
                "token_usage": token_usage,
                "reasoning_length": len(reasoning),
                "has_reasoning": bool(reasoning),
                "status": "success",
            },
        }
        _telemetry_queue.enqueue(TelemetryEvent("span", span_data))

        _telemetry_queue.enqueue(TelemetryEvent("generation", {
            "trace_id": self.trace_id,
            "name": "llm_generation",
            "start_time": self._now() - 1,
            "end_time": self._now(),
            "model": serialized.get("kwargs", {}).get("model_name", "unknown") if "serialized" in locals() else "unknown",
            "input": str(prompts) if "prompts" in locals() else "",
            "output": str(response),
            "usage": token_usage,
            "metadata": {"provider": PRIMARY_PROVIDER, "fallback": FALLBACK_PROVIDER},
        }))

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track LLM error - fallback trigger."""
        if run_id in self._span_stack:
            self._span_stack.remove(run_id)
        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "error": str(error),
                "status": "error",
                "fallback_triggered": True,
            },
        }
        _telemetry_queue.enqueue(TelemetryEvent("span", span_data))

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Track tool start - latency measurement."""
        self._tool_latencies[run_id] = time.time()

        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "parent_span_id": parent_run_id,
            "name": f"tool:{serialized.get('name', 'unknown')}",
            "start_time": self._now(),
            "metadata": {
                **self.metadata,
                **(metadata or {}),
                "tool_name": serialized.get("name"),
                "session_id": self.session_id,
                "user_id": self.user_id,
            },
            "attributes": {
                "input": input_str[:5000],
                "component_type": "tool",
            },
        }
        self._span_stack.append(run_id)
        _telemetry_queue.enqueue(TelemetryEvent("span", span_data))

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track tool end - latency, constraint adherence."""
        start_time = self._tool_latencies.pop(run_id, time.time())
        latency_ms = (time.time() - start_time) * 1000

        if run_id in self._span_stack:
            self._span_stack.remove(run_id)

        constraint_check = self._check_constraints(serialized.get("name") if "serialized" in locals() else "unknown", output)

        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "output": output[:5000],
                "latency_ms": latency_ms,
                "constraint_adherence": constraint_check,
                "status": "success" if constraint_check.get("passed", True) else "constraint_violation",
            },
        }
        _telemetry_queue.enqueue(TelemetryEvent("span", span_data))

        if not constraint_check.get("passed", True):
            self._constraint_violations.append({
                "tool": serialized.get("name") if "serialized" in locals() else "unknown",
                "violation": constraint_check,
                "timestamp": self._now(),
            })

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Track tool error."""
        start_time = self._tool_latencies.pop(run_id, time.time())
        latency_ms = (time.time() - start_time) * 1000

        if run_id in self._span_stack:
            self._span_stack.remove(run_id)
        span_data = {
            "trace_id": self.trace_id,
            "span_id": run_id,
            "end_time": self._now(),
            "attributes": {
                "error": str(error),
                "latency_ms": latency_ms,
                "status": "error",
            },
        }
        _telemetry_queue.enqueue(TelemetryEvent("span", span_data))

    def _check_constraints(self, tool_name: str, output: str) -> Dict[str, Any]:
        """Check constraint adherence for tool output."""
        checks = {"passed": True, "checks": {}}

        if len(output) > 100000:
            checks["passed"] = False
            checks["checks"]["output_length"] = {"passed": False, "value": len(output), "limit": 100000}

        if tool_name == "search" and not output.strip():
            checks["passed"] = False
            checks["checks"]["empty_results"] = {"passed": False}

        return checks

    def _now(self) -> float:
        return time.time()

    def get_constraint_violations(self) -> List[Dict[str, Any]]:
        return self._constraint_violations.copy()


# ---------------------------------------------------------------------------
# Tracer for LangChain
# ---------------------------------------------------------------------------

class TelemetryTracer(BaseTracer):
    """LangChain tracer that integrates with our telemetry system."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._handler = TelemetryCallbackHandler()

    @property
    def handler(self) -> TelemetryCallbackHandler:
        return self._handler

    def _persist_run(self, run: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def get_telemetry_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> TelemetryCallbackHandler:
    return TelemetryCallbackHandler(session_id=session_id, user_id=user_id, tags=tags, metadata=metadata)


def get_telemetry_tracer(**kwargs: Any) -> TelemetryTracer:
    return TelemetryTracer(**kwargs)


def is_telemetry_enabled() -> bool:
    return bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY) or bool(PHOENIX_API_KEY)


def start_telemetry() -> None:
    """Start the telemetry background thread."""
    _telemetry_queue.start()


def stop_telemetry() -> None:
    """Stop the telemetry background thread."""
    _telemetry_queue.stop()


def flush_telemetry() -> None:
    """Flush all pending telemetry."""
    _telemetry_queue.flush_all()