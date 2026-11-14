"""Layer 5 orchestration tool: runs Hermes' multi-agent state machine for a task.

Wraps ``agent/orchestration_engine.run_orchestration`` as a callable Hermes gateway
tool. The orchestration engine classifies the task, routes it to the agent whose
ROLE owns that task type (agy / codex / claude), dispatches, verifies and
(optionally) retries through a LangGraph state machine, returning the final state
dict as JSON.

Fleet roles: agy = Android/Godot/heavy compute (VPS), codex = complex coding and
synthesis (Termux), claude = security review and research (Termux). General LLM
queries that need no agent tooling stay on the gateway. Agents in backoff
(quota 429, transport failure) are skipped automatically with exponential
backoff; the run state reports which agents were skipped and why.
"""

import asyncio
import json

from tools.registry import registry, tool_error


ORCHESTRATE_SCHEMA = {
    "name": "orchestrate",
    "description": (
        "Run Hermes Layer 5 orchestration for a task: classify it, route to the "
        "agent whose role owns that task type (agy = Android/Godot/heavy compute "
        "on the VPS, codex = complex coding and synthesis on Termux, claude = "
        "security review and research on Termux), dispatch, verify and retry "
        "through the multi-agent state machine. Agents in backoff (quota 429, "
        "transport failure) are skipped automatically with exponential backoff. "
        "Use this for work that needs an agent's tools; route general LLM queries "
        "through the gateway instead. Returns the final orchestration state as JSON."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task to orchestrate (free-form request).",
            },
            "max_retries": {
                "type": "integer",
                "description": "Maximum number of orchestration retries (default 2).",
                "default": 2,
            },
        },
        "required": ["task"],
        "additionalProperties": False,
    },
}


def orchestrate_tool(args: dict) -> str:
    """Execute Layer 5 orchestration for *task*, returning the result dict as JSON."""
    task = (args.get("task") or "").strip()
    if not task:
        return tool_error("task is required")
    max_retries = args.get("max_retries", 2)
    try:
        max_retries = int(max_retries)
    except (TypeError, ValueError):
        return tool_error("max_retries must be an integer")
    try:
        from agent.orchestration_engine import run_orchestration

        result = run_orchestration(task, max_retries=max_retries)
    except Exception as exc:  # fail-open: surface a JSON error, never crash the gateway
        return tool_error(f"orchestration failed: {exc}")
    return json.dumps(result, ensure_ascii=False, default=str)


async def orchestrate_handler(args: dict, **kw) -> str:
    """Async-safe handler: validation then blocking orchestration on a worker thread.

    Mirrors ``orchestrate_tool`` (lazy import + fail-open) but awaits
    ``asyncio.to_thread(run_orchestration, task, max_retries)`` so the gateway
    event loop is never frozen by the LangGraph run.
    """
    task = (args.get("task") or "").strip()
    if not task:
        return tool_error("task is required")
    max_retries = args.get("max_retries", 2)
    try:
        max_retries = int(max_retries)
    except (TypeError, ValueError):
        return tool_error("max_retries must be an integer")
    try:
        from agent.orchestration_engine import run_orchestration

        result = await asyncio.to_thread(run_orchestration, task, max_retries=max_retries)
    except Exception as exc:  # fail-open: surface a JSON error, never crash the gateway
        return tool_error(f"orchestration failed: {exc}")
    return json.dumps(result, ensure_ascii=False, default=str)


registry.register(
    name="orchestrate",
    toolset="delegation",
    schema=ORCHESTRATE_SCHEMA,
    handler=orchestrate_handler,
    is_async=True,
    emoji="\U0001f3af",  # 🎯
)
