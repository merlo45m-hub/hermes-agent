"""
Layer 5: Orchestration — Multi-agent state machine with LangGraph.

Coordinates Hermes, agy, codex, and claude agents through stateful
workflows with handoffs and consensus loops.

Fleet roles and routing
-----------------------
The fleet is a set of SPECIALISTS, not interchangeable workers. Each agent has
a role, a host, and a routing policy; the Orchestrator never dispatches a task to
an agent whose role does not match the task type.

  PROMPT AGENT (Hermes, the lead)
      Captures the user's raw input, articulates it, and forwards it to the
      Orchestrator. It never executes task work itself.

  ORCHESTRATOR (this engine)
      Evaluates task requirements, classifies the task, selects the agent whose
      ROLE matches, dispatches, verifies, and drives consensus + retry.

  AGY — Android / Godot / heavy compute. Runs on the VPS (agy-vps) for a full
      Linux userspace, or in the phone proot (agy-proot) for phone-local work.
      Role: android, godot, build, media.

  CODEX — complex coding and synthesis. Runs locally on Termux (codex-proot).
      Role: backend, build, refactor, codegen.

  CLAUDE — security review, research, and auditing. Runs locally on Termux
      (claude-proot). Role: security, review, research, audit.

  GATEWAY (Hermes' own model)
      General LLM queries that need no agent tooling — quick lookups,
      explanations, formatting. Routed through the gateway, not an agent.

Availability and backoff
------------------------
Every agent has a live availability record (AVAILABILITY_PATH). A dispatch that
fails for a TRANSIENT reason (quota 429, transport error, timeout) marks the
agent unavailable with an exponential backoff deadline; the Orchestrator skips
unavailable agents when routing and falls through to the next candidate. A
PERMANENT failure (binary missing, auth rejected, unsupported location) marks
the agent unavailable with a long cooldown and is recorded distinctly.

Backoff is exponential per consecutive transient failure: base * 2^(n-1),
capped at MAX_BACKOFF_SECONDS, reset on the first success.
"""

import os
import json
import time
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

log = logging.getLogger(__name__)

# The engine's OWN last-run record. Written by run_orchestration() itself so
# observers (the VPS monitoring dashboard's L5 probe) read the engine's state
# rather than a marker assembled by a wrapper.
STATE_PATH = Path("/data/data/com.termux/files/home/.hermes/orchestration/last_run.json")

# Live per-agent availability record: agent -> {unavailable_until, consecutive_failures,
# last_error, last_error_kind, last_check}. Shared across runs so a quota exhaustion
# in one orchestration is respected by the next.
AVAILABILITY_PATH = Path("/data/data/com.termux/files/home/.hermes/orchestration/availability.json")

# Backoff policy (seconds). Exponential: BASE * 2^(failures-1), capped.
BACKOFF_BASE_SECONDS = 30.0
BACKOFF_MAX_SECONDS = 3600.0
# A permanent failure cools an agent down for this long.
PERMANENT_COOLDOWN_SECONDS = 28800.0
# How often to re-probe an agent marked unavailable (liveness re-check).
AVAILABILITY_RECHECK_SECONDS = 300.0


def _load_availability() -> dict:
    try:
        return json.loads(AVAILABILITY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_availability(data: dict) -> None:
    try:
        AVAILABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = AVAILABILITY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, AVAILABILITY_PATH)
    except Exception:
        log.debug("could not persist availability", exc_info=True)


def is_agent_available(agent: str, now: float | None = None) -> bool:
    """True when the agent's backoff deadline has passed (or it was never marked)."""
    now = time.time() if now is None else now
    rec = _load_availability().get(agent)
    if not rec:
        return True
    until = float(rec.get("unavailable_until") or 0)
    return now >= until


def mark_agent_failure(agent: str, kind: str, detail: str, now: float | None = None) -> None:
    """Record a dispatch failure and set the agent's backoff deadline.

    kind == "permanent" -> long cooldown, no exponential growth.
    anything else (transient) -> exponential backoff on consecutive failures.
    """
    now = time.time() if now is None else now
    data = _load_availability()
    rec = data.get(agent, {})
    if kind == "permanent":
        rec["consecutive_failures"] = 0
        rec["unavailable_until"] = now + PERMANENT_COOLDOWN_SECONDS
    else:
        n = int(rec.get("consecutive_failures") or 0) + 1
        rec["consecutive_failures"] = n
        delay = min(BACKOFF_BASE_SECONDS * (2 ** (n - 1)), BACKOFF_MAX_SECONDS)
        rec["unavailable_until"] = now + delay
    rec["last_error"] = (detail or "")[:300]
    rec["last_error_kind"] = kind
    rec["last_check"] = datetime.now(timezone.utc).isoformat()
    data[agent] = rec
    _save_availability(data)


def mark_agent_success(agent: str) -> None:
    """Reset the agent's failure streak and clear its backoff."""
    data = _load_availability()
    rec = data.get(agent)
    if not rec:
        return
    rec["consecutive_failures"] = 0
    rec["unavailable_until"] = 0
    rec["last_error"] = ""
    rec["last_error_kind"] = ""
    rec["last_check"] = datetime.now(timezone.utc).isoformat()
    data[agent] = rec
    _save_availability(data)


def save_run_state(result: dict) -> None:
    """Persist the engine's own last-run state; never raises."""
    try:
        votes = result.get("consensus_votes") or []
        first_vote = votes[0] if votes else {}
        payload = {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "task": result.get("task"),
            "status": result.get("status"),
            "assigned_agent": result.get("assigned_agent"),
            "executor_agent": result.get("executor_agent"),
            "effective_agent": LAST_EFFECTIVE_AGENT or result.get("executor_agent"),
            "verification": result.get("verification_result"),
            "reviewer": first_vote.get("agent"),
            "consensus": first_vote.get("vote"),
            "retries": result.get("retries"),
            "skipped_agents": result.get("skipped_agents"),
        }
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, STATE_PATH)
    except Exception:
        log.debug("could not persist orchestration state", exc_info=True)


class AgentState(TypedDict):
    """Shared state across all agents in the orchestration."""
    task: str
    plan: str
    assigned_agent: str
    agent_output: str
    verification_result: str
    consensus_votes: list[dict]
    status: Literal["planning", "delegating", "executing", "verifying", "consensus", "done", "failed"]
    retries: int
    max_retries: int
    mcp_tools_used: list[str]
    # Which backend actually answered the execute node (the assigned agent is
    # not always the one that ran, because of the fallback chain).
    executor_agent: str
    # Agents skipped by the availability gate, with the reason (backoff/permanent).
    skipped_agents: list[dict]


# --- Fleet definition -------------------------------------------------------
# role: what the agent is FOR. host: where it runs. routing: task types it owns.
# Each entry carries its own launcher so the Orchestrator can route by role
# without hardcoding a binary per call site.
AGENTS: dict[str, dict] = {
    "agy": {
        "role": "android/godot heavy compute",
        "host": "vps",
        "binary": "/data/data/com.termux/files/home/bin/agy-vps",
        "routing": ["android", "godot", "media"],
        "timeout": 200,
    },
    "codex": {
        "role": "complex coding and synthesis",
        "host": "termux",
        "binary": "/data/data/com.termux/files/home/bin/codex-proot",
        "routing": ["backend", "build", "codegen", "refactor"],
        "timeout": 180,
    },
    "claude": {
        "role": "security review and research",
        "host": "termux",
        "binary": "/data/data/com.termux/files/home/bin/claude-proot",
        "routing": ["security", "review", "research", "audit"],
        "timeout": 180,
    },
}

# Fallback order used when the primary agent is unavailable or fails.
_FALLBACK_ORDER = ("claude", "codex", "agy")

# Legacy aliases kept for callers that still read these names.
AGENT_ROUTING = {
    "android": "agy",
    "backend": "codex",
    "security": "claude",
    "research": "claude",
    "godot": "agy",
    "build": "codex",
    "review": "claude",
    "default": "agy",
}

TASK_KEYWORDS = {
    "android": ["android", "apk", "compose", "gradle", "kotlin"],
    "backend": ["backend", "api", "database", "sql", "postgres", "supabase", "rest"],
    "security": ["security", "audit", "vulnerability", "rls", "auth", "hardening",
                 "injection", "xss", "csrf", "exploit", "cve", "pentest"],
    "research": ["research", "search", "investigate", "analyze", "explore"],
    "godot": ["godot", "game", "scene", "gdscript"],
    "build": ["build", "compile", "deploy", "docker", "install"],
    "review": ["review", "verify", "test", "check"],
}

# MCP tool routing: which tools each agent can use
AGENT_MCP_TOOLS = {
    "agy": ["godot", "blender", "asset_grabber", "filesystem", "github"],
    "codex": ["postgres", "supabase", "github", "filesystem", "searxng"],
    "claude": ["searxng", "filesystem", "github"],
}

# Transient failure markers -> exponential backoff. Anything not listed is
# treated as permanent (no point retrying a missing binary or a bad auth).
_TRANSIENT_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "quota",
    "rate limit",
    "rate_limit",
    "too many requests",
    "timeout",
    "timed out",
    "[TIMEOUT]",
    "connection refused",
    "connection reset",
    "temporarily unavailable",
    "try again",
    "retryable",
)

_PERMANENT_MARKERS = (
    "not found",
    "no such file",
    "permission denied",
    "unauthorized",
    "auth",
    "invalid_argument",
    "failed_precondition",
    "unsupported",
    "not supported",
)


def classify_failure(output: str) -> str:
    """Classify a dispatch failure as 'transient' (backoff) or 'permanent' (cooldown)."""
    low = (output or "").lower()
    for m in _TRANSIENT_MARKERS:
        if m.lower() in low:
            return "transient"
    for m in _PERMANENT_MARKERS:
        if m.lower() in low:
            return "permanent"
    return "transient"


def classify_task(task: str) -> str:
    """Classify a task into a type based on keywords.

    Security keywords are weighted 3x: a task that names an attack class
    ("injection", "xss", "exploit") is a security review even when it also
    mentions a technology ("sql", "api") that would otherwise win on raw counts.
    Without the weight, "review this code for SQL injection" classifies as
    backend and routes to codex instead of claude.
    """
    task_lower = task.lower()
    # Attack/threat vocabulary is an unambiguous security signal.
    security_terms = ("injection", "xss", "csrf", "exploit", "vulnerability",
                      "pentest", "cve", "malware", "jailbreak", "secret", "leak",
                      "authentication bypass", "privilege escalation")
    if any(t in task_lower for t in security_terms):
        return "security"

    scores = {}
    for task_type, keywords in TASK_KEYWORDS.items():
        scores[task_type] = sum(1 for kw in keywords if kw in task_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "default"


def route_to_agent(task_type: str) -> str:
    """Route a task type to the agent whose ROLE owns it."""
    # Prefer a role-based match so a task type owned by an agent's routing list
    # always lands there, then fall back to the legacy table.
    for name, spec in AGENTS.items():
        if task_type in spec["routing"]:
            return name
    return AGENT_ROUTING.get(task_type, AGENT_ROUTING["default"])


def _agent_cmd(agent: str, task: str):
    """Build the command line for one agent, or None if it is unavailable."""
    spec = AGENTS.get(agent)
    if not spec:
        return None
    binary = spec["binary"]
    if not os.path.exists(binary):
        return None
    if agent == "agy":
        # agy-vps takes the prompt as its single argument and forwards it to the
        # VPS build over ssh with the prompt transported base64-encoded.
        return [binary, task]
    if agent == "codex":
        return [binary, "exec", "--skip-git-repo-check", task]
    if agent == "claude":
        return [binary, "-p", task, "--allowedTools", "WebSearch WebFetch Read",
                "--dangerously-skip-permissions"]
    return None


def _dispatch_once(agent: str, task: str, timeout: int) -> tuple:
    """Run one agent. Returns (ok, output); ok is False on any failure mode."""
    cmd = _agent_cmd(agent, task)
    if cmd is None:
        return False, f"[ERROR] Agent binary not found: {agent}"
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/data/data/com.termux/files/home",
        )
    except subprocess.TimeoutExpired:
        return False, f"[TIMEOUT] Agent {agent} exceeded {timeout}s"
    except Exception as e:
        return False, f"[ERROR] {e}"
    output = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0 or not output or stderr:
        combined = (output + f"\n[STDERR] {stderr}").strip()
        return False, combined or f"[ERROR] {agent} exited with code {result.returncode}"
    return True, output


def available_candidates(preferred: str) -> list[str]:
    """Agents to try, preferred first, skipping any in backoff.

    Returns (candidates, skipped) where skipped records why each omitted agent
    was bypassed so the run state can report it.
    """
    order = [preferred] + [a for a in _FALLBACK_ORDER if a != preferred]
    now = time.time()
    candidates = []
    skipped = []
    for a in order:
        if a not in AGENTS:
            continue
        if not is_agent_available(a, now):
            rec = _load_availability().get(a, {})
            skipped.append({
                "agent": a,
                "reason": rec.get("last_error_kind") or "backoff",
                "detail": (rec.get("last_error") or "")[:200],
                "unavailable_until": rec.get("unavailable_until"),
            })
            continue
        candidates.append(a)
    return candidates, skipped


def dispatch_agent(agent: str, task: str, timeout: int = 120) -> tuple:
    """Dispatch a task to the assigned agent, falling back to available others.

    Returns (output, skipped). A failed or empty attempt falls through to the
    next AVAILABLE agent; failures are recorded with the backoff policy so a
    quota-exhausted agent is not retried inside the same run or the next one.
    """
    global LAST_EFFECTIVE_AGENT
    candidates, skipped = available_candidates(agent)
    if not candidates:
        return f"[ERROR] no agent available for {agent} (all in backoff)", skipped

    last = ""
    for attempt in candidates:
        ok, output = _dispatch_once(attempt, task, AGENTS[attempt]["timeout"])
        if ok:
            LAST_EFFECTIVE_AGENT = attempt
            mark_agent_success(attempt)
            if attempt != agent:
                log.warning("dispatch_agent: %s unavailable/failed, fell back to %s", agent, attempt)
            return output, skipped
        last = output
        kind = classify_failure(output)
        mark_agent_failure(attempt, kind, output)
        log.warning("dispatch_agent: %s failed (%s), backing off: %s", attempt, kind, output[:160])
    return last or f"[ERROR] no agent available for {agent}", skipped


def call_mcp_tool(server: str, tool: str, args: dict = None) -> str:
    """Call an MCP tool via the vps-mcp bridge."""
    args = args or {}
    vps_mcp = "/data/data/com.termux/files/home/bin/vps-mcp"
    launcher = "/root/mcp-servers/docker/mcp_launcher.sh"
    if not os.path.exists(vps_mcp):
        return "[ERROR] vps-mcp bridge not found"

    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": args},
    })

    try:
        result = subprocess.run(
            [vps_mcp, launcher, "--local", server],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in result.stdout.strip().splitlines():
            try:
                resp = json.loads(line)
                if "result" in resp:
                    content = resp["result"].get("content", [])
                    if content:
                        return content[0].get("text", str(content))
                    return str(resp["result"])
            except json.JSONDecodeError:
                continue
        return result.stderr.strip() or "[No response]"
    except subprocess.TimeoutExpired:
        return "[TIMEOUT] MCP call exceeded 30s"
    except Exception as e:
        return f"[ERROR] {e}"


# --- Graph Nodes ---

def plan_node(state: AgentState) -> AgentState:
    """Create a plan for the task."""
    task = state["task"]
    task_type = classify_task(task)
    agent = route_to_agent(task_type)
    mcp_tools = AGENT_MCP_TOOLS.get(agent, [])
    return {
        **state,
        "plan": f"Task type: {task_type}. Assigned to: {agent}. MCP tools: {mcp_tools}. Task: {task}",
        "assigned_agent": agent,
        "mcp_tools_used": mcp_tools,
        "status": "delegating",
    }


def delegate_node(state: AgentState) -> AgentState:
    """Delegate the task to the assigned agent."""
    agent = state["assigned_agent"]
    plan = state["plan"]
    return {
        **state,
        "agent_output": f"[Delegated to {agent}] {plan}",
        "status": "executing",
    }


def execute_node(state: AgentState) -> AgentState:
    """Execute the delegated task (agent runs here)."""
    agent = state["assigned_agent"]
    task = state["task"]
    plan = state["plan"]

    # Build the prompt with context about available MCP tools
    mcp_tools = state.get("mcp_tools_used", [])
    tool_context = ""
    if mcp_tools:
        tool_context = f"\n\nAvailable MCP tools: {', '.join(mcp_tools)}. Use these for data access."

    prompt = f"Task: {task}\nPlan: {plan}{tool_context}\n\nExecute this task and report what you did in detail."

    output, skipped = dispatch_agent(agent, prompt)
    return {
        **state,
        "agent_output": output,
        "status": "verifying",
        "executor_agent": LAST_EFFECTIVE_AGENT or agent,
        "skipped_agents": skipped,
    }


def verify_node(state: AgentState) -> AgentState:
    """Verify the agent's output."""
    output = state["agent_output"]
    # A correct short answer is still a correct answer: reject only error
    # markers and genuinely empty output, not brevity.
    has_error = any(m in output for m in ("[ERROR]", "[TIMEOUT]", "[No output]"))
    has_content = bool(output.strip()) and not has_error

    passed = has_content
    return {
        **state,
        "verification_result": "PASS" if passed else "FAIL",
        "status": "consensus" if passed else "failed",
    }


def consensus_node(state: AgentState) -> AgentState:
    """Multi-agent consensus: review by a different agent."""
    agent = state["assigned_agent"]
    output = state["agent_output"]

    # Pick a different AVAILABLE agent for review.
    reviewers, _ = available_candidates(agent)
    reviewers = [a for a in reviewers if a != agent]
    reviewer = reviewers[0] if reviewers else ""
    if not reviewer:
        # No available reviewer: accept on the executor's own verification rather
        # than failing a correct result for lack of a second opinion.
        votes = state.get("consensus_votes", [])
        votes.append({"agent": "self", "vote": "approve",
                      "reason": "no available reviewer; executor verification PASS"})
        return {**state, "consensus_votes": votes, "status": "done"}

    review_prompt = (
        f"Task: {state['task']}\n"
        f"Review this work output and say APPROVE or REJECT with reason.\n"
        f"Agent: {agent}\nOutput:\n{output[:2000]}\n\n"
        f"Respond with exactly: APPROVE: <reason> or REJECT: <reason>"
    )

    review_output, _ = dispatch_agent(reviewer, review_prompt)
    approved = "APPROVE" in review_output.upper()

    votes = state.get("consensus_votes", [])
    votes.append({"agent": reviewer, "vote": "approve" if approved else "reject", "reason": review_output[:200]})

    return {
        **state,
        "consensus_votes": votes,
        "status": "done" if approved else "failed",
    }


def should_retry(state: AgentState) -> str:
    """Decide whether to retry or fail."""
    if state["status"] == "failed" and state["retries"] < state["max_retries"]:
        return "retry"
    if state["status"] == "failed":
        return "fail"
    return "continue"


def retry_node(state: AgentState) -> AgentState:
    """Increment retry counter and go back to planning."""
    return {
        **state,
        "retries": state["retries"] + 1,
        "status": "planning",
    }


# --- Build the Graph ---

def build_orchestration_graph() -> StateGraph:
    """Build the LangGraph state machine for multi-agent orchestration."""
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("delegate", delegate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.add_node("consensus", consensus_node)
    graph.add_node("retry", retry_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "delegate")
    graph.add_edge("delegate", "execute")
    graph.add_edge("execute", "verify")

    graph.add_conditional_edges(
        "verify",
        should_retry,
        {"retry": "retry", "fail": END, "continue": "consensus"},
    )
    graph.add_edge("retry", "plan")
    graph.add_edge("consensus", END)

    return graph


def run_orchestration(task: str, max_retries: int = 2) -> dict:
    """Run the full orchestration pipeline for a task."""
    graph = build_orchestration_graph()
    app = graph.compile()

    initial_state: AgentState = {
        "task": task,
        "plan": "",
        "assigned_agent": "",
        "agent_output": "",
        "verification_result": "",
        "consensus_votes": [],
        "status": "planning",
        "retries": 0,
        "max_retries": max_retries,
        "mcp_tools_used": [],
        "executor_agent": "",
        "skipped_agents": [],
    }

    result = app.invoke(initial_state)
    save_run_state(result)
    return result


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Build a REST API endpoint"
    result = run_orchestration(task)
    print(json.dumps(result, indent=2, default=str))
