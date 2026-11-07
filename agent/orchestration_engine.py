"""
Layer 5: Orchestration — Multi-agent state machine with LangGraph.

Coordinates Hermes, agy, codex, and claude agents through stateful
workflows with handoffs and consensus loops.
"""

import os
import json
import logging
import subprocess
import shutil
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

log = logging.getLogger(__name__)


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


# Agent routing table: task type -> agent
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
    "security": ["security", "audit", "vulnerability", "rls", "auth", "hardening"],
    "research": ["research", "search", "investigate", "analyze", "explore"],
    "godot": ["godot", "game", "scene", "gdscript"],
    "build": ["build", "compile", "deploy", "docker", "install"],
    "review": ["review", "verify", "test", "audit", "check"],
}

# Agent binary paths.
# agy runs on the VPS: the phone's agy port (1.3.5) cannot resolve DNS because
# the proot has no /etc/resolv.conf and /etc is a read-only symlink to
# /system/etc, so glibc's resolver falls back to [::1]:53. agy-vps forwards the
# prompt over the vps-mcp bridge with shell quoting preserved.
AGENT_BINARIES = {
    "agy": "/data/data/com.termux/files/home/bin/agy-vps",
    "codex": "/data/data/com.termux/files/usr/bin/codex",
    "claude": "/data/data/com.termux/files/usr/bin/claude",
}

# Preferred order for the consensus reviewer and for dispatch fallbacks.
# claude is the only backend that currently works reliably on the phone: agy
# cannot resolve DNS (the proot has no /etc/resolv.conf, and /etc is read-only)
# and codex waits on stdin in this invocation form.
_FALLBACK_ORDER = ("claude", "codex", "agy")

# Per-backend attempt budget. agy runs on the VPS over ssh, so it needs a
# materially larger window than the local backends. The worst-case sum must stay
# under the orchestrate tool's 420s cap, or a hung agy takes the whole call down
# instead of failing cleanly to the fallbacks: 200 + 80 + 90 = 370s.
_AGENT_TIMEOUTS = {"agy": 200, "codex": 80, "claude": 90}

# Records which backend actually answered the most recent dispatch (the assigned
# agent is not always the one that ran). Read by the L5 evidence marker writer.
LAST_EFFECTIVE_AGENT = None

# MCP tool routing: which tools each agent can use
AGENT_MCP_TOOLS = {
    "agy": ["godot", "blender", "asset_grabber", "filesystem", "github"],
    "codex": ["postgres", "supabase", "github", "filesystem", "searxng"],
    "claude": ["searxng", "filesystem", "github"],
}


def classify_task(task: str) -> str:
    """Classify a task into a type based on keywords."""
    task_lower = task.lower()
    scores = {}
    for task_type, keywords in TASK_KEYWORDS.items():
        scores[task_type] = sum(1 for kw in keywords if kw in task_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "default"


def route_to_agent(task_type: str) -> str:
    """Route a task type to the appropriate agent."""
    return AGENT_ROUTING.get(task_type, AGENT_ROUTING["default"])


def _agent_cmd(agent: str, task: str):
    """Build the command line for one agent, or None if it is unavailable."""
    binary = AGENT_BINARIES.get(agent)
    if not binary or not os.path.exists(binary):
        return None
    if agent == "agy":
        # agy-vps takes the prompt as its single argument and forwards it to the
        # VPS build over ssh with shell quoting preserved.
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


def dispatch_agent(agent: str, task: str, timeout: int = 120) -> str:
    """Dispatch a task to the assigned agent, falling back to the others.

    No single backend is dependable on the phone: agy cannot resolve DNS (the
    proot has no /etc/resolv.conf and /etc is read-only), codex waits on stdin in
    this invocation form, and claude is the only one that answers reliably. So a
    failed or empty attempt falls through to the next agent instead of failing
    the whole orchestration.
    """
    global LAST_EFFECTIVE_AGENT
    order = [agent] + [a for a in _FALLBACK_ORDER if a != agent]
    last = ""
    for attempt in order:
        ok, output = _dispatch_once(attempt, task, _AGENT_TIMEOUTS.get(attempt, timeout))
        if ok:
            LAST_EFFECTIVE_AGENT = attempt
            if attempt != agent:
                log.warning("dispatch_agent: %s produced nothing, fell back to %s", agent, attempt)
            return output
        last = output
    return last or f"[ERROR] no agent available for {agent}"


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

    output = dispatch_agent(agent, prompt, timeout=100)
    return {
        **state,
        "agent_output": output,
        "status": "verifying",
        "executor_agent": LAST_EFFECTIVE_AGENT or agent,
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

    # Pick a different agent for review, preferring responsive verification agents
    reviewers = [a for a in _FALLBACK_ORDER if a != agent and os.path.exists(AGENT_BINARIES.get(a, ""))]
    reviewer = reviewers[0] if reviewers else "claude"

    review_prompt = (
        f"Task: {state['task']}\n"
        f"Review this work output and say APPROVE or REJECT with reason.\n"
        f"Agent: {agent}\nOutput:\n{output[:2000]}\n\n"
        f"Respond with exactly: APPROVE: <reason> or REJECT: <reason>"
    )

    review_output = dispatch_agent(reviewer, review_prompt, timeout=90)
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
    }

    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Build a REST API endpoint"
    result = run_orchestration(task)
    print(json.dumps(result, indent=2, default=str))
