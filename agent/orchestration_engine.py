"""
Layer 5: Orchestration — Multi-agent state machine with LangGraph.

Coordinates Hermes, agy, codex, and claude agents through stateful
workflows with handoffs and consensus loops.
"""

import os
import json
import subprocess
import shutil
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END


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

# Agent binary paths
AGENT_BINARIES = {
    "agy": "/data/data/com.termux/files/usr/bin/agy",
    "codex": "/data/data/com.termux/files/usr/bin/codex",
    "claude": "/data/data/com.termux/files/usr/bin/claude",
}

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


def dispatch_agent(agent: str, task: str, timeout: int = 300) -> str:
    """Dispatch a task to an agent via subprocess."""
    binary = AGENT_BINARIES.get(agent)
    if not binary or not os.path.exists(binary):
        return f"[ERROR] Agent binary not found: {agent} ({binary})"

    if agent == "agy":
        cmd = [binary, "--print", task, "--dangerously-skip-permissions"]
    elif agent == "codex":
        cmd = [binary, "exec", "--skip-git-repo-check", task]
    elif agent == "claude":
        cmd = [binary, "-p", task, "--allowedTools", "WebSearch WebFetch Read"]
    else:
        return f"[ERROR] Unknown agent: {agent}"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/data/data/com.termux/files/home",
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            output += f"\n[STDERR] {result.stderr.strip()}"
        return output if output else "[No output]"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] Agent {agent} exceeded {timeout}s"
    except Exception as e:
        return f"[ERROR] {e}"


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

    prompt = f"Task: {task}\nPlan: {plan}{tool_context}\n\nExecute this task and report what you did."

    output = dispatch_agent(agent, prompt, timeout=120)
    return {
        **state,
        "agent_output": output,
        "status": "verifying",
    }


def verify_node(state: AgentState) -> AgentState:
    """Verify the agent's output."""
    output = state["agent_output"]
    # Check for errors in output
    has_error = "[ERROR]" in output or "[TIMEOUT]" in output
    # Check for meaningful content
    has_content = len(output) > 50 and "[No output]" not in output

    passed = not has_error and has_content
    return {
        **state,
        "verification_result": "PASS" if passed else "FAIL",
        "status": "consensus" if passed else "failed",
    }


def consensus_node(state: AgentState) -> AgentState:
    """Multi-agent consensus: review by a different agent."""
    agent = state["assigned_agent"]
    output = state["agent_output"]

    # Pick a different agent for review
    reviewers = [a for a in AGENT_BINARIES if a != agent]
    reviewer = reviewers[0] if reviewers else "claude"

    review_prompt = (
        f"Review this work output and say APPROVE or REJECT with reason.\n"
        f"Agent: {agent}\nOutput:\n{output[:2000]}\n\n"
        f"Respond with exactly: APPROVE: <reason> or REJECT: <reason>"
    )

    review_output = dispatch_agent(reviewer, review_prompt, timeout=60)
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
    }

    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Build a REST API endpoint"
    result = run_orchestration(task)
    print(json.dumps(result, indent=2, default=str))
