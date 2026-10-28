"""Tests for the LangChain orchestration layer (agent.langchain_browser)."""

import sys
from typing import Dict

sys.path.insert(0, ".")

from agent.browser_provider import BrowserProvider
from agent.langchain_browser import (
    AgentExecutor,
    BrowserAgent,
    ConversationBufferMemory,
    LLMCall,
    REQUIRED_TOOLS,
)


class _FakeBrowser(BrowserProvider):
    """Minimal in-memory browser provider (no network)."""

    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, str]] = {}
        self.created: list[str] = []
        self.closed: list[str] = []

    @property
    def name(self) -> str:
        return "fake-browser"

    def is_available(self) -> bool:
        return True

    def create_session(self, task_id: str) -> Dict[str, object]:
        meta = {"session_name": f"h_{task_id}", "bb_session_id": f"s-{task_id}", "cdp_url": "ws://localhost"}
        self.sessions[task_id] = meta
        self.created.append(task_id)
        return meta

    def close_session(self, session_id: str) -> bool:
        self.closed.append(session_id)
        return True

    def emergency_cleanup(self, session_id: str) -> None:
        pass


def _make_agent(plan):
    return BrowserAgent(
        _FakeBrowser(),
        llm=LLMCall(use_oracle=True, oracle_plan=lambda p, h: plan),
    )


# ---------------------------------------------------------------------------
# ConversationBufferMemory (session-persistent)
# ---------------------------------------------------------------------------
def test_conversation_buffer_returns_history():
    m = ConversationBufferMemory(session_id="s1")
    # save_context appends a user turn AND an assistant-turn summarising the output
    m.save_context({"task": "go to x"}, {"action": {"action": "navigate", "url": "https://x.com"}})
    hist = m.load_memory_variables({})["history"]
    assert len(hist) == 2
    assert hist[0] == {"role": "user", "content": "go to x"}
    assert hist[1]["role"] == "assistant"
    assert "x.com" in hist[1]["content"]


def test_conversation_buffer_isolated_per_session():
    a = ConversationBufferMemory(session_id="A")
    a.save_context({"task": "A"}, {"action": "done"})
    b = ConversationBufferMemory(session_id="B")
    assert b.load_memory_variables({})["history"] == []


def test_conversation_buffer_clear():
    m = ConversationBufferMemory(session_id="c")
    m.save_context({"task": "x"}, {"action": "done"})
    m.clear()
    assert m.load_memory_variables({})["history"] == []


# ---------------------------------------------------------------------------
# LLMCall
# ---------------------------------------------------------------------------
def test_llc_call_oracle_plan():
    plan = {"action": "navigate", "url": "https://example.com"}
    llm = LLMCall(use_oracle=True, oracle_plan=lambda p, h: plan)
    assert llm.plan("prompt", []) == plan


# ---------------------------------------------------------------------------
# BrowserAgent run loop
# ---------------------------------------------------------------------------
def test_agent_navigates_then_finishes(monkeypatch):
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(
        browser_tool,
        "browser_navigate",
        lambda url, task_id=None: '{"success": true, "url": "' + url + '"}',
    )
    steps = [{"action": "navigate", "url": "https://example.com"}, {"action": "finish", "answer": "done"}]

    class _TrackingBrowser(BrowserProvider):
        @property
        def name(self):
            return "tracking"

        def is_available(self):
            return True

        def create_session(self, task_id):
            return {"bb_session_id": "s1", "cdp_url": "ws://x"}

        def close_session(self, session_id):
            return True

        def emergency_cleanup(self, session_id):
            pass

    agent = BrowserAgent(
        _TrackingBrowser(),
        llm=LLMCall(use_oracle=True, oracle_plan=lambda p, h: steps.pop(0)),
    )
    result = AgentExecutor(agent).run("go to example.com")
    assert result["status"] == "success"
    assert result["answer"] == "done"
    assert agent._session is not None
    assert agent._session["bb_session_id"] == "s1"


def test_agent_tool_surface_used(monkeypatch):
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(
        browser_tool,
        "browser_navigate",
        lambda url, task_id=None: '{"success": true, "url": "' + url + '"}',
    )
    # The agent must route a user task onto the wrapped provider and exercise the
    # managed tool surface (here navigation via REQUIRED_TOOLS).
    called = {}

    class _RecordingBrowser(BrowserProvider):
        @property
        def name(self):
            return "rec"

        def is_available(self):
            return True

        def create_session(self, task_id):
            self.session = {"bb_session_id": "s1", "cdp_url": "ws://x"}
            return self.session

        def close_session(self, session_id):
            return True

        def emergency_cleanup(self, session_id):
            pass

    agent = BrowserAgent(
        _RecordingBrowser(),
        llm=LLMCall(
            use_oracle=True,
            oracle_plan=lambda p, h: steps.pop(0),
        ),
    )
    steps = [
        {"action": "navigate", "url": "https://example.com"},
        {"action": "finish", "answer": "done"},
    ]
    result = AgentExecutor(agent).run("go to example.com")
    assert result["status"] == "success"
    assert agent._session is not None
    assert agent._session["bb_session_id"] == "s1"
    # Agent keeps a managed prompt + conversation memory across session.
    assert agent.memory.session_id == "browser-agent"
    hist = agent.memory.load_memory_variables({})["history"]
    assert len(hist) >= 1
    # The agent's tools are the managed MCP browser tool surface.
    assert [t.name for t in agent.tools] == [t.name for t in REQUIRED_TOOLS]


def test_browser_actions_call_real_tool_surface(monkeypatch):
    import tools.browser_camofox as browser_camofox
    from agent.langchain_browser import (
        _execute_click,
        _execute_close,
        _execute_navigate,
        _execute_press,
        _execute_screenshot,
        _execute_scroll,
        _execute_snapshot,
        _execute_type,
    )

    session = {"session_name": "agent-task"}
    calls = []

    def response(name):
        def call(*args, task_id=None):
            calls.append((name, args, task_id))
            return '{"success": true, "action": "' + name + '"}'

        return call

    tool_names = (
        "camofox_navigate", "camofox_click", "camofox_type", "camofox_press",
        "camofox_scroll", "camofox_snapshot", "camofox_vision", "camofox_close",
    )
    for name in tool_names:
        monkeypatch.setattr(browser_camofox, name, response(name))

    actions = [
        _execute_navigate(session, "https://example.com"),
        _execute_click(session, "@e1"),
        _execute_type(session, "@e2", "hello"),
        _execute_press(session, "Enter"),
        _execute_scroll(session, "down"),
        _execute_snapshot(session),
        _execute_screenshot(session),
        _execute_close(session),
    ]

    assert all(result["status"] == "success" for result in actions)
    assert [call[0] for call in calls] == [
        "camofox_navigate", "camofox_click", "camofox_type", "camofox_press",
        "camofox_scroll", "camofox_snapshot", "camofox_vision", "camofox_close",
    ]
    assert all(call[2] == "agent-task" for call in calls)
