"""Tests for the BrowserProvider LangChain agent wrapper.

Covers the drop-in BrowserProvider wrapper and the router used by the
LangChain integration.
"""

import sys
from typing import Dict

sys.path.insert(0, ".")

from agent.browser_provider import (
    BrowserProvider,
    LangChromeBrowserAgent,
    create_langchrome_browser_agent,
)
from agent.browser_registry import (
    _registry,
    get_langchrome_browser_agent,
    list_langchrome_browser_agents,
)
from agent.langchain_browser import LLMCall


class _FakeBrowser(BrowserProvider):
    """Minimal in-memory provider used for the wrapper + registry tests."""

    def __init__(self, name: str = "fake-provider") -> None:
        self._name = name
        self.sessions: Dict[str, Dict[str, object]] = {}
        self.created: list[str] = []
        self.closed: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def create_session(self, task_id: str) -> Dict[str, object]:
        meta = {"session_name": f"h_{task_id}", "bb_session_id": f"bb-{task_id}"}
        self.sessions[task_id] = meta
        self.created.append(task_id)
        return meta

    def close_session(self, session_id: str) -> bool:
        self.closed.append(session_id)
        return True

    def emergency_cleanup(self, session_id: str) -> None:
        pass


def test_langchrome_browseragent_preserves_provider_interface():
    prov = _FakeBrowser()
    agent = LangChromeBrowserAgent(prov, llm=LLMCall(model="t"))
    assert agent.name == "langchain-fake-provider"
    assert agent.display_name.startswith("LangChain browser agent")
    assert agent.is_available() is True
    assert agent.create_session("t1")["bb_session_id"] == "bb-t1"
    assert agent.close_session("bb-t1") is True
    agent.emergency_cleanup("bb-t1")
    assert agent._provider is prov


def test_create_langchrome_browser_agent_factory():
    prov = _FakeBrowser()
    agent = create_langchrome_browser_agent(prov, LLMCall(model="t"))
    assert isinstance(agent, LangChromeBrowserAgent)
    assert agent.name == "langchain-fake-provider"


def test_registry_get_and_list_langchrome():
    _registry.register(_FakeBrowser("one"), scope=None)
    _registry.register(_FakeBrowser("two"), scope=None)

    wrapped = list_langchrome_browser_agents(LLMCall(model="t"))
    names = {a.name for a in wrapped}
    assert names == {"langchain-one", "langchain-two"}


def test_registry_forwarding_is_original_provider_not_value():
    prov = _FakeBrowser("one")
    agent = create_langchrome_browser_agent(prov, LLMCall(model="t"))
    # Wrapping must not shadow the real provider's session lifecycle.
    s = agent.create_session("t1")
    assert s["bb_session_id"] == "bb-t1"
    assert agent.close_session("bb-t1") is True
