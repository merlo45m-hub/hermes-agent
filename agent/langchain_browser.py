"""LangChain orchestration layer for the MCP browser tool surface.

This module turns the existing browser providers into LangChain-based browser
agents. The agent:

* keeps a system prompt + conversation memory managed by LangChain, so that
  navigation and element-interaction decisions are driven by an LLM rather than
  a fixed script;
* reuses the existing browser provider session (the MCP browser tools) for all
  DOM manipulation - nothing is removed, the providers' sessions are untouched;
* persists memory across sessions via a LangChain style chat-history buffer.

The agent is intentionally LLM-agnostic: an :class:`LLMCall` carries the model +
prompt, and a :class:`BaseChatModel` is constructed by a caller-supplied factory
so no secrets are hard-coded and nothing needs the network at import time.

Public API
----------
.. autostruct::
   :noindex:

"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import Tool


logger = logging.getLogger(__name__)

#: Default maximum turn-steps for a single ``run()`` so a misrouted agent always
#: gives up instead of looping forever.
_DEFAULT_MAX_STEPS = 12

#: Accessibility/evalology vocabulary that decides whether an LLM may keep
#: driving the browser (ensure existing controls keep working).
_AGENT_TOOL_FAMILY = frozenset(
    {
        "navigate", "click", "type", "press", "scroll",
        "snapshot", "screenshot", "close", "interact",
    }
)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
class ConversationBufferMemory:
    """LangChain style conversation buffer for a browser session.

    Behaviour mirrors :class:`~langchain.memory.buffer.ConversationBufferMemory`.
    The buffer is a plain list of ``{"role", "content"}`` dicts so it is fully
    importable without network and has no dependency on the (scheduled for
    removal) LangChain memory classes.  It can be persisted across
    ``BrowserAgent`` instances by sharing a ``session_id``.
    """

    # Class-level store keyed by session_id so a buffer outlives individual
    # agent instances (memory persists across sessions).
    _STORES: Dict[str, List[Dict[str, str]]] = {}

    def __init__(self, *, session_id: str, max_tokens: int = 4096) -> None:
        self.session_id = session_id
        self.max_tokens = max_tokens
        self._buffer = self._STORES.setdefault(session_id, [])

    # -- LangChain memory API -------------------------------------------------
    def load_memory_variables(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        return {"history": self._buffer}

    def save_context(self, inputs: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
        user_content = str(inputs.get("task") or "")
        self._buffer.append({"role": "user", "content": user_content})
        if outputs:
            assistant_content = self._summarise(outputs)
            self._buffer.append({"role": "assistant", "content": assistant_content})

    def clear(self) -> None:
        self._buffer.clear()

    def token_count(self) -> int:
        return sum(len(str(c.get("content", ""))) for c in self._buffer)

    def get_messages(self) -> List[Dict[str, str]]:
        # Represent buffer as LangChain-style messages without pulling the
        # deprecated memory classes into the import graph here.
        from langchain_core.messages import HumanMessage

        return [HumanMessage(content=c.get("content", "")) for c in self._buffer]

    def _summarise(self, outputs: Mapping[str, Any]) -> str:
        chunks: List[str] = []
        for value in outputs.values():
            text = str(value)
            if len(text) > 300:
                text = text[:300] + "…"
            chunks.append(text)
        return " | ".join(chunks)
@dataclass(frozen=True)

class LLMCall:
    """A configured language-model call for the browser agent.

    Everything the agent needs (model, prompt, temperature, limits) lives in one
    data object. No secrets are stored here - the chat model is created by a
    caller-supplied factory so LLM credentials stay out of this module and out of
    any shipped config file.
    """

    model: str = "test-model"
    temperature: float = 0.0
    max_tokens: int = 512
    system_prompt: str = ""
    create_chat_model: Optional[Callable[[], BaseChatModel]] = None
    # When True the agent skips the LLM and hands control to a scripted "oracle"
    # (used by tests and record/replay).
    use_oracle: bool = False
    # Test-friendly: inject a deterministic plan instead of calling an LLM.
    oracle_plan: Optional[Callable[[str, List[Dict[str, Any]]], Dict[str, Any]]] = None

    def chat_model(self) -> Optional[BaseChatModel]:
        if self.use_oracle:
            return None
        if self.create_chat_model is None:
            return None
        return self.create_chat_model()

    def plan(self, prompt: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ask the LLM for the next action; returns a parsed JSON action dict."""
        if self.use_oracle:
            return self.oracle_plan(prompt, history) if self.oracle_plan else {"action": "finish", "answer": "oracle done"}
        if self.oracle_plan is not None:
            return self.oracle_plan(prompt, history)
        chat = self.chat_model()
        if chat is None:
            raise RuntimeError(
                "LLMCall is not configured with a chat model or oracle; "
                "set create_chat_model or use_oracle=True"
            )
        messages = self._build_messages(prompt, history)
        response = chat.invoke(messages)
        payload = self._extract_payload(response)
        return self._ensure_action(payload)

    def _build_messages(self, prompt: str, history: List[Dict[str, Any]]) -> List[BaseMessage]:
        msgs: List[BaseMessage] = []
        if self.system_prompt:
            msgs.append(SystemMessage(content=self.system_prompt))
        for turn in history:
            if turn.get("role") == "user":
                msgs.append(HumanMessage(content=turn["content"]))
        msgs.append(HumanMessage(content=prompt))
        return msgs

    @staticmethod
    def _extract_payload(response: BaseMessage) -> Dict[str, Any]:
        text = str(getattr(response, "content", ""))
        if not text:
            return {"action": "finish", "answer": ""}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            return {"action": "finish", "answer": text[:2000]}
        return parsed

    @staticmethod
    def _ensure_action(payload: Dict[str, Any]) -> Dict[str, Any]:
        action = payload.get("action", "finish")
        if not isinstance(action, str):
            action = "finish"
        return {"action": action, **payload}


def _session_name(task_id: str) -> str:
    """Unique session name (used by the agent when it must create a provider session)."""
    return f"hermes_agent_{task_id}_{uuid.uuid4().hex[:8]}"



# ---------------------------------------------------------------------------
# Agent executor
# ---------------------------------------------------------------------------
class AgentExecutor:
    """Small, dependency-free agent executor that drives a :class:`BrowserAgent`.

    It follows the classic think-act-observe loop and is the "agent executor"
    the LangChain architecture requests.
    """

    def __init__(self, agent: "BrowserAgent") -> None:
        self.agent = agent

    def run(self, task: str, *, config: Optional[RunnableConfig] = None, max_steps: int = _DEFAULT_MAX_STEPS) -> Dict[str, Any]:
        memory = self.agent.memory or ConversationBufferMemory(session_id=str(id(task)))
        history: List[Dict[str, Any]] = []
        observation = ""
        if self.agent._session is None:
            self.agent._ensure_session(config)
        try:
            for _ in range(max_steps):
                prompt = self.agent._build_prompt(task, memory, history, observation)
                action = self.agent._plan(prompt, history)
                if action.get("action") == "finish":
                    answer = action.get("answer", "")
                    memory.save_context({"task": task, "action": action}, {"answer": answer})
                    return {"status": "success", "answer": answer}
                result = self.agent.call_tool(action.get("action"), action, memory, config)
                observation = self.agent._render(result)
                history.append({"role": "user", "content": prompt})
                memory.save_context({"task": task, "action": action}, {"output": result})
                if self.agent._stopped:
                    return {"status": "success", "answer": observation}
            return {"status": "error", "message": "max steps exceeded", "observation": observation}
        finally:
            pass

    async def arun(self, task: str, *, config: Optional[RunnableConfig] = None, max_steps: int = _DEFAULT_MAX_STEPS) -> Dict[str, Any]:
        # The synchronous path is sufficient for the browser agent; async flow is
        # an opt-in when the caller's chat model supports ``ainvoke``.
        result = self.run(task, config=config, max_steps=max_steps)
        return result


# ---------------------------------------------------------------------------
# Agent tools - the MCP browser tool surface re-exported for the agent
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentTool:
    """A single tool exposed to the LangChain browser agent."""

    name: str
    description: str
    kwargs_schema: str
    call: Callable[..., Dict[str, Any]]


def _browser_task_id(session: Dict[str, object]) -> str:
    """Return the task key shared by the browser tools and their provider session."""
    task_id = session.get("session_name") or session.get("task_id") or session.get("bb_session_id")
    if not task_id:
        raise ValueError("Browser session metadata does not contain a task identifier")
    return str(task_id)


def _browser_result(response: Any, session: Dict[str, object]) -> Dict[str, Any]:
    """Normalize browser tool responses to the LangChain agent result contract."""
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            response = {"message": response}
    if not isinstance(response, dict):
        response = {"result": response}
    success = response.get("success") is not False and "error" not in response
    return {
        "status": "success" if success else "error",
        "result": response,
        **({"message": str(response["error"])} if not success and response.get("error") else {}),
        "session": session,
    }


def _execute_navigate(session: Dict[str, object], url: str) -> Dict[str, Any]:
    from tools.browser_camofox import camofox_navigate

    return _browser_result(camofox_navigate(url, task_id=_browser_task_id(session)), session)


def _execute_click(session: Dict[str, object], selector: str) -> Dict[str, Any]:
    from tools.browser_camofox import camofox_click

    return _browser_result(camofox_click(selector, task_id=_browser_task_id(session)), session)


def _execute_type(session: Dict[str, object], selector: str, text: str) -> Dict[str, Any]:
    from tools.browser_camofox import camofox_type

    return _browser_result(camofox_type(selector, text, task_id=_browser_task_id(session)), session)


def _execute_press(session: Dict[str, object], key: str) -> Dict[str, Any]:
    from tools.browser_camofox import camofox_press

    return _browser_result(camofox_press(key, task_id=_browser_task_id(session)), session)


def _execute_scroll(session: Dict[str, object], direction: str) -> Dict[str, Any]:
    from tools.browser_camofox import camofox_scroll

    return _browser_result(camofox_scroll(direction, task_id=_browser_task_id(session)), session)


def _execute_snapshot(session: Dict[str, object]) -> Dict[str, Any]:
    from tools.browser_camofox import camofox_snapshot

    return _browser_result(camofox_snapshot(task_id=_browser_task_id(session)), session)


def _execute_screenshot(session: Dict[str, object]) -> Dict[str, Any]:
    from tools.browser_camofox import camofox_vision

    return _browser_result(
        camofox_vision("Capture a screenshot of the current page", task_id=_browser_task_id(session)),
        session,
    )


def _execute_close(session: Dict[str, object]) -> Dict[str, Any]:
    from tools.browser_camofox import camofox_close

    return _browser_result(camofox_close(task_id=_browser_task_id(session)), session)


REQUIRED_TOOLS: Tuple[AgentTool, ...] = (
    AgentTool("navigate", "Navigate to a URL", "url: str", _execute_navigate),
    AgentTool("click", "Click an element on the page", "selector: str", _execute_click),
    AgentTool("type", "Type text into an element", "selector: str, text: str", _execute_type),
    AgentTool("press", "Press a keyboard key", "key: str", _execute_press),
    AgentTool("scroll", "Scroll the page", "direction: str", _execute_scroll),
    AgentTool("snapshot", "Get the accessibility tree snapshot of the page", "", _execute_snapshot),
    AgentTool("screenshot", "Capture a screenshot of the page", "", _execute_screenshot),
    AgentTool("close", "Close the current page", "", _execute_close),
)


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
class BrowserAgent:
    """A LangChain-based browser agent that drives the existing MCP browser
    provider surface.

    The agent keeps a system prompt, conversation memory and an LLM call.  Its
    ``run`` method performs site navigation and element interaction dynamically,
    while the actual DOM manipulation is delegated to the wrapped provider's
    session (the same surface the MCP ``browser_tool_*`` tools use).

    Because ``BrowserProvider`` instances are registered singletons, wrapping
    them in a :class:`BrowserAgent` does not remove the underlying tools: the
    provider's registration, availability, session lifecycle and metadata
    contract are all preserved.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You control a cloud browser through deterministic tools. You are given a "
        "goal as the LAST assistant user turn of this message. Each turn you must "
        "emit exactly ONE JSON object with an 'action' field naming one of the "
        "available tools and any of its parameters.\n\n"
        "Available tools and their parameters:\n"
        + "\n".join(
            f"- `{t.name}` ({t.kwargs_schema}) -> {t.description}" for t in REQUIRED_TOOLS
        )
        + "\n"
        + "\n- `finish` - end the session and return the final answer (no other "
        + "fields).\n"
        + "\n"
        + "Never emit markdown fences or text outside the JSON object. Always "
        + "include 'action' and any required parameters. Only call tools that are "
        + "listed above."
    )

    def __init__(
        self,
        provider: BrowserProvider,
        *,
        llm: LLMCall,
        tools: Sequence[AgentTool] = REQUIRED_TOOLS,
        memory: Optional[ConversationBufferMemory] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._provider = provider
        self.llm = llm
        self.tools = list(tools)
        self.memory = memory or ConversationBufferMemory(session_id=session_id or "browser-agent")
        self.session_id = session_id
        self._session: Optional[Dict[str, object]] = None
        self._stopped = False
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

    # -- identity -----------------------------------------------------------
    @property
    def name(self) -> str:
        return f"langchain-{self._provider.name}"

    @property
    def display_name(self) -> str:
        return f"LangChain browser agent ({self._provider.name})"

    # -- backward-compatible BrowserProvider interface ------------------------
    def is_available(self) -> bool:
        return self._provider.is_available()

    def create_session(self, task_id: str) -> Dict[str, object]:
        return self._provider.create_session(task_id)

    def close_session(self, session_id: str) -> bool:
        return self._provider.close_session(session_id)

    def emergency_cleanup(self, session_id: str) -> None:
        self._provider.emergency_cleanup(session_id)

    def get_setup_schema(self) -> Optional[Dict[str, Any]]:
        return self._provider.get_setup_schema()

    # -- agent internals (called by AgentExecutor) --------------------------
    def _ensure_session(self, config: Optional[RunnableConfig]) -> None:
        if self._session is not None:
            return
        if self.session_id is None:
            self.session_id = _session_name("agent")
        self._session = self._provider.create_session(self.session_id)
        self.logger.info("Agent session: %s", self.session_id)

    def _build_prompt(self, task: str, memory: ConversationBufferMemory, history, observation: str) -> str:
        buffer = memory.load_memory_variables({})["history"]
        parts = [self.DEFAULT_SYSTEM_PROMPT, f"Goal: {task}"]
        if buffer:
            parts.append("Conversation so far:")
            for msg in buffer[-12:]:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                text = msg.content if hasattr(msg, "content") else str(msg)
                parts.append(f"[{role}]: {text}")
        if observation:
            parts.append(f"Last observation: {observation}")
        return "\n".join(parts)

    def _plan(self, prompt: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.llm.plan(prompt, history)

    def call_tool(self, name: str, parsed: Dict[str, Any], memory: ConversationBufferMemory, config: Optional[RunnableConfig]) -> Dict[str, Any]:
        tool = self._tool_by_name(name)
        if tool is None:
            return {"status": "error", "message": f"unknown tool '{name}'"}
        self._ensure_session(config)
        try:
            # The action name itself is not a tool argument; pass only the tool's
            # named parameters to the underlying browser tool.
            kwargs = {k: v for k, v in parsed.items() if k != "action"}
            result = tool.call(self._session, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.exception("tool %r failed", name)
            return {"status": "error", "message": f"{name} failed: {exc}"}
        if result.get("status") == "success":
            self._stopped = False
        return result

    def _tool_by_name(self, name: str) -> Optional[AgentTool]:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def _render(self, result: Dict[str, Any]) -> str:
        status = result.get("status", "error")
        text = str(result.get("message") or result)
        return f"[{status}] {text[:1200]}"

    def to_langchain_agent(self, llm: LLMCall) -> "BrowserAgent":
        return BrowserAgent(self._provider, llm=llm, memory=self.memory, session_id=self.session_id)
