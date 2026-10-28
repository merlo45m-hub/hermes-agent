"""Browser Provider ABC: pluggable cloud browser backends (Browserbase, Browser Use, Firecrawl, …).

Providers register via :meth:`PluginContext.register_browser_provider`; the active one (selected by
``browser.cloud_provider``) services every cloud-mode ``browser_*`` tool call. They live in
``<repo>/plugins/browser/<name>/`` (built-in) or ``~/.hermes/plugins/browser/<name>/`` (user).

Session metadata contract (legacy ``CloudBrowserProvider`` shape; ``tools.browser_tool`` needs no
translation). ``bb_session_id`` is a legacy key name kept verbatim — it holds the provider's session ID
regardless of provider::

    {
        "session_name": str,        # unique name for agent-browser --session
        "bb_session_id": str,       # provider session ID (for close/cleanup)
        "cdp_url": str,             # CDP websocket URL
        "expires_at": str,          # optional provider-authoritative ISO timestamp
        "features": dict,           # feature flags that were enabled
        "external_call_id": str,    # optional, managed-gateway billing key
    }
"""

from __future__ import annotations

import abc
from typing import Dict

from agent.provider_base import ProviderBase


class BrowserProvider(ProviderBase):
    """Abstract base class for a cloud browser backend.

    Subclasses implement :attr:`name` (the ``browser.cloud_provider`` value), :meth:`is_available`, and
    the lifecycle trio :meth:`create_session` / :meth:`close_session` / :meth:`emergency_cleanup`.
    ``get_setup_schema`` may add ``"post_setup"`` (e.g. ``"agent_browser"``) to trigger the install hook.
    """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when this provider can service calls. Cheap check only (env var, token readable, dep
        importable) — must NOT make network calls; runs at tool-registration time and on every
        ``hermes tools`` paint."""

    @abc.abstractmethod
    def create_session(self, task_id: str) -> dict[str, object]:
        """Create a cloud browser session and return the metadata dict from the module docstring.
        May raise ``ValueError`` (missing credentials) or ``RuntimeError`` (network / API failure);
        the dispatcher surfaces these to the user."""

    @abc.abstractmethod
    def close_session(self, session_id: str) -> bool:
        """Release a cloud session by provider session ID. Returns True on success, False on failure;
        should not raise (log and return False so the dispatcher's cleanup loop keeps moving)."""

    @abc.abstractmethod
    def emergency_cleanup(self, session_id: str) -> None:
        """Best-effort teardown from atexit / signal handlers. Must tolerate missing credentials and
        network errors; must not raise."""


# ---------------------------------------------------------------------------
# LangChain orchestration support
# ---------------------------------------------------------------------------
from typing import Any, Dict, Optional

import logging

from agent.langchain_browser import (
    BrowserAgent,
    ConversationBufferMemory,
    LLMCall,
    REQUIRED_TOOLS,
)


class LangChromeBrowserAgent(BrowserProvider):
    """BrowserProvider wrapper driven by a LangChain agent.

    Implements the full :class:`BrowserProvider` interface by delegating to the
    wrapped (real) backend, so registration, availability checks, session
    lifecycle and the session-metadata contract are *unchanged*.  The agent
    additionally exposes the LangChain surface — managed prompts, conversation
    memory and an LLM call — for dynamic site navigation + element interaction.
    """

    def __init__(self, provider: BrowserProvider, *, llm: LLMCall) -> None:
        self._provider = provider
        self.llm = llm
        self.memory = ConversationBufferMemory(
            session_id=getattr(self, "_langchain_session_id", "langchain-agent")
        )
        self.session_id: Optional[str] = None
        self._session: Optional[Dict[str, object]] = None
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

    @property
    def name(self) -> str:
        return f"langchain-{self._provider.name}"

    @property
    def display_name(self) -> str:
        return f"LangChain browser agent ({self._provider.name})"

    def is_available(self) -> bool:
        return self._provider.is_available()

    def create_session(self, task_id: str) -> Dict[str, object]:
        if self._session is None:
            self._session = self._provider.create_session(task_id)
        return self._session

    def close_session(self, session_id: str) -> bool:
        return self._provider.close_session(session_id)

    def emergency_cleanup(self, session_id: str) -> None:
        self._provider.emergency_cleanup(session_id)

    def get_setup_schema(self) -> Optional[Dict[str, Any]]:
        return self._provider.get_setup_schema()

    @property
    def _browser_provider(self) -> BrowserProvider:
        return self._provider

    # -- helper the registry / vendor plugins use --------------------------
    def langchrome(self, llm: LLMCall) -> "LangChromeBrowserAgent":
        return LangChromeBrowserAgent(self._provider, llm=llm)


def create_langchrome_browser_agent(
    provider: BrowserProvider, llm: LLMCall
) -> LangChromeBrowserAgent:
    """Factory: turn an existing :class:`BrowserProvider` into a LangChain agent.

    The returned wrapper keeps the full BrowserProvider interface, so it is
    drop-in for the raw provider everywhere the cloud browser tools read it.
    """
    return LangChromeBrowserAgent(provider, llm=llm)
