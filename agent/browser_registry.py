"""Browser provider registry: cloud browser backends registered by plugins via
:meth:`PluginContext.register_browser_provider`, consumed by ``tools.browser_tool_cloud._get_cloud_provider``.

Active-provider precedence (see :func:`_resolve`): ``browser.cloud_provider`` in config.yaml wins
regardless of ``is_available()`` (so the dispatcher surfaces a typed "X_API_KEY is not set" error
instead of silently switching); else the legacy auto-detect walk ``browser-use`` → ``browserbase``
filtered by availability; else ``None`` (local browser mode). There is no capability split here —
every provider implements the full :class:`agent.browser_provider.BrowserProvider` lifecycle.
"""


import logging
from typing import Optional

from agent.browser_provider import BrowserProvider
from agent.provider_registry import ProviderRegistry, is_available_safe

logger = logging.getLogger(__name__)


_registry: ProviderRegistry[BrowserProvider] = ProviderRegistry(
    label="Browser", provider_cls=BrowserProvider, logger=logger,
)
_registry.export(globals())


# Auto-detect order when ``browser.cloud_provider`` is unset (historic order: Browser Use first because
# it covers both the managed Nous gateway and the direct API key path; Browserbase as the older
# direct-credentials fallback). Firecrawl is deliberately absent — see :func:`_resolve`.
_LEGACY_PREFERENCE = ("browser-use", "browserbase")


def _resolve(configured: Optional[str]) -> Optional[BrowserProvider]:
    """Resolve the active browser provider (rules in the module docstring).

    Intentionally NO "single-eligible shortcut" (unlike ``agent.web_search_registry._resolve``): only
    ``_LEGACY_PREFERENCE`` names are auto-eligible. Firecrawl shares its API key with the *web* extract
    plugin, so a user with ``FIRECRAWL_API_KEY`` must never be routed to a paid cloud browser without
    setting ``browser.cloud_provider``; the same gate applies to third-party browser-provider plugins.
    """
    snapshot = _registry.merged()
    if configured == "local":
        return None
    if configured:
        provider = snapshot.get(configured)
        if provider is not None:
            return provider
        logger.debug(
            "browser cloud_provider '%s' configured but not registered; falling back to auto-detect",
            configured,
        )
    for legacy in _LEGACY_PREFERENCE:
        provider = snapshot.get(legacy)
        if provider is not None and is_available_safe(
            provider, logger,
            "Browser provider %s.is_available() raised %s — treating as unavailable",
            level=logging.WARNING, exc_info=True,
        ):
            return provider
    return None


# LangChain agent resolution / routing (additive; the legacy ``_resolve`` keeps
# returning raw providers for backward compatibility).


def _browser_config_for_langchain(config: Optional[dict]) -> bool:
    """Return True when the active config opts the browser surface into LangChain.

    Reads ``browser.use_langchain_agent`` (default ``false``) so existing
    deployments keep serving raw MCP browser tools until they opt in.
    """
    if not isinstance(config, dict):
        return False
    browser = config.get("browser") if isinstance(config, dict) else None
    if not isinstance(browser, dict):
        return False
    return bool(browser.get("use_langchain_agent") or browser.get("langchain_agent"))


def list_langchrome_browser_agents(
    llm: "LLMCall", *, scope: Optional[str] = None
) -> list["LangChromeBrowserAgent"]:
    """Return every bundled browser provider wrapped as a LangChain agent."""
    from agent.browser_provider import LangChromeBrowserAgent

    return [LangChromeBrowserAgent(p, llm=llm) for p in _registry.list_providers(scope=scope)]


def get_langchrome_browser_agent(
    provider_name: str, llm: "LLMCall", *, scope: Optional[str] = None
) -> Optional["LangChromeBrowserAgent"]:
    """Return one bundled provider wrapped as a LangChain agent, or None."""
    from agent.browser_provider import LangChromeBrowserAgent

    raw = _registry.get_provider(provider_name, scope=scope)
    if raw is None:
        return None
    return LangChromeBrowserAgent(raw, llm=llm)


def resolve_browser_provider(
    config: Optional[dict], llm: "LLMCall", *, scope: Optional[str] = None
) -> Optional[BrowserProvider]:
    """Resolve the active browser provider, wrapping it in a LangChain agent when
    ``browser.use_langchain_agent`` is set.  Always returns a
    :class:`BrowserProvider` (raw or wrapped), so the existing tool surface keeps
    working unchanged::
        >>> prov = resolve_browser_provider({"browser": {"cloud_provider": "browser-use"}}, llm, scope=None)
        >>> isinstance(prov, BrowserProvider)
        True
    """
    from agent.browser_provider import LangChromeBrowserAgent

    configured = None
    if isinstance(config, dict):
        browser = config.get("browser") if isinstance(config, dict) else None
        if isinstance(browser, dict) and "cloud_provider" in browser:
            configured = browser.get("cloud_provider")

    raw = _resolve(configured)
    if raw is None:
        return None
    if _browser_config_for_langchain(config):
        return LangChromeBrowserAgent(raw, llm=llm)
    return raw
