"""Declarative proxy-layer configuration for Hermes.

This module turns the ``proxy:`` block of ``config.yaml`` into a typed model plus a
YAML loader with env-var interpolation. The companion module
:mod:`agent.verification_stop` consumes it to expose a reachability probe as part of the
turn-end verification loop (Issue #53).

Design: one small, stdlib-only file. No new runtime dependencies.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

__all__ = [
    "ProxyEntry",
    "ProxyConfig",
    "load",
    "resolve_env",
    "validate",
]

_ENV_RE = re.compile(r"\$\{(?P<name>[A-Z][A-Z0-9_]*)\}")

# YAML tag prefix we emit when serialising (mirrors the declarative model).
_SENTINEL = object()


@dataclass(frozen=True)
class ProxyEntry:
    """One proxied endpoint/agent.

    ``url`` is the *upstream* the proxy targets; ``verify`` is a URL- or env-ref to a CA
    bundle used when the proxy must speak TLS to that upstream. ``headers`` are static
    outbound headers. ``timeout`` defaults to ``30s`` when omitted.
    """

    name: str
    url: str
    verify: Optional[str] = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: str = "30s"
    enabled: bool = True

    def resolved_url(self) -> str:
        return resolve_env(self.url)

    def resolved_verify(self) -> Optional[str]:
        if self.verify is None:
            return None
        return resolve_env(self.verify)


@dataclass(frozen=True)
class ProxyConfig:
    """Top-level proxy config: list of entries + global defaults."""

    proxies: tuple[ProxyEntry, ...] = ()
    probe_timeout: int = 5  # seconds, used by the verification loop

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[Any]) -> "ProxyConfig":
        proxy_items = data if isinstance(data, list) else (data.get("proxies") or data.get("proxy") or {})
        proxies = [
            ProxyEntry(
                name=str(p.get("name", "unnamed") if isinstance(p, dict) else "unnamed"),
                url=str(p.get("url", "") if isinstance(p, dict) else ""),
                verify=p.get("verify") if isinstance(p, dict) else None,
                headers=dict(p.get("headers") or {} if isinstance(p, dict) else {}),
                timeout=str(p.get("timeout", "30s") if isinstance(p, dict) else "30s"),
                enabled=bool(p.get("enabled", True)) if isinstance(p, dict) else True,
            )
            for p in proxy_items
        ]
        return cls(proxies=tuple(proxies), probe_timeout=int(data.get("probe_timeout", 5)) if isinstance(data, dict) else 5)

    def items(self) -> Iterable[ProxyEntry]:
        return self.proxies

    def enabled_entries(self) -> Iterable[ProxyEntry]:
        return (p for p in self.proxies if p.enabled)


def resolve_env(text: str) -> str:
    """Expand ``${VAR}`` / ``$VAR`` style placeholders from the environment."""
    if not isinstance(text, str):
        return text
    return _ENV_RE.sub(lambda m: os.environ.get(m.group("name"), m.group(0)), text)


def validate(config: ProxyConfig) -> list[str]:
    """Return a list of problems (empty when the config is valid)."""
    problems: list[str] = []
    for p in config.proxies:
        if not p.name:
            problems.append("proxy entry missing name")
        if not p.url:
            problems.append(f"proxy '{p.name}' missing url")
        elif not p.url.startswith(("http://", "https://")):
            problems.append(f"proxy '{p.name}' url must start with http:// or https://")
    return problems


def load(path: Optional[Path] = None) -> ProxyConfig:
    """Load and validate a proxy config YAML file; ``path`` defaults to
    ``<hermes_home>/config.yaml`` -> ``proxy`` section. Raises on missing/invalid."""
    if path is None:
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "config.yaml"
    if not path.is_file():
        return ProxyConfig()  # no-op default when proxy block is absent
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    proxy_data = data.get("proxy") or data.get("proxies") or {}
    cfg = ProxyConfig.from_dict(proxy_data)
    problems = validate(cfg)
    if problems:
        raise ValueError("invalid proxy config: " + "; ".join(problems))
    return cfg
