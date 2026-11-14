"""LiteLLM failover configuration and probe.

Issue #49. Declares a hot-standby failover block for the LiteLLM gateway path
(Layer 7): a primary inference endpoint plus a standby that is promoted when the
primary is unreachable. Exposes a typed config model (YAML-loadable), a health
probe, and a failover manager that selects the healthy endpoint at connect-time.

Design: dataclasses + stdlib + httpx for the probe. No new runtime deps beyond
what the gateway already uses.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

__all__ = [
    "LiteLLMFailoverConfig",
    "Endpoint",
    "LiteLLMFailoverProbe",
    "FailoverManager",
    "load",
    "validate",
]

_ENV_RE = re.compile(r"\$\{(?P<name>[A-Z][A-Z0-9_]*)\}")


def resolve_env_url(text: str) -> str:
    """Expand ``$VAR`` / ``${VAR}`` placeholders in a URL from the environment."""
    if not isinstance(text, str):
        return text
    return _ENV_RE.sub(lambda m: os.environ.get(m.group("name"), m.group(0)), text)


@dataclass(frozen=True)
class Endpoint:
    """One LiteLLM inference endpoint (primary or standby)."""

    model: str
    base_url: str
    health_path: str = "/health"
    timeout: float = 10.0
    enabled: bool = True

    def health_url(self) -> str:
        return resolve_env_url(f"{self.base_url.rstrip('/')}{self.health_path}")


@dataclass(frozen=True)
class LiteLLMFailoverConfig:
    """Top-level failover block: primary + standby + policy."""

    primary: Optional[Endpoint] = None
    standby: Optional[Endpoint] = None
    probe_timeout: float = 10.0
    failover_reasons: tuple[str, ...] = ("connection_error", "http_5xx", "rate_limit")
    standby_priority: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LiteLLMFailoverConfig":
        if not data:
            return cls()
        primary = data.get("primary")
        standby = data.get("standby")
        return cls(
            primary=(
                Endpoint(
                    model=str(primary.get("model", "")),
                    base_url=str(primary.get("base_url", "")),
                    health_path=str(primary.get("health_path", "/health")),
                    timeout=float(primary.get("timeout", 10.0)),
                    enabled=bool(primary.get("enabled", True)),
                )
                if isinstance(primary, dict)
                else None
            ),
            standby=(
                Endpoint(
                    model=str(standby.get("model", "")),
                    base_url=str(standby.get("base_url", "")),
                    health_path=str(standby.get("health_path", "/health")),
                    timeout=float(standby.get("timeout", 10.0)),
                    enabled=bool(standby.get("enabled", True)),
                )
                if isinstance(standby, dict)
                else None
            ),
            probe_timeout=float(data.get("probe_timeout", 10.0)),
            failover_reasons=tuple(
                str(r) for r in data.get("failover_reasons", ["connection_error", "http_5xx", "rate_limit"])
            ),
            standby_priority=int(data.get("standby_priority", 1)),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        for label, entry in (("primary", self.primary), ("standby", self.standby)):
            if entry is None:
                problems.append(f"failover {label} endpoint is required")
                continue
            if not entry.model:
                problems.append(f"failover {label} missing model")
            if not entry.base_url:
                problems.append(f"failover {label} missing base_url")
            elif not entry.base_url.startswith(("http://", "https://")):
                problems.append(f"failover {label} base_url must start with http:// or https://")
        return problems


def load(path: Optional[Path] = None) -> LiteLLMFailoverConfig:
    """Load and validate a failover YAML file; ``path`` defaults to the ``failover:``
    block of ``<hermes_home>/config.yaml``. Raises on missing/invalid."""
    if path is None:
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "config.yaml"
    if not path.is_file():
        return LiteLLMFailoverConfig()
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    failover_data = data.get("failover") or {}
    cfg = LiteLLMFailoverConfig.from_dict(failover_data)
    problems = cfg.validate()
    if problems:
        raise ValueError("invalid failover config: " + "; ".join(problems))
    return cfg


class LiteLLMFailoverProbe:
    """HTTP health probe for a single LiteLLM endpoint."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def probe(self, endpoint: Endpoint) -> tuple[bool, str]:
        """Return ``(reachable, detail)`` for ``endpoint``."""
        try:
            resp = self._client.get(endpoint.health_url(), timeout=endpoint.timeout)
            ok = 200 <= resp.status_code < 400
            detail = f"http {resp.status_code}"
            return ok, detail
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"


class FailoverManager:
    """Select the healthy endpoint at connect-time.

    Probes primary first, then standby. Returns the base_url of the first
    healthy endpoint, or ``None`` when neither is reachable.
    """

    def __init__(
        self,
        config: LiteLLMFailoverConfig,
        *,
        client: Optional[httpx.Client] = None,
        client_class: Any = httpx.Client,
        **client_kwargs: Any,
    ) -> None:
        self.config = config
        self._client_class = client_class
        self._client_kwargs = client_kwargs
        self._client = client

    def _new_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return self._client_class(timeout=self.config.probe_timeout)

    def select_endpoint(self) -> Optional[str]:
        probe = LiteLLMFailoverProbe(self._new_client())
        for entry in (self.config.primary, self.config.standby):
            if not entry or not entry.enabled:
                continue
            ok, _detail = probe.probe(entry)
            if ok:
                return entry.base_url
        return None
