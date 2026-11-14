"""Simulated-outage failover tests for :mod:`agent.lightllm_failover`.

Issue #49. The failover manager is exercised against a controlled mock HTTP
layer so no real network is needed: a primary outage, a standby takeover, and a
both-down case.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import httpx

from agent.lightllm_failover import (
    Endpoint,
    FailoverManager,
    LiteLLMFailoverConfig,
    load,
)


def _mock_client(responses):
    """Build a mock httpx.Client whose ``get`` returns pre-programmed responses."""
    mock = MagicMock(spec=httpx.Client)
    mock.get.side_effect = responses
    return mock


def _cfg(primary=None, standby=None):
    return LiteLLMFailoverConfig(
        primary=primary
        or Endpoint(model="p", base_url="https://primary:8080", health_path="/health"),
        standby=standby
        or Endpoint(model="s", base_url="https://standby:8080", health_path="/health"),
    )


class FailoverSelectTest(unittest.TestCase):
    def test_primary_healthy_returns_primary(self):
        primary = Endpoint(model="p", base_url="https://primary:8080", health_path="/health")
        standby = Endpoint(model="s", base_url="https://standby:8080", health_path="/health")
        client = _mock_client([MagicMock(status_code=200)])
        manager = FailoverManager(_cfg(primary, standby), client=client)
        self.assertEqual(manager.select_endpoint(), "https://primary:8080")

    def test_primary_down_standby_healthy_returns_standby(self):
        primary = Endpoint(model="p", base_url="https://primary:8080", health_path="/health")
        standby = Endpoint(model="s", base_url="https://standby:8080", health_path="/health")
        # primary -> 503, standby -> 200
        client = _mock_client([MagicMock(status_code=503), MagicMock(status_code=200)])
        manager = FailoverManager(_cfg(primary, standby), client=client)
        self.assertEqual(manager.select_endpoint(), "https://standby:8080")

    def test_primary_down_connection_error_standby_up_returns_standby(self):
        primary = Endpoint(model="p", base_url="https://primary:8080", health_path="/health")
        standby = Endpoint(model="s", base_url="https://standby:8080", health_path="/health")
        # primary connection error, standby -> 200
        client = _mock_client([ConnectionError("boom"), MagicMock(status_code=200)])
        manager = FailoverManager(_cfg(primary, standby), client=client)
        self.assertEqual(manager.select_endpoint(), "https://standby:8080")

    def test_both_down_returns_none(self):
        primary = Endpoint(model="p", base_url="https://primary:8080", health_path="/health")
        standby = Endpoint(model="s", base_url="https://standby:8080", health_path="/health")
        client = _mock_client([ConnectionError("primary boom"), MagicMock(status_code=404)])
        manager = FailoverManager(_cfg(primary, standby), client=client)
        self.assertIsNone(manager.select_endpoint())

    def test_disabled_entries_skipped(self):
        primary = Endpoint(model="p", base_url="https://primary:8080", health_path="/health", enabled=True)
        standby = Endpoint(model="s", base_url="https://standby:8080", health_path="/health", enabled=False)
        client = _mock_client([MagicMock(status_code=200)])
        manager = FailoverManager(_cfg(primary, standby), client=client)
        self.assertEqual(manager.select_endpoint(), "https://primary:8080")


class ConfigTest(unittest.TestCase):
    def test_from_dict(self):
        cfg = LiteLLMFailoverConfig.from_dict(
            {
                "primary": {"model": "gpt", "base_url": "https://p", "timeout": 5},
                "standby": {"model": "s", "base_url": "https://s"},
            }
        )
        self.assertEqual(cfg.primary.model, "gpt")
        self.assertEqual(cfg.primary.timeout, 5.0)
        self.assertEqual(cfg.standby.base_url, "https://s")
        self.assertEqual(cfg.standby.health_path, "/health")
        self.assertEqual(cfg.probe_timeout, 10.0)
        self.assertEqual(cfg.failover_reasons, ("connection_error", "http_5xx", "rate_limit"))

    def test_from_dict_empty(self):
        cfg = LiteLLMFailoverConfig.from_dict({})
        self.assertIsNone(cfg.primary)
        self.assertIsNone(cfg.standby)

    def test_validate_ok(self):
        cfg = LiteLLMFailoverConfig.from_dict(
            {
                "primary": {"model": "p", "base_url": "https://primary:8080"},
                "standby": {"model": "s", "base_url": "https://standby:8080"},
            }
        )
        self.assertEqual(cfg.validate(), [])

    def test_validate_missing_primary(self):
        cfg = LiteLLMFailoverConfig.from_dict(
            {"standby": {"model": "s", "base_url": "https://s"}}
        )
        self.assertTrue(any("primary" in p for p in cfg.validate()))

    def test_validate_bad_url(self):
        cfg = LiteLLMFailoverConfig.from_dict(
            {"primary": {"model": "p", "base_url": "ftp://p"}, "standby": {"model": "s", "base_url": "https://s"}}
        )
        self.assertTrue(any("http" in p for p in cfg.validate()))

    def test_resolve_env_url(self):
        import os

        os.environ["HERMES_FB_HOST"] = "env.example"
        try:
            self.assertEqual(
                Endpoint(model="p", base_url="https://${HERMES_FB_HOST}/v1").health_url(),
                "https://env.example/v1/health",
            )
        finally:
            del os.environ["HERMES_FB_HOST"]


class LoadTest(unittest.TestCase):
    def test_load_nonexistent_returns_empty(self):
        cfg = load(Path("/nonexistent/failover.yaml"))
        self.assertIsInstance(cfg, LiteLLMFailoverConfig)

    def test_load_missing_file_returns_default(self):
        # A path that does not exist yields a default config (no-op), mirroring
        # the proxy_config convention.
        cfg = load(Path("/tmp/does-not-exist-failover-12345.yaml"))
        self.assertIsInstance(cfg, LiteLLMFailoverConfig)


if __name__ == "__main__":
    unittest.main()
