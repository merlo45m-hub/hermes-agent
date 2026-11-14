"""Tests for :mod:`agent.proxy_config` (Issue #53 - declarative proxy config).

Run: python3 -m pytest tests/agent/test_proxy_config.py -q
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import yaml

from agent.proxy_config import ProxyConfig, ProxyEntry, load, validate


class ProxyEntryTest(unittest.TestCase):
    def test_resolves_env_in_url(self):
        # Missing env var leaves the placeholder untouched (no KeyError).
        entry = ProxyEntry(name="s1", url="https://${UNSET_HOST}/v1")
        self.assertEqual(entry.resolved_url(), "https://${UNSET_HOST}/v1")

    def test_env_var_resolution(self):
        # Provide the var so it resolves.
        os.environ["HERMES_TEST_HOST"] = "env.example"
        try:
            entry = ProxyEntry(name="s1", url="https://${HERMES_TEST_HOST}/v1")
            self.assertEqual(entry.resolved_url(), "https://env.example/v1")
        finally:
            del os.environ["HERMES_TEST_HOST"]

    def test_default_timeout(self):
        entry = ProxyEntry(name="s1", url="https://example.com")
        self.assertEqual(entry.timeout, "30s")


class ProxyConfigTest(unittest.TestCase):
    def test_from_dict(self):
        cfg = ProxyConfig.from_dict({"proxies": [{"name": "p1", "url": "https://a.com", "timeout": "10s"}]})
        self.assertEqual(len(cfg.proxies), 1)
        self.assertEqual(cfg.proxies[0].timeout, "10s")

    def test_validate_ok(self):
        cfg = ProxyConfig.from_dict({"proxies": [{"name": "p1", "url": "https://a.com"}]})
        self.assertEqual(validate(cfg), [])

    def test_validate_missing_url(self):
        cfg = ProxyConfig.from_dict({"proxies": [{"name": "p1", "url": ""}]})
        self.assertTrue(any("url" in p for p in validate(cfg)))

    def test_validate_bad_scheme(self):
        cfg = ProxyConfig.from_dict({"proxies": [{"name": "p1", "url": "ftp://a.com"}]})
        self.assertTrue(any("http" in p for p in validate(cfg)))


class LoadTest(unittest.TestCase):
    def test_load_absent_returns_empty(self):
        # A path that does not exist yields an empty config.
        cfg = load(Path("/nonexistent/proxy.yaml"))
        self.assertEqual(cfg.proxies, ())

    def _write_yaml(self, text: str) -> Path:
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="proxy_test_")
        os.close(fd)
        Path(path).write_text(text, encoding="utf-8")
        return Path(path)

    def test_load_valid(self):
        text = """
proxies:
  - name: primary
    url: https://api.example.com
    verify: /etc/ssl/certs/ca.pem
    timeout: 20s
"""
        p = self._write_yaml(text)
        try:
            cfg = load(p)
            self.assertEqual(len(cfg.proxies), 1)
            self.assertEqual(cfg.proxies[0].name, "primary")
            self.assertEqual(cfg.proxies[0].resolved_url(), "https://api.example.com")
        finally:
            p.unlink(missing_ok=True)

    def test_load_invalid_returns_error(self):
        text = "proxies:\n  - name: p1\n    url: not-a-url\n"
        p = self._write_yaml(text)
        try:
            with self.assertRaises(ValueError):
                load(p)
        finally:
            p.unlink(missing_ok=True)

    def test_from_dict_without_section(self):
        # A config may put proxies directly at top level (no 'proxy' key).
        cfg = ProxyConfig.from_dict({"p1": {"name": "x", "url": "https://a.com"}})
        # from_dict keys off "proxies" list; a bare dict yields no entries.
        self.assertEqual(cfg.proxies, ())


if __name__ == "__main__":
    unittest.main()
