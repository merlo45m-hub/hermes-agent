"""Configurable Voice Context Engine (Layer 1, Phase 4).

Customizable to the user's architecture:
- Per-application context profiles (game, terminal, general)
- Mnemosyne working memory integration
- Configurable artifact mappings, valid commands, thresholds
- Context providers (game state, Mnemosyne, tool context)

Configuration via ~/.hermes/voice_context_config.yaml or environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.expanduser("~/.hermes/voice_context_config.yaml")

DEFAULT_CONFIG = {
    "active_profile": "game",
    "profiles": {
        "game": {
            "name": "Game (Muakbank Zombie)",
            "valid_commands": ["attack", "switch", "reload", "heal", "sprint", "equip", "fire", "open", "use"],
            "artifact_mappings": {
                "mk deer": "mkdir",
                "pie thon": "python",
                "shot gun": "shotgun",
                "med kit": "medkit",
                "re load": "reload",
                "heal th": "heal",
            },
            "destructive_commands": ["rm", "rmdir", "chmod", "sudo", "mkfs", "dd", "format", "del"],
            "confidence_threshold": 0.65,
            "context_sources": ["game_state", "mnemosyne"],
        },
        "terminal": {
            "name": "Terminal/Shell",
            "valid_commands": [],
            "artifact_mappings": {
                "mk deer": "mkdir",
                "pie thon": "python",
                "cd dot": "cd ..",
                "less s": "ls",
            },
            "destructive_commands": ["rm", "rmdir", "chmod", "sudo", "mkfs", "dd", "format", "del", "kill"],
            "confidence_threshold": 0.70,
            "context_sources": ["shell_state", "mnemosyne"],
        },
        "general": {
            "name": "General Purpose",
            "valid_commands": [],
            "artifact_mappings": {},
            "destructive_commands": ["rm", "rmdir", "chmod", "sudo", "mkfs", "dd", "format", "del"],
            "confidence_threshold": 0.60,
            "context_sources": ["mnemosyne"],
        },
    },
    "mnemosyne": {
        "enabled": True,
        "working_memory_keys": ["current_task", "current_location", "current_weapon"],
        "recall_limit": 5,
    },
    "rewriter": {
        "enabled": True,
        "primary_provider": "nvidia_nim",
        "fallback_provider": "openrouter",
        "temperature": 0,
        "max_tokens": 100,
        "reasoning_effort": "none",
        "timeout_seconds": 10,
        "max_retries": 1,
        "cache_enabled": True,
    },
}


@dataclass
class GameContext:
    """Current application state for context-aware voice processing."""

    current_weapon: str = ""
    current_location: str = ""
    current_health: int = 100
    current_task: str = ""
    last_command: str = ""
    current_directory: str = ""
    current_tool: str = ""
    valid_commands: list[str] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)


class ContextEngine:
    """Configurable context engine for voice input sanitization."""

    def __init__(self, config: dict = None):
        self.config = config or self._load_config()
        self._context = GameContext()
        self._profile_name = self.config.get("active_profile", "game")
        self._profile = self.config["profiles"].get(self._profile_name, self.config["profiles"]["general"])
        self._mnemosyne_enabled = self.config.get("mnemosyne", {}).get("enabled", False)
        self._rewriter_enabled = self.config.get("rewriter", {}).get("enabled", True)

    def _load_config(self) -> dict:
        try:
            import yaml
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH) as f:
                    loaded = yaml.safe_load(f)
                if loaded:
                    merged = DEFAULT_CONFIG.copy()
                    merged.update(loaded)
                    return merged
        except ImportError:
            logger.warning("PyYAML not available, using default config")
        except Exception as e:
            logger.warning("Failed to load config: %s, using defaults", e)
        return DEFAULT_CONFIG.copy()

    def save_config(self) -> None:
        try:
            import yaml
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info("Config saved to %s", CONFIG_PATH)
        except ImportError:
            logger.warning("PyYAML not available, cannot save config")
        except Exception as e:
            logger.warning("Failed to save config: %s", e)

    @property
    def profile(self) -> dict:
        return self._profile

    @property
    def profile_name(self) -> str:
        return self._profile_name

    def switch_profile(self, name: str) -> bool:
        if name in self.config["profiles"]:
            self._profile_name = name
            self._profile = self.config["profiles"][name]
            self.config["active_profile"] = name
            logger.info("Switched to profile: %s", name)
            return True
        logger.warning("Unknown profile: %s", name)
        return False

    def set_context(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._context, key):
                setattr(self._context, key, value)
            else:
                self._context.working_memory[key] = value

    def get_context(self) -> GameContext:
        return self._context

    def reset_context(self) -> None:
        self._context = GameContext()

    def build_context_prompt(self) -> str:
        ctx = self._context
        parts = []
        if ctx.current_weapon:
            parts.append(f"Current weapon: {ctx.current_weapon}")
        if ctx.current_location:
            parts.append(f"Current location: {ctx.current_location}")
        if ctx.current_health < 100:
            parts.append(f"Health: {ctx.current_health}%")
        if ctx.current_task:
            parts.append(f"Current task: {ctx.current_task}")
        if ctx.last_command:
            parts.append(f"Last command: {ctx.last_command}")
        if ctx.current_directory:
            parts.append(f"Current directory: {ctx.current_directory}")
        if ctx.current_tool:
            parts.append(f"Active tool: {ctx.current_tool}")
        vcs = ctx.valid_commands or self._profile.get("valid_commands", [])
        if vcs:
            parts.append(f"Valid commands: {', '.join(vcs)}")
        if ctx.working_memory:
            for key, value in ctx.working_memory.items():
                parts.append(f"{key}: {value}")
        if self._mnemosyne_enabled:
            try:
                mnemo_ctx = self._pull_mnemosyne_context()
                if mnemo_ctx:
                    parts.append(f"Mnemosyne context: {mnemo_ctx}")
            except Exception as e:
                logger.debug("Mnemosyne context pull failed: %s", e)
        return "\n".join(parts) if parts else "No context available"

    def _pull_mnemosyne_context(self) -> str:
        try:
            import mnemosyne
            return None
        except ImportError:
            return None

    def is_artifact(self, text: str) -> bool:
        mappings = self._profile.get("artifact_mappings", {})
        return text.lower().strip() in mappings

    def get_suggestion(self, text: str) -> str:
        mappings = self._profile.get("artifact_mappings", {})
        return mappings.get(text.lower().strip())

    def is_destructive(self, command: str) -> bool:
        destructive = self._profile.get("destructive_commands", [])
        cmd_lower = command.lower().strip()
        for dc in destructive:
            if cmd_lower.startswith(dc) or f" {dc} " in cmd_lower or f" {dc}\t" in cmd_lower:
                return True
        return False

    def validate_command(self, command: str) -> bool:
        if not command or not command.strip():
            return False
        vcs = self._context.valid_commands or self._profile.get("valid_commands", [])
        if vcs:
            cmd_base = command.strip().split()[0].lower()
            if cmd_base not in [c.lower() for c in vcs]:
                return False
        if self.is_destructive(command):
            return False
        return True

    def enhance(self, raw: str) -> dict:
        return {
            "raw": raw,
            "context": self.build_context_prompt(),
            "is_artifact": self.is_artifact(raw),
            "is_valid": self.validate_command(raw),
            "suggested": self.get_suggestion(raw),
            "profile": self._profile_name,
        }

    def to_dict(self) -> dict:
        return {
            "profile": self._profile_name,
            "context": {
                "current_weapon": self._context.current_weapon,
                "current_location": self._context.current_location,
                "current_health": self._context.current_health,
                "current_task": self._context.current_task,
                "last_command": self._context.last_command,
                "current_directory": self._context.current_directory,
                "current_tool": self._context.current_tool,
                "valid_commands": self._context.valid_commands,
                "working_memory": self._context.working_memory,
            },
            "config": self.config,
        }

    def add_artifact_mapping(self, artifact: str, correction: str) -> None:
        """Add a new artifact mapping to the current profile."""
        self._profile.setdefault("artifact_mappings", {})[artifact.lower()] = correction
        self.config["profiles"][self._profile_name] = self._profile

    def add_valid_command(self, command: str) -> None:
        """Add a valid command to the current profile."""
        vcs = self._profile.setdefault("valid_commands", [])
        if command not in vcs:
            vcs.append(command)
        self.config["profiles"][self._profile_name] = self._profile

    def add_destructive_command(self, command: str) -> None:
        """Add a destructive command to the current profile."""
        dcs = self._profile.setdefault("destructive_commands", [])
        if command not in dcs:
            dcs.append(command)
        self.config["profiles"][self._profile_name] = self._profile


_engine: "ContextEngine | None" = None


def get_engine() -> ContextEngine:
    global _engine
    if _engine is None:
        _engine = ContextEngine()
    return _engine


def configure(profile: str = None, **kwargs) -> ContextEngine:
    engine = get_engine()
    if profile:
        engine.switch_profile(profile)
    if kwargs:
        engine.config.update(kwargs)
    return engine


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voice Context Engine")
    parser.add_argument("--profile", choices=["game", "terminal", "general"], help="Active profile")
    parser.add_argument("--show-config", action="store_true", help="Show current config")
    parser.add_argument("--save-config", action="store_true", help="Save default config to file")
    args = parser.parse_args()

    engine = ContextEngine()
    if args.profile:
        engine.switch_profile(args.profile)
    if args.save_config:
        engine.save_config()
        print(f"Config saved to {CONFIG_PATH}")
    if args.show_config:
        print(json.dumps(engine.config, indent=2))
    else:
        engine.set_context(
            current_weapon="shotgun",
            current_location="zombie_forest",
            current_health=75,
            current_task="clearing zombies",
        )
        print("Context prompt:")
        print(engine.build_context_prompt())
        print()
        for inp in ["attack zombie", "switch to rifle", "mk deer", "heal", "rm -rf /"]:
            r = engine.enhance(inp)
            print(f"Input: {inp!r} -> artifact={r['is_artifact']}, valid={r['is_valid']}, suggested={r['suggested']}")