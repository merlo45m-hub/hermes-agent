"""
voice_context_engine.py — Conversational context engine for Layer 1.
Supports game (Muakbank), terminal, and general profiles with state management,
aliases, YAML configuration, and Mnemosyne working memory integration.
"""

import copy
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

# Resolve MCP servers directory to import VoiceInputProcessor domain data
if "/root/mcp-servers" not in sys.path:
    sys.path.insert(0, "/root/mcp-servers")

# --- Host-portable paths. The same plugin runs on the phone (Termux, HOME
# under /data/data/com.termux/files/home) and on the VPS (HOME=/root). Resolve
# the config and Mnemosyne DB against the actual HOME so the context engine
# pulls real memory on BOTH hosts instead of silently degrading to empty.
_HOME = Path(os.environ.get("HOME", "/root"))

CONFIG_PATH = os.environ.get(
    "LAYER1_CONFIG_PATH",
    str(Path(__file__).resolve().parent / "voice_context_config.yaml"),
)
MNEMOSYNE_DB_PATH = os.environ.get(
    "LAYER1_MNEMOSYNE_DB",
    str(_HOME / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"),
)

try:
    from VoiceInputProcessor import (
        WEAPONS,
        WEAPON_ALIASES,
        FOODS,
        FOOD_ALIASES,
        SPECIAL_ABILITIES,
        TARGET_TYPES,
        TARGET_ALIASES,
    )
except ImportError:
    WEAPONS = [
        "fists", "bat", "knife", "pistol", "shotgun", "rifle",
        "axe", "lightsaber", "firegun", "medkit"
    ]
    WEAPON_ALIASES = {
        "gun": "pistol",
        "boomstick": "shotgun",
        "ar": "rifle",
        "ak": "rifle",
        "sidearm": "pistol",
    }
    FOODS = ["burger", "pizza", "fries", "soda", "sushi", "takis"]
    FOOD_ALIASES = {"borger": "burger", "cola": "soda"}
    SPECIAL_ABILITIES = ["rage", "shield", "stealth", "heal", "special"]
    TARGET_TYPES = ["zombie", "boss", "dog", "cat", "runner", "spitter"]
    TARGET_ALIASES = {"undead": "zombie"}


PROFILES: Dict[str, Dict[str, Any]] = {
    "game": {
        "confidence_threshold": 0.65,
        "valid_commands": [
            "ATTACK",
            "JUMP",
            "SPRINT",
            "MOVE",
            "SWITCH_WEAPON",
            "USE_SPECIAL",
            "EAT_FOOD",
            "INTERACT",
            "RELOAD",
        ],
        "aliases": {**WEAPON_ALIASES, **FOOD_ALIASES},
        "suggested_actions": {
            "ATTACK": ["RELOAD", "SWITCH_WEAPON", "SPRINT"],
            "SWITCH_WEAPON": ["ATTACK", "RELOAD"],
            "EAT_FOOD": ["ATTACK", "SPRINT"],
            "MOVE": ["ATTACK", "JUMP", "SPRINT"],
            "DEFAULT": ["ATTACK", "SWITCH_WEAPON", "RELOAD", "JUMP", "SPRINT"],
        },
    },
    "terminal": {
        "confidence_threshold": 0.70,
        "valid_commands": [],  # no command filter
        "aliases": {},
        "suggested_actions": {
            "DEFAULT": ["status", "build", "run", "logs"],
        },
    },
    "general": {
        "confidence_threshold": 0.60,
        "valid_commands": [],  # no filtering
        "aliases": {},
        "suggested_actions": {
            "DEFAULT": ["help", "summary", "next"],
        },
    },
}


def _fetch_mnemosyne_context(db_path: str = MNEMOSYNE_DB_PATH) -> List[str]:
    """
    Pull the latest *fact-like* working-memory entries (tasks, insights, facts).
    Raw conversation logs are skipped — they are already in the chat history and
    injecting them every turn is pure token waste. Graceful empty on failure.
    """
    memory_rows: List[str] = []
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path, timeout=1.0)
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT content FROM working_memory "
                    "WHERE content NOT LIKE '[USER]%' "
                    "AND content NOT LIKE '[conversation]%' "
                    "AND content NOT LIKE '[builtin_memory%' "
                    "AND content NOT LIKE '[assistant]%' "
                    "ORDER BY created_at DESC LIMIT 3"
                )
                for r in cur.fetchall():
                    if r and r[0]:
                        memory_rows.append(str(r[0]).strip()[:160])
            finally:
                conn.close()
    except Exception:
        # Gracefully handle unreachable DB / permissions
        pass
    return memory_rows


class VoiceContextEngine:
    """
    Conversational context engine managing execution profiles,
    domain vocabulary aliases, system state, and memory context.
    """

    def __init__(
        self,
        config_path: str = CONFIG_PATH,
        default_profile: str = "game",
    ):
        self.config_path = config_path
        self.profiles = copy.deepcopy(PROFILES)
        self.current_profile = default_profile
        self.state: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

        self._load_or_create_config()

    def _load_or_create_config(self) -> None:
        """Load YAML configuration or initialize default config file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        loaded_profiles = data.get("profiles", {})
                        for k, v in loaded_profiles.items():
                            if k in self.profiles and isinstance(v, dict):
                                self.profiles[k].update(v)
                            else:
                                self.profiles[k] = v
                        if "default_profile" in data and data["default_profile"] in self.profiles:
                            self.current_profile = data["default_profile"]
            except Exception:
                pass
        else:
            try:
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        {
                            "default_profile": self.current_profile,
                            "profiles": self.profiles,
                        },
                        f,
                        default_flow_style=False,
                    )
            except Exception:
                pass

    def update_state(self, state: Dict[str, Any]) -> None:
        """Updates the internal conversational and application state."""
        if isinstance(state, dict):
            self.state.update(state)

    def switch_profile(self, name: str) -> None:
        """Switches the active profile (e.g. game, terminal, general)."""
        name_lower = name.lower()
        if name_lower not in self.profiles:
            raise ValueError(
                f"Unknown profile '{name}'. Available: {list(self.profiles.keys())}"
            )
        self.current_profile = name_lower

    def add_alias(self, phonetic: str, canonical: str) -> None:
        """Adds a phonetic-to-canonical alias mapping for the active profile."""
        profile_data = self.profiles.setdefault(self.current_profile, {})
        aliases = profile_data.setdefault("aliases", {})
        aliases[phonetic.strip().lower()] = canonical.strip().lower()

    def memory_context(self) -> List[str]:
        """Latest Mnemosyne working-memory entries (graceful empty on failure)."""
        return _fetch_mnemosyne_context()

    def process(self, processor_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches VoiceInputProcessor output with profile validation,
        suggested subsequent commands, and state/artifact metadata.
        """
        result = dict(processor_result)
        profile_data = self.profiles.get(self.current_profile, {})
        valid_commands = profile_data.get("valid_commands", [])
        threshold = profile_data.get("confidence_threshold", 0.65)

        action = result.get("action", "UNKNOWN")
        confidence = float(result.get("confidence", 1.0))
        is_valid_base = bool(result.get("is_valid", False))

        # Check valid flag per profile policy
        if valid_commands:
            # Profile enforces specific command list
            valid_for_profile = (
                is_valid_base
                and (action in valid_commands)
                and (confidence >= threshold)
            )
        else:
            # Profile has no command filter
            valid_for_profile = is_valid_base and (confidence >= threshold)

        # Retrieve suggestions
        suggested_map = profile_data.get("suggested_actions", {})
        suggested = suggested_map.get(action) or suggested_map.get("DEFAULT", [])

        # Fetch Mnemosyne context
        mnemosyne_context = _fetch_mnemosyne_context()

        # Build context artifact
        artifact = {
            "profile": self.current_profile,
            "task": self.state.get("task", ""),
            "state_snapshot": dict(self.state),
            "mnemosyne_memory": mnemosyne_context,
            "aliases_count": len(profile_data.get("aliases", {})),
        }

        result["profile"] = self.current_profile
        result["valid"] = valid_for_profile
        result["suggested"] = list(suggested)
        result["artifact"] = artifact

        # Record history
        self.history.append({
            "action": action,
            "profile": self.current_profile,
            "valid": valid_for_profile,
            "sanitized_text": result.get("sanitized_text", "")
        })

        return result
