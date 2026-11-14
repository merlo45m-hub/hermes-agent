"""Validate the LightLLM failover Ansible role tree (Issue #55).

No ansible is required: every YAML file is parsed with PyYAML and checked for the
structural shape the roles expect. Jinja templates (``.j2``) are deliberately
skipped, and the inventory is an INI file, not YAML — both are validated
manually. Exits non-zero on the first problem found.

Usage:
    python3 ansible/validate.py [--dir ansible]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# (relative path, kind, required keys) where kind is "mapping" or "list".
# - mapping files must decode to a dict with the listed keys.
# - list files must decode to a list of dicts each carrying a "name" key.
YAML_FILES = [
    ("roles/lightllm_primary/defaults/main.yml", "mapping", {"lightllm_primary_host", "lightllm_primary_port"}),
    ("roles/lightllm_primary/vars/main.yml", "mapping", {"lightllm_service_name"}),
    ("roles/lightllm_primary/tasks/main.yml", "list", None),
    ("roles/lightllm_primary/handlers/main.yml", "list", None),
    ("roles/lightllm_primary/meta/main.yml", "mapping", {"galaxy_info"}),
    ("roles/lightllm_standby/defaults/main.yml", "mapping", {"lightllm_standby_port"}),
    ("roles/lightllm_standby/vars/main.yml", "mapping", {"lightllm_service_name"}),
    ("roles/lightllm_standby/tasks/main.yml", "list", None),
    ("roles/lightllm_standby/handlers/main.yml", "list", None),
    ("roles/lightllm_standby/meta/main.yml", "mapping", {"galaxy_info"}),
    ("site.yml", "list", {"name", "hosts"}),
]


def check_yaml(path: Path, kind: str, required: set[str] | None) -> list[str]:
    problems: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: unreadable: {exc}"]

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return [f"{path}: invalid YAML: {exc}"]

    if kind == "mapping":
        if not isinstance(data, dict):
            return [f"{path}: top-level YAML is not a mapping"]
        if required:
            for key in required:
                if key not in data:
                    problems.append(f"{path}: missing key {key!r}")
    elif kind == "list":
        if not isinstance(data, list):
            return [f"{path}: top-level YAML is not a list"]
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                problems.append(f"{path}: item {i} is not a mapping")
            else:
                if "name" not in item:
                    problems.append(f"{path}: item {i} missing key 'name'")
                if required:
                    for key in required:
                        if key not in item:
                            problems.append(f"{path}: item {i} missing key {key!r}")
    else:
        problems.append(f"{path}: unknown kind {kind!r}")
    return problems


def check_site_plays(path: Path) -> list[str]:
    """site.yml is a list of play dicts, not a mapping."""
    problems: list[str] = []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"{path}: invalid YAML: {exc}"]

    if not isinstance(data, list) or not data:
        return [f"{path}: expected a non-empty list of plays"]
    for i, play in enumerate(data):
        if not isinstance(play, dict):
            problems.append(f"{path}: play {i} is not a mapping")
        else:
            for key in ("name", "hosts"):
                if key not in play:
                    problems.append(f"{path}: play {i} missing key {key!r}")
    return problems


def check_inventory(path: Path) -> list[str]:
    """inventory.example is INI, not YAML: confirm known group headers exist."""
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: unreadable: {exc}"]
    for group in ("[lightllm_primary]", "[lightllm_standby]"):
        if group not in text:
            problems.append(f"{path}: missing group header {group}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the LightLLM Ansible role tree.")
    parser.add_argument("--dir", default="ansible", help="Ansible root (default: ansible)")
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    problems: list[str] = []

    for rel, kind, required in YAML_FILES:
        f = root / rel
        if not f.is_file():
            problems.append(f"missing file: {f}")
            continue
        problems.extend(check_yaml(f, kind, required))

    # site.yml is a list of plays, not a mapping.
    problems.extend(check_site_plays(root / "site.yml"))

    # inventory.example is INI, not YAML.
    problems.extend(check_inventory(root / "inventory.example"))

    # Jinja templates are rendered, not YAML - skip them (checked for existence
    # only). Each role owns its own two templates.
    tdir = root / "roles"
    template_specs = {
        "lightllm_primary": ["lightllm-primary.service.j2", "lightllm-primary.yaml.j2"],
        "lightllm_standby": ["lightllm-standby.service.j2", "lightllm-standby.yaml.j2"],
    }
    for role, tmpl_names in template_specs.items():
        for tmpl in tmpl_names:
            f = tdir / role / "templates" / tmpl
            if not f.is_file():
                problems.append(f"missing template: {f}")

    if problems:
        print(f"validation failed: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"Validation passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
