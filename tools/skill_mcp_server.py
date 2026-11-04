#!/usr/bin/env python3
"""Standalone stdio MCP server: skill discovery over ~/.hermes/skills and repo skills.

Exposes two tools:
  skills_list  - name + description (from SKILL.md frontmatter) for each skill dir
  skill_view   - SKILL.md or linked file content, jailed to the skills dirs
"""
import json
import os
import sys
from pathlib import Path

HERMES_SKILLS = Path(os.path.expanduser("~/.hermes/skills"))
REPO_SKILLS = Path(os.path.expanduser("~/Hermes-Agent/skills"))
SKILL_DIRS = [HERMES_SKILLS, REPO_SKILLS]

TOOLS = [
    {
        "name": "skills_list",
        "description": "List skills in ~/.hermes/skills and ~/Hermes-Agent/skills with name and description from SKILL.md frontmatter.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "skill_view",
        "description": "Return SKILL.md content or a linked file by path, jailed to the skills directories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Skill dir name or path under a skills dir (e.g. 'web' or 'mcp-server-manager/SKILL.md').",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]

def _frontmatter(skill_dir):
    md = skill_dir / "SKILL.md"
    if not md.is_file():
        return {}
    text = md.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}
    data = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data

def skills_list():
    out = []
    seen = set()
    for base in SKILL_DIRS:
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            real = str(entry.resolve())
            if real in seen:
                continue
            seen.add(real)
            fm = _frontmatter(entry)
            out.append(
                {
                    "name": entry.name,
                    "description": fm.get("description", ""),
                    "path": real,
                    "source": "hermes" if base == HERMES_SKILLS else "repo",
                }
            )
    return out

def _is_within(child, parent):
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

def _resolve(path_str):
    p = Path(path_str)
    candidates = [p] if p.is_absolute() else [base / p for base in SKILL_DIRS]
    for base in SKILL_DIRS:
        b = base.resolve()
        for cand in candidates:
            rp = cand.resolve()
            if rp == b or _is_within(rp, b):
                return rp
    raise ValueError("path is jailed outside the skills directories")

def skill_view(path_str):
    target = _resolve(path_str)
    if target.is_dir():
        target = target / "SKILL.md"
    if not target.is_file():
        return f"Error: not found: {path_str}"
    return target.read_text(encoding="utf-8")

def _handle(req):
    method = req.get("method")
    params = req.get("params") or {}
    rid = req.get("id")
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            if name == "skills_list":
                result = {"skills": skills_list()}
            elif name == "skill_view":
                result = {"content": skill_view(args.get("path", ""))}
            else:
                raise ValueError("unknown tool " + str(name))
        except ValueError as exc:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(exc)}}
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "unknown method " + str(method)}}

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            resp = _handle(req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
