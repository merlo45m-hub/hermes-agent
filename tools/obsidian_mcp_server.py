#!/usr/bin/env python3
"""Standalone stdio JSON-RPC MCP server for an Obsidian vault.

Exposes three tools:
  obsidian_list   - list markdown notes under a vault subpath, newest first
                    (name, size, mtime)
  obsidian_read   - read a note by path, jailed to the vault root, capped at 8000 chars
  obsidian_write  - write a new note under projects/ only; never overwrite an
                    existing note without the overwrite flag
"""
import json
import os
import sys
from pathlib import Path

VAULT_ROOT = Path(os.path.expanduser(
    "/storage/emulated/0/ObsidianVault/hermes_brain/second_brain")).resolve()
MAX_READ_CHARS = 8000
WRITE_JAIL = "projects"

TOOLS = [
    {
        "name": "obsidian_list",
        "description": "List markdown notes under a vault subpath (relative to the vault root), newest first, with name, size and mtime.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subpath": {
                    "type": "string",
                    "description": "Vault-relative subpath to list (default: vault root).",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "obsidian_read",
        "description": "Read a note by vault-relative path, jailed to the vault root, content capped at 8000 chars.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Vault-relative path of the note to read (e.g. 'projects/ice-tracker.md').",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "obsidian_write",
        "description": "Write a new note under projects/ only. Fails if the note exists unless overwrite=true.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Vault-relative path under projects/ (e.g. 'projects/new-note.md').",
                },
                "content": {
                    "type": "string",
                    "description": "Markdown content to write.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Allow overwriting an existing note (default false).",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
]


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_vault(path_str: str) -> Path:
    """Resolve a vault-relative path, jailed strictly inside the vault root."""
    p = Path(path_str)
    if p.is_absolute():
        raise ValueError("absolute paths are not allowed; use a vault-relative path")
    target = (VAULT_ROOT / p).resolve()
    if not _is_within(target, VAULT_ROOT):
        raise ValueError("path is jailed outside the vault root")
    return target


def _fmt_mtime(path: Path) -> str:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def obsidian_list(subpath: str = ""):
    target = _resolve_vault(subpath) if subpath else VAULT_ROOT
    if not target.is_dir():
        return {"notes": [], "error": f"not a directory: {subpath or '.'}"}
    notes = []
    for entry in target.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".md":
            continue
        st = entry.stat()
        notes.append(
            {
                "name": entry.name,
                "path": str(entry.relative_to(VAULT_ROOT)),
                "size": st.st_size,
                "mtime": st.st_mtime,
            }
        )
    notes.sort(key=lambda n: n["mtime"], reverse=True)
    for n in notes:
        n["mtime"] = round(n["mtime"], 3)
    return {"notes": notes, "count": len(notes)}


def obsidian_read(path_str: str):
    target = _resolve_vault(path_str)
    if not target.is_file():
        return {"error": f"not found: {path_str}"}
    text = target.read_text(encoding="utf-8")
    truncated = False
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS]
        truncated = True
    return {"content": text, "chars": len(text), "truncated": truncated}


def obsidian_write(path_str: str, content: str, overwrite: bool = False):
    target = _resolve_vault(path_str)
    # Write jail: only under projects/
    rel = target.relative_to(VAULT_ROOT)
    parts = rel.parts
    if not parts or parts[0] != WRITE_JAIL:
        raise ValueError(f"writes are jailed to '{WRITE_JAIL}/' only")
    if target.is_file() and not overwrite:
        return {"error": f"note already exists: {path_str} (set overwrite=true to replace)"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"written": str(rel), "bytes": len(content.encode("utf-8")), "overwrite": overwrite}


def _handle(req):
    method = req.get("method")
    params = req.get("params") or {}
    rid = req.get("id")
    if method == "initialize":
        # Echo the client's offered protocol version so the SDK's handshake
        # accepts the result; expose the tools capability.
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": params.get("protocolVersion") or "2025-06-18",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "obsidian", "version": "1.0.0"},
            },
        }
    if method in ("notifications/initialized", "initialized", "notifications/cancelled"):
        return None  # notification: no response
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "server/discover":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {"capabilities": {"tools": {"listChanged": False}}},
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            if name == "obsidian_list":
                result = obsidian_list(args.get("subpath", ""))
            elif name == "obsidian_read":
                result = obsidian_read(args.get("path", ""))
            elif name == "obsidian_write":
                result = obsidian_write(
                    args.get("path", ""),
                    args.get("content", ""),
                    bool(args.get("overwrite", False)),
                )
            else:
                raise ValueError("unknown tool " + str(name))
        except ValueError as exc:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(exc)}}
        return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(result)}]}}
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
        if resp is None:
            continue  # notification: nothing to send
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
