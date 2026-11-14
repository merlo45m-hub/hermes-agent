"""Regex sanity audit.

Issue #49. Walks a tree of Python files and reports every ``re.compile(...)``
call, flagging patterns that are risky for catastrophic backtracking or malformed
character classes. Purpose: a cheap, repeatable audit of the regex surface that
the failover / config / routing code depends on.

Usage:
    python3 scripts/regex_audit.py [--path .]

Example:
    python3 scripts/regex_audit.py --path agent/
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

# (compiled pattern, label) pairs fed to re.compile to sanity-check a measured
# pattern string. Each must compile cleanly in Python 3.14.
_RISKY = [
    # Quantifier directly around a non-capturing group: the classic
    # catastrophic-backtracking shape (e.g. (?:[a-z]+)+).
    (re.compile(r"\(\?:[^()]*\+\)"), "nested quantifier"),
    # Quantifier over an ungrouped run: unbounded repeat of a literal run.
    (re.compile(r"[a-z0-9]+\+"), "unbounded quantifier"),
    # Unbounded .* / .+ (wildcard repeat) — flagged as info unless nested.
    (re.compile(r"\.\*"), "unbounded .* (info)"),
    # Ambiguous alternation: '|' inside a group with no anchoring.
    (re.compile(r"\|[^|]*\|"), "ambiguous alternation"),
    # Unclosed character class: '[' with nothing matching ']' before end.
    (re.compile(r"\[[^\]]*$"), "unclosed char class"),
]

# Unanchored ungreedy-ish substrings reported as info, not error.
_INFO = [
    (re.compile(r"\.\?"), "ungreedy .*? (info)"),
    (re.compile(r"\(\?#"), "verbose mode comment (info)"),
]


def _risk_flags(pattern: str) -> list[str]:
    flags: list[str] = []
    for rx, label in _RISKY:
        if rx.search(pattern):
            flags.append(label)
    for rx, label in _INFO:
        if rx.search(pattern):
            flags.append(label + " (info)")
    return flags


def _follow_dot_star(node: ast.expr) -> str:
    """Best-effort string conversion of the first arg of re.compile."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return ast.unparse(node)
    return ast.unparse(node)


def audit_file(path: Path) -> list[dict[str, object]]:
    """Return a list of findings for one Python file."""
    findings: list[dict[str, object]] = []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        findings.append({"file": str(path), "line": 0, "kind": "read_error", "message": str(exc)})
        return findings

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        findings.append({"file": str(path), "line": 0, "kind": "syntax_error", "message": str(exc)})
        return findings

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "compile" and isinstance(node.func.value, ast.Name) and node.func.value.id == "re":
                if node.args:
                    first = node.args[0]
                    try:
                        pattern = _follow_dot_star(first)
                    except Exception:
                        pattern = "<unparseable>"
                    flags = _risk_flags(pattern)
                    severity = "error" if flags else "ok"
                    findings.append(
                        {
                            "file": str(path),
                            "line": node.lineno,
                            "kind": "re.compile",
                            "pattern": pattern,
                            "severity": severity,
                            "flags": flags,
                        }
                    )
    return findings


def audit_tree(root: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    py_files = sorted(root.rglob("*.py"))
    for f in py_files:
        findings.extend(audit_file(f))
    findings.sort(key=lambda x: (str(x["file"]), int(x["line"])))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit re.compile(...) patterns in Python files.")
    parser.add_argument("--path", default=".", help="Root directory to scan (default: .)")
    parser.add_argument("--pattern", default="re.compile", help="Call name to audit (default: re.compile)")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    findings = audit_tree(root)
    issues = [f for f in findings if f["severity"] == "error"]

    print(f"Regex audit of {root}")
    compiled = [f for f in findings if f["kind"] == "re.compile"]
    print(f"files scanned: {len(findings)}")
    print(f"findings: {len(compiled)}")
    print(f"risky: {len(issues)}")
    print("-" * 72)
    for f in findings:
        if f["severity"] == "error":
            print(f"[{f['severity']:>7}] {f['file']}:{f['line']}  {f['kind']}")
            print(f"          pattern: {f['pattern']!r}")
            for flag in f["flags"]:
                print(f"          flag: {flag}")
    if not issues:
        print("no risky patterns found")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
