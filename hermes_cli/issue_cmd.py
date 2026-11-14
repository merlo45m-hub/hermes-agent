"""hermes issue — create a GitHub issue from the CLI.

Subcommands:
  create   Open a new GitHub issue in the given owner/repo.
"""
from __future__ import annotations

import os
import sys

import requests

from hermes_cli.github_issue import GithubIssueError, create_issue as _create


def cmd_issue_create(args) -> None:
    """Handle ``hermes issue create owner repo --title --body --label``."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("error: GITHUB_TOKEN is not set", file=sys.stderr)
        raise SystemExit(1)

    try:
        result = _create(
            owner=args.owner,
            repo=args.repo,
            title=args.title,
            body=args.body,
            labels=args.labels,
            token=token,
        )
    except GithubIssueError as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        raise SystemExit(1)

    print(f"Created issue #{result['number']} in {args.owner}/{args.repo}: {result['html_url']}")
    if args.json_flag:
        import json

        print(json.dumps(result, indent=2))


def cmd_issue(args) -> None:
    """Top-level dispatcher for ``hermes issue [subcommand]``."""
    sub = getattr(args, "issue_command", None)
    if sub is None or sub in ("", None, "create"):
        # no-op parser default for the simple create case
        return cmd_issue_create(args)
    print(f"Unknown issue subcommand: {sub}")
    raise SystemExit(2)
