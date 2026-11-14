"""hermes issue — create GitHub issues from the CLI.

Usage:
  hermes issue create owner repo --title "..." --body "..." [--label bug]

Reads the GitHub token from the GITHUB_TOKEN environment variable.
"""
from __future__ import annotations

import argparse

from hermes_cli.issue_cmd import cmd_issue


def build_issue_parser(subparsers) -> None:
    """Attach the ``issue`` subcommand to ``subparsers``."""
    issue_parser = subparsers.add_parser(
        "issue",
        help="Create a GitHub issue",
        description="Create a GitHub issue for the given owner/repo from the command line.",
    )
    sub = issue_parser.add_subparsers(dest="issue_command")
    sub.add_parser("create", help="Create a new issue")
    issue_parser.set_defaults(func=cmd_issue)
