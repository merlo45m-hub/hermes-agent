"""GitHub issue creation tool.

Thin wrapper over the GitHub REST API v3 ``/repos/{owner}/{repo}/issues`` endpoint.
Pure function: takes ``owner, repo, title, body, labels, token`` and returns the created
issue dict. Exposes a single, stable surface for the CLI command (``hermes issue create``)
and for any future UI.

Raises :class:`GithubIssueError` on any failure so callers can present a single error
channel. No network access is performed outside :func:`create_issue`.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests

__all__ = ["GithubIssueError", "create_issue"]


class GithubIssueError(Exception):
    """Raised when the GitHub API call fails (auth, network, or validation)."""

    def __init__(self, message: str, *, status: Optional[int] = None, response: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.status = status
        self.response = response


def _token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise GithubIssueError("GITHUB_TOKEN environment variable is not set")
    return token


def create_issue(
    *,
    owner: str,
    repo: str,
    title: str,
    body: Optional[str] = None,
    labels: Optional[list[str]] = None,
    token: Optional[str] = None,
) -> dict[str, Any]:
    """Create an issue and return the API response dict.

    Args:
        owner: Repository owner.
        repo: Repository name.
        title: Issue title.
        body: Issue body (markdown); defaults to ``title`` when omitted.
        labels: Optional label names to attach.
        token: GitHub API token. Falls back to the ``GITHUB_TOKEN`` env var.

    Returns:
        The parsed JSON object returned by GitHub (contains ``number``, ``html_url``,
        ``id``, ``title``, ``state``, etc.).

    Raises:
        GithubIssueError: When the token is missing, the request fails, or GitHub
            rejects the payload (401/403/422).
    """
    token = token or _token()
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    payload: dict[str, Any] = {"title": title, "body": body or title}
    if labels:
        payload["labels"] = list(labels)

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "hermes-agent",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
    except requests.RequestException as exc:
        raise GithubIssueError(f"network error contacting GitHub: {exc}") from exc

    if resp.status_code in (401, 403):
        body = resp.text[:300] if resp.text else ""
        raise GithubIssueError(
            f"GitHub API rejected the request (status {resp.status_code}). "
            f"Check the GITHUB_TOKEN scope (repo read/write) and pain: {body}",  # noqa: E501
            status=resp.status_code,
            response=resp.json() if resp.text else None,
        )
    if resp.status_code == 422:
        raise GithubIssueError(
            f"GitHub rejected the issue payload (status 422 — validation failed). Body: {resp.text[:300]}",
            status=422,
            response=resp.json() if resp.text else None,
        )
    if resp.status_code >= 400:
        raise GithubIssueError(
            f"GitHub API error (status {resp.status_code}): {resp.text[:300]}", status=resp.status_code
        )

    try:
        return resp.json()
    except ValueError:
        raise GithubIssueError("GitHub returned an unexpected, non-JSON response") from None
