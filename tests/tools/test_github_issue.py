"""Tests for :mod:`hermes_cli.github_issue`.

setUp runs with ``GITHUB_TOKEN`` unset and mocks the network at the HTTP layer so no
real calls are made. Covers success + the 422 (validation) + missing-token paths.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import requests

from hermes_cli.github_issue import GithubIssueError, create_issue

TOKEN = "ghp_test_token_12345"


class CreateIssueSuccessTest(unittest.TestCase):
    @patch("hermes_cli.github_issue.requests.post")
    def test_creates_issue_with_title_and_body(self, mock_post):
        mock_resp = unittest.mock.Mock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "number": 7,
            "html_url": "https://github.com/owner/repo/issues/7",
            "id": 12345,
            "title": "Test title",
            "state": "open",
        }
        mock_post.return_value = mock_resp

        result = create_issue(
            owner="owner", repo="repo", title="Test title", body="Test body\nline2", token=TOKEN
        )

        self.assertEqual(result["number"], 7)
        self.assertEqual(result["title"], "Test title")
        mock_post.assert_called_once()
        called_kwargs = mock_post.call_args
        self.assertEqual(called_kwargs[1]["json"]["title"], "Test title")
        self.assertEqual(called_kwargs[1]["json"]["body"], "Test body\nline2")
        self.assertIn("Authorization", called_kwargs[1]["headers"])

    @patch("hermes_cli.github_issue.requests.post")
    def test_creates_issue_with_labels(self, mock_post):
        mock_resp = unittest.mock.Mock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"number": 1}
        mock_post.return_value = mock_resp

        create_issue(owner="o", repo="r", title="t", labels=["bug", "enhancement"], token=TOKEN)
        called_kwargs = mock_post.call_args
        self.assertEqual(called_kwargs[1]["json"]["labels"], ["bug", "enhancement"])

    @patch("hermes_cli.github_issue.requests.post")
    def test_defaults_body_to_title_when_body_omitted(self, mock_post):
        mock_resp = unittest.mock.Mock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"number": 1}
        mock_post.return_value = mock_resp

        create_issue(owner="o", repo="r", title="T", token=TOKEN)
        self.assertEqual(mock_post.call_args[1]["json"]["body"], "T")


class CreateIssueFailureTest(unittest.TestCase):
    def test_missing_token_raises(self):
        with self.assertRaises(GithubIssueError) as ctx:
            create_issue(owner="o", repo="r", title="t")
        self.assertIn("GITHUB_TOKEN", str(ctx.exception))

    @patch("hermes_cli.github_issue.requests.post")
    def test_no_token_global_clear(self, mock_post):
        # The function guards on GITHUB_TOKEN first, so block the mock to force the
        # "token not set" path: remove the env var AND ensure it is not set.
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            mock_post.side_effect = requests.RequestException("boom")
            with self.assertRaises(GithubIssueError) as ctx:
                create_issue(owner="o", repo="r", title="t")
        self.assertIn("GITHUB_TOKEN", str(ctx.exception))

    @patch("hermes_cli.github_issue.requests.post")
    def test_422_validation_error(self, mock_post):
        mock_resp = unittest.mock.Mock()
        mock_resp.status_code = 422
        mock_resp.text = "Validation Failed"
        mock_resp.json.return_value = {"message": "title is required"}
        mock_post.return_value = mock_resp

        with self.assertRaises(GithubIssueError) as ctx:
            create_issue(owner="o", repo="r", title="", token=TOKEN)
        self.assertEqual(ctx.exception.status, 422)

    @patch("hermes_cli.github_issue.requests.post")
    def test_401_unauthorized(self, mock_post):
        mock_resp = unittest.mock.Mock()
        mock_resp.status_code = 401
        mock_resp.text = "Bad credentials"
        mock_post.return_value = mock_resp

        with self.assertRaises(GithubIssueError) as ctx:
            create_issue(owner="o", repo="r", title="t", token=TOKEN)
        self.assertEqual(ctx.exception.status, 401)

    @patch("hermes_cli.github_issue.requests.post")
    def test_non_json_response(self, mock_post):
        mock_resp = unittest.mock.Mock()
        mock_resp.status_code = 201
        mock_resp.json.side_effect = ValueError("not json")
        mock_post.return_value = mock_resp

        with self.assertRaises(GithubIssueError) as ctx:
            create_issue(owner="o", repo="r", title="t", token=TOKEN)
        self.assertIn("unexpected", str(ctx.exception))

    @patch("hermes_cli.github_issue.requests.post")
    def test_4xx_request(self, mock_post):
        mock_resp = unittest.mock.Mock()
        mock_resp.status_code = 500
        mock_resp.text = "server error"
        mock_post.return_value = mock_resp

        with self.assertRaises(GithubIssueError) as ctx:
            create_issue(owner="o", repo="r", title="t", token=TOKEN)
        self.assertEqual(ctx.exception.status, 500)


if __name__ == "__main__":
    unittest.main()
