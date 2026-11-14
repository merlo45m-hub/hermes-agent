---
name: create-issue
description: >
  Create a GitHub issue from the Hermes CLI. Use when the user asks to open a GitHub
  issue, file a bug, or "create an issue on the repo". Depends on GITHUB_TOKEN.
allowed-tools:
  - sh: hermes issue create owner repo --title --body --label
  - sh: echo $GITHUB_TOKEN
  - sh: gh issue list --limit 5  # sanity check token access
  - sh: gh auth status  # verify auth
---

# Create a GitHub issue

## Usage

  hermes issue create owner repo --title "..." --body "..." [--label bug] [--label enhancement]

## Config

Set `GITHUB_TOKEN` (a GitHub fine-grained/personal access token with `repo` permissions
for private repos, `public_repo` for public). The `hermes issue create` command reads it
directly from the environment.

## Steps

1. Call `echo $GITHUB_TOKEN` and confirm it is present. If it is missing, stop and tell
   the user to export `GITHUB_TOKEN` before continuing.
2. Sanity-check the token: `gh auth status` (or `curl -s -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user`). If the token fails, stop and report that the token is invalid.
3. Run: `hermes issue create owner repo --title "<title>" --body "<body>" [--label X] [--label Y]`
4. Show the returned `number` and `html_url` to the user.

## Rules

- Title must be ≤ 256 chars; body is markdown and feeds the issue body.
- One label per `--label` flag.
- If the title is empty or the token is absent, print a clear error and stop.
