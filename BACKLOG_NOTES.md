# Backlog completion notes (per issue)

## Issue #53 - automated verification loops + declarative config mgmt for proxy layers
- Goal: turn proxy-layer configuration into a declarative (YAML) model plus a verification loop
  that re-checks the proxied agents' health/routing after each edit-turn.
- Build plan:
  1. Add `agent/proxy_config.py`: declarative model + YAML loader for `proxy:` block.
  2. Extend `agent/verification_stop.py` to accept a `root` and verify proxy reachability.
  3. Add integration test `tests/agent/test_proxy_config.py`.

## Issue #54 - GitHub Create Issue Feature
- Goal: `hermes issue create` CLI + `tools/github_issue.py` tool + skill `create-issue`.
- Build plan:
  1. `tools/github_issue.py`: GitHub REST API wrapper (create issue) using GITHUB_TOKEN env.
  2. `hermes_cli/subcommands/issue.py`: CLI command.
  3. `skills/github/create-issue/SKILL.md`: user-facing skill.
  4. Tests in `tests/tools/test_github_issue.py` using unittest.mock.

## Issue #56 - summarize gateway and agent benchmark metrics
- Goal: `hermes metrics summarize` prints a compact summary of gateway health + benchmark timeline.
- Build plan:
  1. `hermes_cli/subcommands/metrics.py`: summarize command.
  2. Reuse `agent/monitoring/gateway_health.py` + `benchmark_session_timeline.py`.
  3. Test with a synthetic metrics JSON file.

## Issue #47 - Architecture Summary: Layer 5 vs Layer 7
- Goal: document and compare Layer 5 (orchestration / LangGraph) vs Layer 7 (LiteLLM inference).
- Build plan: write docs/LAYER5-v7.md with comparison, responsibilities, failure modes.

## Issue #55 - LightLLM failover Ansible roles
- Goal: ansible roles to deploy LightLLM with a hot standby failover.
- Build plan:
  1. `ansible/roles/lightllm_primary/` + `lightllm_standby/`.
  2. `site.yml` + vars + handlers + tasks.
  3. Validate YAML with python (no ansible installed).

## Issue #49 - LiteLLM failover block, regex audit, simulated-outage failover test
- Goal: LiteLLM failover config block, audit regex patterns, simulate an outage failover.
- Build plan:
  1. Add `agent/lightllm_failover.py` config block.
  2. Regex audit via a script scanning `re.compile(...)`.
  3. Simulated-outage failover test using mocked http calls.
