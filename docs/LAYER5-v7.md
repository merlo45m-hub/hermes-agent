# Architecture Summary — Layer 5 vs Layer 7

*Hermes Agent — personal AI agent core.* This note compares the two layers that own the
agent path from a user request to an LLM reply.

## Layer 5 — Orchestration (multi-agent state machine)

**Purpose:** Own *what* work gets done and *which specialist* does it. Decides routing,
consensus, retry, and verification. It never executes task work itself.

**Key files**
- `agent/orchestration_engine.py` — the LangGraph `StateGraph` that wires PROMPT AGENT,
  ORCHESTRATOR, AGY, CODEX, CLAUDE, and GATEWAY nodes; availability/backoff records.
- `agent/agent_init.py` / `agent/agent_runtime_helpers.py` — create per-agent runtimes.
- `agent/orchestration_engine.py` (fleet roles) — role match + dispatch policy.

**Data flow**
```
user input -> PROMPT AGENT -> ORCHESTRATOR -> selected specialist (AGY/CODEX/CLAUDE)
    -> results -> consensus loop -> verified final answer
```

**Failure modes & failover**
- Specialist unavailable after a transient failure -> exponential backoff skip, fall through
  to the next candidate. Permanent failures (missing binary, auth rejected) marked
  unavailable with a long cooldown.
- Handoffs + consensus loops are LangGraph state machine features; retries are bounded and
  logged as `orchestration_engine` state changes.

**Metrics that observe it**
- `agent/orchestration_engine.py` writes `last_run.json` + an `availability.json` ledger that
  the monitoring dashboard's L5 probe reads.

## Layer 7 — LLM inference (LiteLLM gateway path)

**Purpose:** Own *how* the model is reached — provider routing, token budget, rate-limit
handling, and the relay runtime that actually renders a reply.

**Key files**
- `agent/relay_llm.py` / `agent/relay_runtime.py` — model routing + runtime that streams
  replies.
- `agent/anthropic_adapter.py` / `agent/gemini_native_adapter.py` — provider adapters.
- `gateway/config.py` — gateway routing and the proxy layer (`ironsh/iron-proxy`) egress.
- `agent/proxy_sources/iron_proxy.py` — credential-injecting egress firewall for sandbox
  traffic.

**Data flow**
```
model request -> relay_llm (routing) -> provider adapter -> LiteLLM/OpenAI endpoint
    -> stream -> relay_runtime -> user reply
```

**Failure modes & failover**
- Provider 429/quota -> `try_activate_fallback` swaps the runtime; the system-prompt
  `Model:`/`Provider:` lines are rewritten in place + synced into in-flight `api_messages`
  (see `agent/chat_completion_helpers.py` and `tests/agent/test_failover_identity.py`).
- `ironsh/iron-proxy` egress: sandbox tokens are exchanged for real credentials; failures
  warn and never block agent startup.

**Metrics that observe it**
- `agent/monitoring/gateway_health.py` produces the `gateway_health` metrics; `otlp_exporter.py`
  maps `llm_turn` events (tokens, latency, status) to OTel spans.

## Comparison

| Axis | Layer 5 (Orchestration) | Layer 7 (LLM inference) |
|---|---|---|
| Owns | *What* + *who* | *How* the model is reached |
| State | LangGraph multi-agent state machine | Relay runtime + provider adapters |
| Failure handling | skip unavailable specialist, backoff, consensus | provider fallback + proxy credential injection |
| Metrics | availability ledger, last_run | gateway_health, llm_turn spans |
| Failure blast radius | a dispatched specialist | the model/provider path |

## Data flow diagram

```
[User input]
     |
     v
+----------------------------------------------------------+
|                    LAYER 5  Orchestration                |
|  PROMPT -> ORCHESTRATOR -> AGY / CODEX / CLAUDE          |
|  (routing + consensus + retry + verify)                  |
+----------------------------------------------------------+
     |  task result
     v
+----------------------------------------------------------+
|                    LAYER 7  LLM inference                |
|  relay_llm -> adapter -> LiteLLM/OpenAI -> stream        |
|  (failover -> standby model, proxy credential inject)    |
+----------------------------------------------------------+
     |  final answer
     v
[User reply]
```
