# Layer 6 (GenAIOps) — Offline Evals Harness

Automated, reproducible offline evaluation harness for Hermes Enterprise AI Layer 6 (GenAIOps), implementing the first initiative of the **True Enterprise AI FDE Footprint**:
> *"Feed a subset of your recorded Phoenix traces into an automated evaluation tool. Set up automated test assertions (measuring context recall, tool selection accuracy, and schema adherence) that run on code changes or agent prompts."*

---

## 1. OSS Framework Selection & Justification

**Chosen Framework**: **DeepEval** (v4.3.3, Open Source, Apache-2.0 License).

### Why DeepEval over Ragas?
1. **Assertion-First Architecture**:
   - DeepEval is designed ground-up around test assertions (`assert_test`, `BaseMetric`, `LLMTestCase`), making it the natural choice for automated CI/CD and GenAIOps gating.
2. **Zero Paid API Subscriptions & Clean Isolation**:
   - DeepEval runs 100% locally with open-source deterministic, lexical, and structural metrics. No OpenAI, Anthropic, or proprietary LLM API keys are required.
   - Telemetry is explicitly disabled (`DEEPEVAL_TELEMETRY_OPT_OUT=YES`) ensuring zero external network leaks.
3. **Dependency Stability**:
   - During evaluation of Ragas (v0.4.3), tight dependency conflicts were encountered between `langchain-community` (which recently removed `ChatVertexAI` from community modules), `langchain-openai`, and `langchain-core` (`ContextOverflowError` import collisions).
   - In contrast, DeepEval installed cleanly on Python 3.12 (`/root/evals-venv`) with modern Pydantic v2 and `jsonschema` support without dependency friction.

---

## 2. Core Automated Assertions

The harness measures three critical production metrics across recorded Arize Phoenix traces:

| Assertion | Metric Class | Description | Pass Threshold |
|---|---|---|---|
| **Context Recall** | `ContextRecallMetric` | Measures whether the context retrieved and utilized by the agent contains the essential facts, parameters, and entities required for the user's intent. | $\ge 0.75$ (75%) |
| **Tool Selection Accuracy** | `ToolSelectionAccuracyMetric` | Asserts that the MCP tool invoked (e.g. `todo.todo_list`, `intelligent-job-search.search_jobs`, `tor.list_hidden_services`) matches the intended capability. | $1.0$ (100%) |
| **Schema Adherence** | `SchemaAdherenceMetric` | Validates tool inputs and payload arguments against declared MCP JSON schemas using `jsonschema.Draft202012Validator`. | $1.0$ (100%) |

---

## 3. Files Created

- **Evaluation Harness Script**:
  - Local path: `/data/data/com.termux/files/home/Hermes-Agent/evals/layer6_offline_evals.py`
  - VPS path: `/root/evals/layer6_offline_evals.py`
- **MCP Tool Schema Registry**:
  - Local path: `/data/data/com.termux/files/home/Hermes-Agent/evals/mcp_schemas.json`
  - VPS path: `/root/evals/mcp_schemas.json`
- **Documentation**:
  - Local path: `/data/data/com.termux/files/home/Hermes-Agent/evals/README_LAYER6_EVALS.md`
  - VPS path: `/root/evals/README.md`
- **VPS Python Environment**:
  - Dedicated virtualenv: `/root/evals-venv/` (Python 3.12.3, DeepEval 4.3.3, jsonschema 4.26.0)

---

## 4. How to Run

### A. Run On-Demand directly on VPS
```bash
ssh vps "/root/evals-venv/bin/python3 /root/evals/layer6_offline_evals.py --limit 15"
```

### B. Run Remotely from Phone / Workstation (over WireGuard tunnel)
```bash
/data/data/com.termux/files/home/hermes-env/bin/python3 \
  /data/data/com.termux/files/home/Hermes-Agent/evals/layer6_offline_evals.py \
  --endpoint http://10.9.0.1:6006/graphql \
  --limit 15
```

### CLI Arguments
- `--limit N`: Number of recorded Phoenix traces to pull and evaluate (default: `15`).
- `--endpoint URL`: Phoenix GraphQL endpoint (default: `http://127.0.0.1:6006/graphql` or remote `http://10.9.0.1:6006/graphql`).
- `--threshold FLOAT`: Context recall passing threshold (default: `0.75`).
- `--no-probes`: Skip negative assertion validation probes.
- `--output-json PATH`: Export detailed scorecard and per-trace diagnostics to JSON.
- `--verbose`: Enable verbose per-trace reasoning output.

---

## 5. Sample Output

```text
==============================================================================
  LAYER 6 (GenAIOps) — OFFLINE EVALUATION HARNESS
==============================================================================
Phoenix Endpoint : http://127.0.0.1:6006/graphql
Total Recorded   : 16,214 traces (project 'default')
Eval Target      : 15 recorded traces
Evaluation Tool  : DeepEval v4.3.3 (Open Source, Apache-2.0)
Schemas Loaded   : 54 tools from /root/evals/mcp_schemas.json
Thresholds       : Context Recall ≥ 0.75, Tool Acc = 1.0, Schema Adh = 1.0
------------------------------------------------------------------------------
Pulling recorded Phoenix traces...
Successfully retrieved 15 traces with MCP tool executions.

PER-TRACE EVALUATION RESULTS:
------------------------------------------------------------------------------
TRACE ID       | MCP TOOL                       | RECALL   | TOOL ACC | SCHEMA   | STATUS
------------------------------------------------------------------------------
968cac52046b   | tor.list_hidden_services       |   1.00   |   1.00   |   1.00   | PASS
d25b059df244   | jellyfin.get_status            |   1.00   |   1.00   |   1.00   | PASS
c83eb022088a   | supabase.health_check          |   1.00   |   1.00   |   1.00   | PASS
b0635aa6fe34   | todo.todo_list                 |   1.00   |   1.00   |   1.00   | PASS
dfb964a6c88e   | postgres.health_check          |   1.00   |   1.00   |   1.00   | PASS
c8be6ec50ed2   | todo.todo_list                 |   1.00   |   1.00   |   1.00   | PASS
fbdb681e1f4d   | todo.todo_add                  |   1.00   |   1.00   |   1.00   | PASS
74efbe341bb5   | todo.todo_add                  |   1.00   |   1.00   |   1.00   | PASS
83e04903ad7b   | todo.todo_add                  |   1.00   |   1.00   |   1.00   | PASS
38cb63a204d4   | todo.todo_add                  |   1.00   |   1.00   |   1.00   | PASS
ffdd093014ad   | supabase.health_check          |   1.00   |   1.00   |   1.00   | PASS
4e94264213f8   | supabase.health_check          |   1.00   |   1.00   |   1.00   | PASS
77ac12469d1b   | supabase.health_check          |   1.00   |   1.00   |   1.00   | PASS
b043a56b7858   | postgres.sql                   |   1.00   |   1.00   |   1.00   | PASS
1df02e5486bd   | intelligent-job-search.search_ |   1.00   |   1.00   |   1.00   | PASS
probe-schema   | todo.todo_add [Negative Probe: |   1.00   |   1.00   |   0.00   | FAIL
probe-tool-m   | postgres.sql [Negative Probe:  |   0.00   |   0.00   |   1.00   | FAIL
------------------------------------------------------------------------------

EVALUATION SCORECARD (Real Phoenix Traces):
==============================================================================
Total Traces Evaluated  : 15
Traces Passing All Evals: 15 / 15 (100.0%)
Context Recall (Mean)   : 1.000 (Threshold: ≥ 0.75) -> PASS
Tool Selection Accuracy : 100.0% (Threshold: 100%)       -> PASS
Schema Adherence Rate   : 100.0% (Threshold: 100%)       -> PASS
==============================================================================

ASSERTION VALIDATION PROBES (Verifying detection of edge cases):
  • [probe-schema-err] todo.todo_add [Negative Probe: Bad Schema]
    Expected Failure Detected: YES (Verified)
    Assertion feedback (schema): Schema validation error(s) for 'todo.todo_add': 'invalid_lane' is not one of ['now', 'today', 'later']; 'high' is not of type 'integer'
  • [probe-tool-mismatch] postgres.sql [Negative Probe: Wrong Tool]
    Expected Failure Detected: YES (Verified)
    Assertion feedback (tool): Selected 'postgres.sql', but expected 'todo.todo_list'.
    Assertion feedback (recall): Missed 1 facts: ['user active tasks'] (Score: 0.0% < 75.0%).

OVERALL SUITE VERDICT: PASSED
```
