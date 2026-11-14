# Tool Argument Coercion - Schema Mismatch Fix

Status: resolved 2026-11-14

## Problem
The model emitted a JSON-encoded string instead of a native array for a tool argument (skill_manage.operations). The coercion layer attempted json.loads() on the string and failed mid-payload:

    Expecting ',' delimiter: line 1 column 5371 (char 5370)

coerce_tool_args then returned the broken string instead of a safe fallback, so the call failed tool schema validation downstream.

## Root Cause
Schema expected array, model emitted a string. _coerce_json received the string, json.loads failed, and the failure path returned the original broken value rather than dropping it.

## Fix
- acp_adapter/tools.py coerce_tool_args: malformed payload now logs a warning (truncated snippet) and returns an empty dict instead of {"raw": raw}.
- tools/arg_coercion.py _coerce_json: malformed payload now logs a warning and returns an empty dict; added isinstance(parsed, expected_python_type) type assertion.
- Valid JSON arrays are still parsed; native lists are untouched.

## Verification
- ast.parse passes on both files.
- coerce_tool_args('{broken json') returns {} and logs WARNING.
- Valid array ['action1','action2'] parses correctly.
- Malformed array '["action1" "action2"]' returns {} with warning.

## Files
- acp_adapter/tools.py
- tools/arg_coercion.py
- docs/tool-coercion-schema-mismatch-fix.md (this note)