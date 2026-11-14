# Tool Argument Coercion — Schema Mismatch Fix

## Overview

Fixed a schema mismatch issue in the tool argument coercion layer where models
emitted a JSON-encoded string instead of a native array, causing tool dispatch
failures.

## Problem

The model sometimes emitted `skill_manage.operations` as a JSON-encoded string
(e.g., `'["action1", "action2"]'` as a string) instead of a native Python list.
The coercion layer's `_coerce_json` function attempted `json.loads()` on this
string, but with a truncated/invalid payload, it raised:

```
Expecting ',' delimiter: line 1 column 5371 (char 5370)
```

This caused `coerce_tool_args` to return the broken string instead of a safe
fallback, which then failed tool schema validation.

## Root Cause

The model emitted a JSON-encoded string (not a native array) for a property
that the schema expects to be an array. The `_coerce_json` function received
the string, tried to parse it as JSON, and when parsing failed it returned
the broken string instead of a safe fallback, propagating the malformed value
downstream.

## Fix

### `acp_adapter/tools.py` — `coerce_tool_args`

- Invalid/malformed JSON payloads now log a warning with a truncated snippet
- Return `{}` (empty dict) instead of `{"raw": raw}` (broken string)
- Prevents downstream schema validation from failing on the raw broken string

### `tools/arg_coercion.py` — `_coerce_json`

- Malformed payloads now log a warning with a truncated snippet
- Return `{}` instead of the original broken string
- Added type-check `isinstance(parsed, expected_python_type)` to ensure the
  parsed value matches the expected type (e.g., `list` for array schemas)
- Valid JSON arrays are preserved; native lists unchanged

## Testing

- AST parse verification: both files parse OK
- `coerce_tool_args('{broken json')` → `{}` + WARNING logged
- Valid JSON array `['action1','action2']` parsed correctly
- Native list unchanged
- Malformed array `'["action1" "action2"]'` → `{}` + warning

## Files Affected

- `acp_adapter/tools.py` — `coerce_tool_args` function
- `tools/arg_coercion.py` — `_coerce_json` function
