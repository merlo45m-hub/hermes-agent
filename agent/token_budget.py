"""
token_budget.py — Token budget management for Layer 1.
Provides token estimation with chars/4 heuristic, JSON overhead, truncation, and reporting.
"""

import json
from typing import List, Dict, Any, Union

DEFAULT_BUDGET: int = 4000


def estimate_tokens(text: str, is_json: bool = False) -> int:
    """
    Estimates token count using the chars/4 heuristic.
    Applies +12% overhead if the content is identified as JSON or is_json is True.
    """
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)

    # Check for JSON structure if not explicitly flagged
    if not is_json:
        stripped = text.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                json.loads(stripped)
                is_json = True
            except Exception:
                is_json = False

    raw_tokens = len(text) / 4.0
    if is_json:
        raw_tokens *= 1.12

    return max(1, int(round(raw_tokens)))


def truncate_to_budget(
    text: str, max_tokens: int, buffer_ratio: float = 0.1
) -> str:
    """
    Truncates text to fit within max_tokens considering a safety buffer ratio.
    Default buffer_ratio is 0.1 (10% safety margin).
    """
    if not text:
        return ""
    effective_max_tokens = int(max_tokens * (1.0 - buffer_ratio))
    if effective_max_tokens <= 0:
        return ""

    current_tokens = estimate_tokens(text)
    if current_tokens <= effective_max_tokens:
        return text

    # Approximate max characters based on chars/4
    max_chars = effective_max_tokens * 4
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars].rstrip()
    return truncated + "..."


def budget_report(
    messages: List[Union[Dict[str, Any], str]], budget: int = DEFAULT_BUDGET
) -> Dict[str, Any]:
    """
    Generates a token budget utilization report across a sequence of messages.
    """
    per_message = []
    total_tokens = 0

    for idx, msg in enumerate(messages):
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, (dict, list)):
                content_str = json.dumps(content)
                msg_is_json = True
            else:
                content_str = str(content)
                msg_is_json = False
            role = msg.get("role", "unknown")
        else:
            content_str = str(msg)
            msg_is_json = False
            role = "unknown"

        tokens = estimate_tokens(content_str, is_json=msg_is_json)
        per_message.append({
            "index": idx,
            "role": role,
            "tokens": tokens,
        })
        total_tokens += tokens

    pct = round((total_tokens / budget) * 100.0, 2) if budget > 0 else 0.0

    return {
        "messages": per_message,
        "total_tokens": total_tokens,
        "budget": budget,
        "pct_of_budget": pct,
        "remaining_tokens": max(0, budget - total_tokens),
        "is_over_budget": total_tokens > budget,
    }
