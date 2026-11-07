"""
schemas.py — Pydantic schema enforcement for Layer 1.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ValidationError


class VoiceCommandResult(BaseModel):
    """
    Standardized command object representing the output of voice sanitization
    and intent parsing.
    """
    is_valid: bool
    action: str
    intent: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    raw_input: str
    sanitized_text: str
    confidence: float
    flags: List[str] = Field(default_factory=list)
    error: str = ""
    timestamp: int = 0
    latency_ms: float = 0.0


class VoiceRewrite(BaseModel):
    """
    Result returned by the dual-provider voice rewriter.
    """
    rewritten: str
    provider: Literal["nvidia", "openrouter", "none"]
    cache_hit: bool
    latency_ms: float
    confidence: float


class VoiceContext(BaseModel):
    """
    Context engine conversational context state representation.
    """
    profile: str
    task: str = ""
    state: Dict[str, Any] = Field(default_factory=dict)
    history: List[Any] = Field(default_factory=list)


class ArticulatedPrompt(BaseModel):
    """
    Result of the Prompt Cache Engine articulation stage.
    """
    input: str
    articulated: str
    provider: Literal["nvidia", "openrouter", "none"]
    cache_hit: bool
    latency_ms: float


def validate_command(d: Dict[str, Any]) -> VoiceCommandResult:
    """
    Validates a dictionary against VoiceCommandResult schema using model_validate.
    Raises ValidationError on invalid shape.
    """
    if hasattr(VoiceCommandResult, "model_validate"):
        return VoiceCommandResult.model_validate(d)
    return VoiceCommandResult.parse_obj(d)
