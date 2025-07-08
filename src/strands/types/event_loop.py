"""Event loop-related type definitions for the SDK."""

from typing import Literal

from typing_extensions import TypedDict


class Usage(TypedDict):
    """Token usage information for model interactions.

    Attributes:
        inputTokens: Number of tokens sent in the request to the model..
        outputTokens: Number of tokens that the model generated for the request.
        totalTokens: Total number of tokens (input + output).
    """

    inputTokens: int
    outputTokens: int
    totalTokens: int


class Metrics(TypedDict):
    """Performance metrics for model interactions.

    Attributes:
        latencyMs (int): Latency of the model request in milliseconds.
    """

    latencyMs: int


StopReason = Literal[
    "content_filtered",
    "end_turn",
    "guardrail_intervened",
    "max_tokens",
    "stop_sequence",
    "tool_use",
]
"""Reason for the model ending its response generation.

- "content_filtered": Content was filtered due to policy violation
- "end_turn": Normal completion of the response
- "guardrail_intervened": Guardrail system intervened
- "max_tokens": Maximum token limit reached
- "stop_sequence": Stop sequence encountered
- "tool_use": Model requested to use a tool
"""
