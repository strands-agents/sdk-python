"""Event loop-related type definitions for the SDK."""

from typing import Final, Literal

from typing_extensions import Required, TypedDict

# Canonical stop-reason literal for the max_iterations cap. Defined as a
# module-level Final so producers (event_loop) and consumers (type literal,
# tests, downstream code) share one source of truth and a typo cannot drift
# a producer out of sync with the StopReason Literal below.
MAX_ITERATIONS_STOP_REASON: Final = "max_iterations"


class Usage(TypedDict, total=False):
    """Token usage information for model interactions.

    Attributes:
        inputTokens: Number of tokens sent in the request to the model.
        outputTokens: Number of tokens that the model generated for the request.
        totalTokens: Total number of tokens (input + output).
        cacheReadInputTokens: Number of tokens read from cache (optional).
        cacheWriteInputTokens: Number of tokens written to cache (optional).
    """

    inputTokens: Required[int]
    outputTokens: Required[int]
    totalTokens: Required[int]
    cacheReadInputTokens: int
    cacheWriteInputTokens: int


class Metrics(TypedDict, total=False):
    """Performance metrics for model interactions.

    Attributes:
        latencyMs (int): Latency of the model request in milliseconds.
        timeToFirstByteMs (int): Latency from sending model request to first
            content chunk (contentBlockDelta or contentBlockStart) from the model in milliseconds.
    """

    latencyMs: Required[int]
    timeToFirstByteMs: int


StopReason = Literal[
    "cancelled",
    "checkpoint",
    "content_filtered",
    "end_turn",
    "guardrail_intervened",
    "interrupt",
    "max_iterations",
    "max_tokens",
    "stop_sequence",
    "tool_use",
]
"""Reason for the model ending its response generation.

- "cancelled": Agent execution was cancelled via agent.cancel()
- "checkpoint": Agent paused for durable checkpoint persistence
- "content_filtered": Content was filtered due to policy violation
- "end_turn": Normal completion of the response
- "guardrail_intervened": Guardrail system intervened
- "interrupt": Agent was interrupted for human input
- "max_iterations": Agent reached its configured max_iterations cap
  (see ``Agent(max_iterations=...)``); the event loop was halted to prevent
  unbounded tool-call recursion.
- "max_tokens": Maximum token limit reached
- "stop_sequence": Stop sequence encountered
- "tool_use": Model requested to use a tool
"""
