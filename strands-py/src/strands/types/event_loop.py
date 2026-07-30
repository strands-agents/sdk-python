"""Event loop-related type definitions for the SDK."""

from typing import Literal

from typing_extensions import Required, TypedDict


class Usage(TypedDict, total=False):
    """Token usage information for model interactions.

    The token counts are **disjoint**: every token the provider billed is counted in exactly
    one of ``inputTokens``, ``outputTokens``, ``cacheReadInputTokens``, or
    ``cacheWriteInputTokens``. Model providers disagree on this — some report the full prompt
    in their input field with cache tokens broken out as a subset of it — so each provider
    normalizes to this contract before reporting.

    The invariant every provider upholds::

        inputTokens + outputTokens + cacheReadInputTokens + cacheWriteInputTokens == totalTokens

    This makes cost a plain weighted sum, which is the point of the contract::

        cost = inputTokens * input_rate
             + cacheReadInputTokens * cache_read_rate    # typically ~0.1x input_rate
             + cacheWriteInputTokens * cache_write_rate  # typically 1.25x-2x input_rate
             + outputTokens * output_rate

    Attributes:
        inputTokens: Number of **net new** prompt tokens sent to the model — tokens that were
            neither read from nor written to the prompt cache. Excludes
            ``cacheReadInputTokens`` and ``cacheWriteInputTokens``; add all three for the full
            prompt size.
        outputTokens: Number of tokens the model generated, including any reasoning tokens.
        totalTokens: Total tokens billed for the request, across all four counters above.
        cacheReadInputTokens: Prompt tokens served from the cache, billed at a discount
            (optional; disjoint from ``inputTokens``).
        cacheWriteInputTokens: Prompt tokens written to the cache, billed at a premium
            (optional; disjoint from ``inputTokens``).
        reasoningOutputTokens: Tokens the model spent on internal reasoning. Unlike the cache
            counters this is a **subset** of ``outputTokens`` (it is billed at the output rate
            and is already counted there), so it must not be added to the total (optional).
    """

    inputTokens: Required[int]
    outputTokens: Required[int]
    totalTokens: Required[int]
    cacheReadInputTokens: int
    cacheWriteInputTokens: int
    reasoningOutputTokens: int


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
    "limit_output_tokens",
    "limit_total_tokens",
    "limit_turns",
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
- "limit_output_tokens": Agent loop stopped because the ``limits["output_tokens"]`` cap was reached
- "limit_total_tokens": Agent loop stopped because the ``limits["total_tokens"]`` cap was reached
- "limit_turns": Agent loop stopped because the ``limits["turns"]`` cap was reached
- "max_tokens": The model provider's per-call output cap was reached
- "stop_sequence": Stop sequence encountered
- "tool_use": Model requested to use a tool
"""
