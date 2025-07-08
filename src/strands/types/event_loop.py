"""Event loop-related type definitions for the SDK."""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class EventLoopConfig:
    """Configuration for the event loop behavior.

    This class defines the configuration parameters for the event loop's retry and throttling behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts for throttled requests (default: 6)
        initial_delay: Initial delay in seconds before retrying (default: 4)
        max_delay: Maximum delay in seconds between retries (default: 240)
    """

    max_attempts: int = 6
    initial_delay: int = 4
    max_delay: int = 240
