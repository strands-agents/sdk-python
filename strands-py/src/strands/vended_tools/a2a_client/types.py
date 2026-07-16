"""Shared types and constants for the a2a_client tool."""

from __future__ import annotations

from typing import Literal, TypedDict

DEFAULT_A2A_CLIENT_DESCRIPTION = (
    "Invokes a remote A2A (Agent-to-Agent) agent by URL. Resolves the agent card at the given "
    "URL and sends a single message. Returns the remote agent's response as text. Only http(s) "
    "URLs to public hosts are permitted; private/loopback/link-local addresses are rejected."
)
"""Description for the a2a_client tool."""

# Cap on the raw agent-card bytes we accept before decoding.
DEFAULT_MAX_CARD_BYTES = 256 * 1024

# Cap on the size of the final response text we return to the model.
DEFAULT_MAX_RESPONSE_BYTES = 256 * 1024

# Cap on model-provided message input.
MAX_MESSAGE_BYTES = 64 * 1024

# Total wall-clock budget for a single invocation.
DEFAULT_TIMEOUT_SECONDS = 60


class A2AClientRemoteCard(TypedDict):
    """Subset of the resolved remote agent card echoed back in the result."""

    name: str
    description: str
    url: str


class A2AClientOutput(TypedDict):
    """Output of an a2a_client invocation.

    Follows the shared multi-agent tool result shape defined in
    ``_multiagent_conventions.md``: a top-level ``status`` / ``output`` /
    ``execution_time_ms`` triple, plus a ``remote_card`` addendum describing
    the resolved remote endpoint.

    ``status`` is only ever ``"success"`` — the a2a_client tool raises
    ``asyncio.CancelledError`` / ``TimeoutError`` on cancellation and timeout
    rather than returning a ``"cancelled"`` variant, matching the other
    network tools (``http_request``, ``web_fetch``). See the a2a_client
    addendum in ``_multiagent_conventions.md`` for the rationale.

    Attributes:
        status: Result status. Always ``"success"`` on a normal return.
        output: Text produced by the remote agent, concatenated from all text parts.
        execution_time_ms: Total wall-clock time for the tool call, in milliseconds.
        remote_card: Subset of the resolved remote agent card.
    """

    status: Literal["success"]
    output: str
    execution_time_ms: int
    remote_card: A2AClientRemoteCard
