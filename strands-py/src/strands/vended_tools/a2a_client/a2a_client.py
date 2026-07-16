"""A2A client vended tool.

Thin shim over :class:`strands.agent.a2a_agent.A2AAgent`. The model supplies a
URL and a message; the tool resolves the remote agent's card and sends the
message. URL validation, size caps, and total timeout are enforced at the tool
boundary — the underlying A2AAgent has no SSRF or size guard of its own.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import TYPE_CHECKING

from ...tools.decorator import tool
from ...types.tools import ToolContext
from .types import (
    DEFAULT_A2A_CLIENT_DESCRIPTION,
    DEFAULT_MAX_CARD_BYTES,
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_MESSAGE_BYTES,
    A2AClientOutput,
    A2AClientRemoteCard,
)
from .url_guard import UrlNotAllowedError, validate_url

if TYPE_CHECKING:
    from a2a.client import ClientConfig
    from a2a.types import AgentCard

    from ...tools.decorator import DecoratedFunctionTool

DEFAULT_MULTIAGENT_DEPTH_CAP = 3


def make_a2a_client(
    *,
    name: str = "a2a_client",
    description: str = DEFAULT_A2A_CLIENT_DESCRIPTION,
    client_config: ClientConfig | None = None,
    allowed_url_prefixes: tuple[str, ...] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_card_bytes: int = DEFAULT_MAX_CARD_BYTES,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    multiagent_depth_cap: int = DEFAULT_MULTIAGENT_DEPTH_CAP,
) -> DecoratedFunctionTool:
    """Create an ``a2a_client`` tool.

    All security-relevant configuration is bound at creation time by the
    developer. The model can only supply ``url`` and ``message`` at call
    time. Auth material carried on ``client_config`` is never accessible to
    the model.

    Args:
        name: Tool name. Defaults to ``"a2a_client"``.
        description: Tool description shown to the model.
        client_config: Optional A2A ``ClientConfig`` — carries the httpx
            client used for authenticated card discovery and message
            sending (SigV4, OAuth, bearer tokens, ...). If provided,
            timeouts on the underlying httpx client are the developer's
            responsibility, but the tool's own ``timeout_seconds`` still
            bounds the total invocation. Never touched by the model.
        allowed_url_prefixes: Optional developer-supplied URL allowlist.
            When set, the model-provided ``url`` must start with one of
            these prefixes.
        timeout_seconds: Wall-clock cap on the entire tool call, including
            card discovery + message send. Default: 60s.
        max_card_bytes: Maximum size of the remote agent card in bytes.
            Enforced against the deserialized card. Default: 256 KiB.
        max_response_bytes: Maximum size of the returned response text in
            bytes. Concatenated text parts beyond this cap are truncated.
            Default: 256 KiB.
        multiagent_depth_cap: Cap on the shared multi-agent recursion counter.
            A parent that calls ``a2a_client`` counts as depth+1. The tool
            refuses to run once the counter reaches the cap. Not propagated
            across the wire — remote agents are opaque. Default: 3.

    Returns:
        A decorated tool that invokes a remote A2A agent.
    """

    @tool(name=name, description=description, context="tool_context")
    async def a2a_client_tool(
        url: str,
        message: str,
        tool_context: ToolContext,
    ) -> A2AClientOutput:
        """Invoke a remote A2A agent by URL.

        Args:
            url: HTTP(S) URL of the remote A2A agent (its base URL — the tool
                appends the ``.well-known/agent-card.json`` path). Only public
                hosts are permitted. Private, loopback, link-local, and cloud
                metadata addresses are rejected.
            message: Message to send to the remote agent. Capped at 64 KiB.
            tool_context: Injected by the framework. Not user-facing.

        Raises:
            asyncio.CancelledError: Parent-agent cancellation. Because
                ``CancelledError`` inherits from ``BaseException`` on Python
                3.8+, it bypasses the tool wrapper's ``except Exception``
                block and cancels the whole agent run — which is the
                intended cancellation semantics, matching how the sibling
                network tools behave.
            TimeoutError: The total-invocation deadline elapsed before the
                remote agent responded.
            ValueError: Invalid input or a rejected URL (SSRF policy,
                developer allowlist, size cap, or blocked hostname).
        """
        # A2A imports are optional. Fail with a clear error before we do anything else.
        try:
            from ...agent.a2a_agent import A2AAgent  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("a2a_client requires the 'a2a' extra: pip install 'strands-agents[a2a]'") from exc

        # ---- Multiagent depth cap ----
        # The a2a_client tool participates in the shared depth counter so an
        # adversarial chain (use_agent -> a2a_client -> ...) still respects
        # the cap. The counter can't be propagated across the wire; that's
        # documented in `_multiagent_conventions.md`.
        depth = int(tool_context.invocation_state.get("multiagent_depth", 0) or 0)
        if depth >= multiagent_depth_cap:
            raise ValueError(f"multiagent_depth={depth} exceeds cap {multiagent_depth_cap}")

        # ---- Input validation ----
        if not isinstance(message, str):
            raise ValueError(f"message must be a string, got {type(message).__name__}")
        message_bytes = len(message.encode("utf-8"))
        if message_bytes > MAX_MESSAGE_BYTES:
            raise ValueError(f"message is {message_bytes} bytes; limit is {MAX_MESSAGE_BYTES}")

        try:
            validate_url(url, allowed_prefixes=allowed_url_prefixes)
        except UrlNotAllowedError as exc:
            # Surface as ValueError so the SDK's tool wrapper formats it as a
            # tool-level error rather than crashing the loop.
            raise ValueError(str(exc)) from None

        cancel_signal = getattr(tool_context.agent, "_cancel_signal", None)
        if cancel_signal is not None and cancel_signal.is_set():
            raise asyncio.CancelledError("cancelled before a2a_client started")

        # ---- Run under a total-timeout budget ----
        started_at = time.monotonic()
        try:
            return await asyncio.wait_for(
                _invoke_remote(
                    A2AAgent=A2AAgent,
                    url=url,
                    message=message,
                    client_config=client_config,
                    allowed_prefixes=allowed_url_prefixes,
                    max_card_bytes=max_card_bytes,
                    max_response_bytes=max_response_bytes,
                    cancel_signal=cancel_signal,
                    started_at=started_at,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"a2a_client timed out after {timeout_seconds} seconds calling {url}") from exc

    return a2a_client_tool


a2a_client = make_a2a_client()
"""Default ``a2a_client`` tool. No allowlist, no custom client config."""


async def _invoke_remote(
    *,
    A2AAgent: type,  # noqa: N803 - class alias
    url: str,
    message: str,
    client_config: ClientConfig | None,
    allowed_prefixes: tuple[str, ...] | None,
    max_card_bytes: int,
    max_response_bytes: int,
    cancel_signal: threading.Event | None = None,
    started_at: float,
) -> A2AClientOutput:
    """Resolve the remote agent card, invoke it, and map the result.

    Args:
        A2AAgent: The ``A2AAgent`` class from ``strands.agent.a2a_agent``.
        url: Pre-validated endpoint URL.
        message: Pre-validated message.
        client_config: Optional developer-supplied A2A client config.
        allowed_prefixes: Optional developer allowlist; re-applied to the
            card-advertised URL so the send target obeys the same bound as
            the model-supplied URL.
        max_card_bytes: Card size cap.
        max_response_bytes: Response text cap.
        cancel_signal: Optional parent ``threading.Event`` polled between the
            card fetch and the send so a parent-agent cancellation aborts the
            invocation before the outbound send.
        started_at: ``time.monotonic()`` value captured before the tool
            entered the timeout wrapper. Used to compute the reported
            ``execution_time_ms``.

    Returns:
        The tool output.
    """
    remote = A2AAgent(endpoint=url, client_config=client_config)
    card = await remote.get_agent_card()
    _assert_card_within_size_limit(card, max_card_bytes)

    # The card's own ``url`` may point at a different host than the endpoint we
    # were given (agents are allowed to advertise a different preferred host).
    # Re-run URL validation against the advertised URL so the send goes to a
    # host that passes the same public-only checks. The developer allowlist is
    # re-applied here: the send target is what the message actually reaches, so
    # a pinned allowlist bounds it just as it bounds the model-supplied URL.
    if card.url and card.url != url:
        try:
            validate_url(card.url, allowed_prefixes=allowed_prefixes)
        except UrlNotAllowedError as exc:
            raise ValueError(f"remote agent card points at disallowed url: {exc}") from None

    if cancel_signal is not None and cancel_signal.is_set():
        raise asyncio.CancelledError("cancelled before a2a_client send")

    result = await remote.invoke_async(message)
    text = _extract_text(result.message)
    if len(text.encode("utf-8")) > max_response_bytes:
        text = _truncate_utf8(text, max_response_bytes)

    return A2AClientOutput(
        status="success",
        output=text,
        execution_time_ms=int((time.monotonic() - started_at) * 1000),
        remote_card=A2AClientRemoteCard(
            name=card.name or "",
            description=card.description or "",
            url=card.url or url,
        ),
    )


def _assert_card_within_size_limit(card: AgentCard, limit: int) -> None:
    """Reject cards whose serialized form is larger than ``limit`` bytes.

    Args:
        card: The resolved agent card.
        limit: Byte cap.

    Raises:
        ValueError: If the serialized card exceeds the limit.
    """
    try:
        serialized = card.model_dump_json()
    except Exception:  # pragma: no cover - defensive; fall back to repr
        serialized = json.dumps(repr(card))
    size = len(serialized.encode("utf-8"))
    if size > limit:
        raise ValueError(f"remote agent card is {size} bytes; limit is {limit}")


def _extract_text(message: object) -> str:
    """Concatenate text blocks from a Strands ``Message``-shaped dict.

    Args:
        message: The message payload from the remote agent (as an
            ``AgentResult.message`` dict).

    Returns:
        Concatenated text. Empty string if no text content.
    """
    if not isinstance(message, dict):
        return ""
    content = message.get("content", [])
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Truncate ``text`` to at most ``max_bytes`` UTF-8 bytes without splitting a codepoint.

    Args:
        text: The string to truncate.
        max_bytes: The byte cap.

    Returns:
        The truncated string with a ``... [truncated]`` suffix appended if
        truncation occurred.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    suffix = "... [truncated]"
    suffix_bytes = len(suffix.encode("utf-8"))
    budget = max(0, max_bytes - suffix_bytes)
    truncated = encoded[:budget].decode("utf-8", errors="ignore")
    return truncated + suffix
