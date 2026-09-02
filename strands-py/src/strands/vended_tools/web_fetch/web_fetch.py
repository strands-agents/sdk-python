"""Web fetch tool: fetch a URL and return relevant content about it.

Provides :func:`make_web_fetch` and the default :data:`web_fetch` instance.
The factory's ``mode`` parameter selects the extraction strategy at
construction time:

* ``agentic`` (default): raw page content is passed to an analyst agent that answers
  a ``prompt``, so the full page never enters the main agent's context. Use when
  targeted answers are needed about potentially large pages.
* ``markdown``: HTML is converted to clean markdown with scripts, styles, and
  noise stripped. Use when the agent needs full pages for reasoning.

The tool delegates all networking to the ``httpx.AsyncClient`` instance
provided by the operator, giving full control over transport configuration,
caching, proxies, redirects, and connection pooling.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, Literal

import httpx

from ...tools.decorator import tool
from ...types.tools import ToolContext
from ._extract import html_to_markdown
from .types import WEB_FETCH_DESCRIPTION_AGENTIC, WEB_FETCH_DESCRIPTION_MARKDOWN


class WebFetchError(ValueError):
    """Raised when a web fetch request fails."""


if TYPE_CHECKING:
    from ...models.model import Model
    from ...tools.decorator import DecoratedFunctionTool

_HEADERS = {
    "User-Agent": "strands-agents-web-fetch/1.0",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
}

_DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB

_ANALYST_PROMPT = (
    "You answer a request about a single fetched web page. Use only the provided "
    "content; if it does not contain the answer, say so plainly. Be concise and "
    "factual, and preserve concrete details (names, numbers, quotes, links) "
    "relevant to the request."
)


def make_web_fetch(
    *,
    name: str = "web_fetch",
    description: str | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    client: httpx.AsyncClient | None = None,
    model: Model | None = None,
    mode: Literal["markdown", "agentic"] = "agentic",
) -> DecoratedFunctionTool:
    """Create a web fetch tool.

    Args:
        name: Tool name. Defaults to ``"web_fetch"``.
        description: Tool description shown to the model. Defaults to a mode-appropriate
            description when ``None``.
        max_bytes: Maximum response body size in bytes. Responses larger than
            this are rejected without buffering the entire body. Defaults to
            5 MiB.
        client: Optional ``httpx.AsyncClient`` to use for requests. When
            provided, the tool uses it directly and will not close it.
            When ``None``, a new client is created per request with
            ``follow_redirects=True`` and httpx's default timeout (5s).
        model: Optional model for the analyst. Only used when ``mode='agentic'``.
            Resolution order: this model, then the host agent's model,
            then ``WebFetchError`` if neither is available.
        mode: Extraction mode. Defaults to ``agentic``.

    Returns:
        A decorated tool that fetches a URL and extracts content according to
        the configured mode:
        - ``agentic`` (default): analyst agent answers a ``prompt`` about the
          content; the full page never enters the main agent's context.
        - ``markdown``: HTML converted to clean markdown; other content
          types returned as-is.
    """
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    if mode not in ("markdown", "agentic"):
        raise ValueError(f"mode must be 'markdown' or 'agentic', got {mode!r}")
    resolved_description = description or (
        WEB_FETCH_DESCRIPTION_MARKDOWN if mode == "markdown" else WEB_FETCH_DESCRIPTION_AGENTIC
    )
    external_client = client
    analyst_model = model

    @tool(name=name, description=resolved_description, context=True)
    async def web_fetch_tool_markdown(
        url: str,
        tool_context: ToolContext | None = None,
    ) -> str:
        """Fetches an HTTP(S) URL and returns clean markdown.

        Raises ``WebFetchError`` if the request fails or the client's timeout is exceeded.

        Args:
            url: The URL to fetch. Must be ``http://`` or ``https://``.
            tool_context: Framework-injected. Not model-visible. Carries the
                agent so the tool can read its cancel signal.
        """
        cancel_signal = _extract_cancel_signal(tool_context)
        content_type, raw = await _fetch_once(
            url=url,
            max_bytes=max_bytes,
            client=external_client,
            cancel_signal=cancel_signal,
        )

        is_markup = "html" in content_type.lower() or "xml" in content_type.lower()
        return html_to_markdown(raw) if is_markup else raw

    @tool(name=name, description=resolved_description, context=True)
    async def web_fetch_tool_agentic(
        url: str,
        prompt: str,
        tool_context: ToolContext | None = None,
    ) -> str:
        """Fetches an HTTP(S) URL and returns an analyst's answer about it.

        Raises ``WebFetchError`` if the request fails or the client's timeout is exceeded.

        Args:
            url: The URL to fetch. Must be ``http://`` or ``https://``.
            prompt: The question or instruction about the page content.
            tool_context: Framework-injected. Not model-visible. Carries the
                agent so the tool can read its cancel signal.
        """
        # Local import to avoid circular dependency
        from ...agent.agent import Agent

        if not prompt.strip():
            raise WebFetchError("web_fetch: agentic mode requires a non-empty prompt.")

        host_model = getattr(tool_context.agent, "model", None) if tool_context else None
        effective_model = analyst_model or host_model
        if effective_model is None:
            raise WebFetchError(
                "web_fetch: agentic mode requires a model. "
                "Pass model= to make_web_fetch or call the tool from an agent."
            )

        cancel_signal = _extract_cancel_signal(tool_context)
        content_type, raw = await _fetch_once(
            url=url,
            max_bytes=max_bytes,
            client=external_client,
            cancel_signal=cancel_signal,
        )

        # Fresh agent per call — no history from one fetch bleeds into the next.
        analyst = Agent(
            model=effective_model,
            system_prompt=_ANALYST_PROMPT,
            callback_handler=None,
        )
        invoke_prompt = f"URL: {url}\n\nRequest: {prompt}\n\n--- Content ---\n{raw}"
        return await _stream_agent(analyst, invoke_prompt, cancel_signal, url)

    return web_fetch_tool_markdown if mode == "markdown" else web_fetch_tool_agentic


web_fetch = make_web_fetch()
"""Default web fetch tool (agentic mode)."""


# ---- Internals ----


async def _fetch_once(
    *,
    url: str,
    max_bytes: int,
    client: httpx.AsyncClient | None,
    cancel_signal: threading.Event | None,
) -> tuple[str, str]:
    """Perform one HTTP GET, returning ``(content_type, body_text)``.

    Raises:
        asyncio.CancelledError: When the agent cancel signal is set.
        WebFetchError: On timeout, transport failure, HTTP error status, or
            body exceeding ``max_bytes``.
    """
    _check_cancelled(cancel_signal)

    owns_client = client is None
    active_client = client if client is not None else httpx.AsyncClient(follow_redirects=True)
    try:
        request = active_client.build_request("GET", url, headers=_HEADERS)
        try:
            response = await active_client.send(request, stream=True)
        except httpx.TimeoutException as error:
            raise WebFetchError(f"Fetch timed out: {url!r}") from error
        except (httpx.RequestError, ValueError) as exc:
            raise WebFetchError(f"Fetch failed: {exc}") from exc
        try:
            content_type = response.headers.get("content-type", "")
            if response.status_code >= 400:
                raise WebFetchError(f"HTTP {response.status_code} {response.reason_phrase}")
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                _check_cancelled(cancel_signal)
                total += len(chunk)
                if total > max_bytes:
                    raise WebFetchError(f"Response body exceeded {max_bytes} bytes. Refusing to buffer more.")
                chunks.append(chunk)
            body = b"".join(chunks)
        finally:
            await response.aclose()
    finally:
        if owns_client:
            await active_client.aclose()

    charset = _parse_charset(content_type)
    try:
        raw = body.decode(charset, errors="replace")
    except LookupError:
        raw = body.decode("utf-8", errors="replace")

    return content_type, raw


async def _stream_agent(
    agent: Any,
    prompt: str,
    cancel_signal: threading.Event | None,
    url: str,
) -> str:
    """Stream an agent invocation and return its result as a string.

    ``stream_async`` guarantees an ``AgentResultEvent`` before the stream ends,
    so the return value is always a non-empty string. Cancellation is checked
    between events.

    Raises:
        asyncio.CancelledError: When the cancel signal is set mid-stream.
        WebFetchError: When the agent raises any other exception.
    """
    result = None
    try:
        async for event in agent.stream_async(prompt):
            if "result" in event:
                result = event["result"]
            if cancel_signal is not None and cancel_signal.is_set():
                agent.cancel()
                raise asyncio.CancelledError("Request cancelled")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise WebFetchError(f"Web fetch analyst failed for {url}: {exc}") from exc
    return str(result)


def _parse_charset(content_type: str) -> str:
    """Extract the charset from a Content-Type header, defaulting to ``utf-8``."""
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            value = part[8:].strip().strip("'\"")
            if value:
                return value
    return "utf-8"


def _extract_cancel_signal(tool_context: ToolContext | None) -> threading.Event | None:
    """Return the agent's cancellation event when available."""
    if tool_context is None:
        return None
    agent: Any = getattr(tool_context, "agent", None)
    signal: Any = getattr(agent, "_cancel_signal", None)
    return signal if isinstance(signal, threading.Event) else None


def _check_cancelled(cancel_signal: threading.Event | None) -> None:
    """Raise :class:`asyncio.CancelledError` if the agent's cancel signal has been set."""
    if cancel_signal is not None and cancel_signal.is_set():
        raise asyncio.CancelledError("Request cancelled")
