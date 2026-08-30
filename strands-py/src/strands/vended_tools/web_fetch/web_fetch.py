"""Web fetch tool: fetch a URL and return relevant content about it.

Distinct from the http_request tool, which returns raw response bodies for API
calls. This tool is intentionally narrow; it performs an HTTP(S) GET, decodes the
body, and extracts the relevant content.

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
from .types import WEB_FETCH_DESCRIPTION


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
    description: str = WEB_FETCH_DESCRIPTION,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    client: httpx.AsyncClient | None = None,
    model: Model | None = None,
) -> DecoratedFunctionTool:
    """Create a web fetch tool.

    Args:
        name: Tool name. Defaults to ``"web_fetch"``.
        description: Tool description shown to the model.
        max_bytes: Maximum response body size in bytes. Responses larger than
            this are rejected without buffering the entire body. Defaults to
            5 MiB.
        client: Optional ``httpx.AsyncClient`` to use for requests. When
            provided, the tool uses it directly and will not close it.
            When ``None``, a new client is created per request with
            ``follow_redirects=True`` and httpx's default timeout (5s).
        model: Optional model for the analyst. Resolution order when
            ``mode='agentic'``: this model, then the host agent's model,
            then ``WebFetchError`` if neither is available.

    Returns:
        A decorated tool that fetches a URL and extracts content according to
        the requested mode:
        - ``markdown``: HTML converted to clean markdown. Conversion is best-effort,
          so content that cannot be converted is returned as-is.
        - ``agentic``: answer to a ``prompt`` about the page content;
          the full page never enters the main agent's context.
    """
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    external_client = client
    analyst_model = model

    @tool(name=name, description=description, context=True)
    async def web_fetch_tool(
        url: str,
        mode: Literal["markdown", "agentic"] = "markdown",
        prompt: str = "",
        tool_context: ToolContext | None = None,
    ) -> str:
        """Fetches an HTTP(S) URL and returns readable content.

        Only ``http://`` and ``https://`` URLs are accepted. Raises
        ``WebFetchError`` if the request fails or the client's timeout is exceeded.

        Args:
            url: The URL to fetch. Must be ``http://`` or ``https://``.
            mode: Extraction mode. ``markdown`` converts HTML to markdown and
                returns it directly. ``agentic`` passes the raw content to an
                analyst agent that answers ``prompt`` — the full page never
                enters the main agent's context.
            prompt: Required when ``mode='agentic'``. The question or
                instruction about the page content.
            tool_context: Framework-injected. Not model-visible. Carries the
                agent so the tool can read its cancel signal.
        """
        cancel_signal = _extract_cancel_signal(tool_context)
        host_model = getattr(tool_context.agent, "model", None) if tool_context else None
        try:
            content_type, raw = await _fetch_once(
                url=url,
                max_bytes=max_bytes,
                client=external_client,
                cancel_signal=cancel_signal,
            )
        except httpx.TimeoutException as error:
            raise WebFetchError(f"Fetch timed out: {url!r}") from error
        except (httpx.RequestError, ValueError) as exc:
            raise WebFetchError(f"Fetch failed: {exc}") from exc

        if mode == "markdown":
            is_markup = "html" in content_type.lower() or "xml" in content_type.lower()
            return html_to_markdown(raw) if is_markup else raw

        elif mode == "agentic":
            from ...agent.agent import Agent  # local import to avoid circular dependency

            if not prompt.strip():
                raise WebFetchError("web_fetch: agentic mode requires a non-empty prompt.")

            effective_model = analyst_model or host_model
            if effective_model is None:
                raise WebFetchError(
                    "web_fetch: agentic mode requires a model. "
                    "Pass model= to make_web_fetch or call the tool from an agent."
                )

            # Fresh agent per call — no history from one fetch bleeds into the next.
            analyst = Agent(
                model=effective_model,
                system_prompt=_ANALYST_PROMPT,
                callback_handler=None,
            )
            invoke_prompt = f"URL: {url}\n\nRequest: {prompt}\n\n--- Content ---\n{raw}"
            return await _stream_agent(analyst, invoke_prompt, cancel_signal, url)

        else:
            raise WebFetchError(f"web_fetch: unknown mode {mode!r}.")

    return web_fetch_tool


web_fetch = make_web_fetch()
"""Default web fetch tool."""


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
        httpx.TimeoutException: When the request times out.
        httpx.RequestError: On any transport-level failure.
        ValueError: When the response body exceeds ``max_bytes`` or the
            status code is >= 400.
    """
    _check_cancelled(cancel_signal)

    owns_client = client is None
    active_client = client if client is not None else httpx.AsyncClient(follow_redirects=True)
    try:
        request = active_client.build_request("GET", url, headers=_HEADERS)
        response = await active_client.send(request, stream=True)
        try:
            content_type = response.headers.get("content-type", "")
            if response.status_code >= 400:
                raise ValueError(f"HTTP {response.status_code} {response.reason_phrase}")
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                _check_cancelled(cancel_signal)
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Response body exceeded {max_bytes} bytes. Refusing to buffer more.")
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
