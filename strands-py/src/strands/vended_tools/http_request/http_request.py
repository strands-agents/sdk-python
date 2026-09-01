"""HTTP request tool for making raw HTTP calls to external APIs.

Provides :func:`make_http_request` (a factory that lets the tool operator
supply a custom HTTP client) and :data:`http_request` (a default instance).

The tool is a thin shim over ``httpx.AsyncClient``. It delegates all
networking to the client instance provided by the operator, giving full
control over transport configuration, authentication, proxies, timeouts,
redirects, and connection pooling.

When redirects are enabled on the client, the tool follows them manually
so that credential headers can be stripped on cross-host hops. Only
standard transport headers (``accept``, ``user-agent``, etc.) survive a
redirect to a different host; operator-configured credentials and
model-supplied auth tokens are dropped.

The parent agent's cancel signal (``Agent._cancel_signal``) is propagated so
an in-flight fetch aborts when the agent is cancelled. Cancellation is
signalled with :class:`asyncio.CancelledError`.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

import httpx

from ...tools.decorator import tool
from ...types.tools import ToolContext
from .types import DEFAULT_HTTP_REQUEST_DESCRIPTION, HttpMethod, HttpRequestOutput

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool

_SAFE_REDIRECT_HEADERS: frozenset[str] = frozenset({"accept", "accept-encoding", "connection", "host", "user-agent"})
"""Header names safe to forward on a cross-host redirect."""


class HttpRequestError(RuntimeError):
    """Raised when a request fails."""


def make_http_request(
    *,
    name: str = "http_request",
    description: str = DEFAULT_HTTP_REQUEST_DESCRIPTION,
    client: httpx.AsyncClient | None = None,
) -> DecoratedFunctionTool:
    """Create an HTTP request tool.

    Args:
        name: Tool name shown to the model.
        description: Tool description shown to the model.
        client: Optional ``httpx.AsyncClient`` instance to use for requests.
            When provided, the tool uses this client directly — timeouts,
            redirects, proxies, auth, and all other transport settings are
            controlled by the client. The tool will not close it.
            When ``None``, a new client is created per request with httpx
            defaults.

    Returns:
        A decorated tool that makes HTTP requests.
    """
    external_client = client

    @tool(name=name, description=description, context=True)
    async def http_request_tool(
        method: HttpMethod,
        url: str,
        headers: dict[str, str] | None = None,
        body: str | None = None,
        timeout: float | None = None,
        tool_context: ToolContext | None = None,
    ) -> HttpRequestOutput:
        """Make an HTTP request to a URL and return the response.

        Args:
            method: HTTP method (``GET``, ``POST``, ``PUT``, ``DELETE``,
                ``PATCH``, ``HEAD``, ``OPTIONS``).
            url: Absolute URL to request.
            headers: Optional request headers.
            body: Optional request body as a string.
            timeout: Optional per-request timeout in seconds. When a client
                is provided, capped at the client's configured timeout.
                When no client is provided, used as-is.
            tool_context: Framework-injected. Not model-visible. Carries the
                agent so the tool can read its cancel signal mid-flight.
        """
        cancel_signal = _extract_cancel_signal(tool_context)
        effective_timeout = _resolve_timeout(timeout, external_client)

        return await _perform_request(
            method=method,
            url=url,
            headers=headers or {},
            body=body,
            timeout=effective_timeout,
            client=external_client,
            cancel_signal=cancel_signal,
        )

    return http_request_tool


http_request = make_http_request()
"""Default HTTP request tool."""


# ---- Internals ----


def _strip_headers_for_cross_host(headers: httpx.Headers) -> httpx.Headers:
    """Keep only safe transport headers, dropping everything else."""
    return httpx.Headers({k: v for k, v in headers.items() if k.lower() in _SAFE_REDIRECT_HEADERS})


def _resolve_timeout(
    model_timeout: float | None,
    client: httpx.AsyncClient | None,
) -> float | None:
    """Return the effective per-request timeout.

    When a client is provided and has a finite timeout configured, the model
    may request a shorter timeout but can never exceed the client's cap.
    When no client is provided (default path), the model's requested timeout
    is used as-is with no upper bound — operators who need a cap should
    supply a client with an explicit timeout.

    Returns ``None`` when no cap applies (no client provided and model
    didn't supply a timeout, or client has no finite timeout set).
    """
    client_timeout: float | None = None
    if client is not None and isinstance(client.timeout, httpx.Timeout):
        # Use the most relevant phase timeout as the cap. Prefer `read`
        # (governs response wait), fall back to `connect`, then `write`.
        for phase in (client.timeout.read, client.timeout.connect, client.timeout.write):
            if phase is not None:
                client_timeout = phase
                break

    if model_timeout is None:
        return client_timeout
    if model_timeout <= 0:
        raise HttpRequestError("timeout must be positive")
    if client_timeout is None:
        return float(model_timeout)
    return min(float(model_timeout), client_timeout)


async def _perform_request(
    *,
    method: HttpMethod,
    url: str,
    headers: dict[str, str],
    body: str | None,
    timeout: float | None,
    client: httpx.AsyncClient | None,
    cancel_signal: threading.Event | None = None,
) -> HttpRequestOutput:
    """Perform the HTTP request, following redirects with credential safety.

    When the client has ``follow_redirects=True``, redirects are followed
    manually so that credential headers can be stripped on cross-host hops.
    The response body is streamed so the cancel signal can be checked between
    chunks.
    """
    owns_client = client is None

    active_client: httpx.AsyncClient
    if client is not None:
        active_client = client
    else:
        active_client = httpx.AsyncClient()

    should_follow = active_client.follow_redirects
    max_redirects = active_client.max_redirects if should_follow else 0

    try:
        _check_cancelled(cancel_signal)
        try:
            extensions: dict[str, object] = {}
            if timeout is not None:
                extensions["timeout"] = {
                    "connect": timeout,
                    "read": timeout,
                    "write": timeout,
                    "pool": timeout,
                }
            request = active_client.build_request(
                method,
                url,
                headers=headers or None,
                content=body,
                extensions=extensions,
            )

            redirect_count = 0
            while True:
                _check_cancelled(cancel_signal)
                response = await active_client.send(request, stream=True, follow_redirects=False)

                if not should_follow or not response.has_redirect_location:
                    break

                redirect_count += 1
                if redirect_count > max_redirects:
                    await response.aclose()
                    raise httpx.TooManyRedirects(
                        "Exceeded maximum allowed redirects.",
                        request=request,
                    )

                await response.aread()
                next_request = response.next_request
                await response.aclose()

                if next_request is None:
                    break

                if request.url.host != next_request.url.host:
                    next_request.headers = _strip_headers_for_cross_host(next_request.headers)

                request = next_request

        except httpx.TimeoutException as error:
            raise HttpRequestError(f"Request timed out: {error}") from error
        except httpx.TooManyRedirects as error:
            raise HttpRequestError(f"Too many redirects: {error}") from error
        except httpx.RequestError as error:
            raise HttpRequestError(f"Request failed: {error}") from error

        try:
            headers_out = _response_headers(response)

            if response.status_code >= 400:
                await response.aclose()
                raise HttpRequestError(f"HTTP {response.status_code} {response.reason_phrase}: {method} {url}")

            body_text = await _read_body(response, cancel_signal)
        finally:
            if not response.is_closed:
                await response.aclose()

        return HttpRequestOutput(
            status=response.status_code,
            status_text=response.reason_phrase or "",
            headers=headers_out,
            body=body_text,
        )
    finally:
        if owns_client:
            await active_client.aclose()


async def _read_body(
    response: httpx.Response,
    cancel_signal: threading.Event | None,
) -> str:
    """Read the response body, checking the cancel signal between chunks."""
    chunks: list[bytes] = []
    async for chunk in response.aiter_bytes():
        _check_cancelled(cancel_signal)
        chunks.append(chunk)
    raw = b"".join(chunks)
    encoding = response.encoding or "utf-8"
    try:
        return raw.decode(encoding, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _response_headers(response: httpx.Response) -> dict[str, str]:
    r"""Return response headers as a lower-cased dict.

    ``multi_items`` preserves every occurrence, so a response with two
    ``Set-Cookie`` lines does not get comma-collapsed into an unparseable
    single value. Repeated headers are joined with a newline separator.
    """
    result: dict[str, str] = {}
    for key, value in response.headers.multi_items():
        lowered = key.lower()
        existing = result.get(lowered)
        result[lowered] = value if existing is None else f"{existing}\n{value}"
    return result


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
