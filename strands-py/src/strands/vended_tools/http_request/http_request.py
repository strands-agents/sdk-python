"""HTTP request tool for calling external APIs."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Literal

import httpx
from pydantic import AnyUrl, TypeAdapter, ValidationError
from pydantic_core import core_schema

from ...tools.decorator import tool
from ...types.tools import JSONSchema
from .types import HttpRequestOutput

HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]


class Timeout(float):
    """Strict, finite, positive request timeout in seconds."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        """Return the validation schema used by the tool decorator."""
        return core_schema.float_schema(strict=True, allow_inf_nan=False, gt=0)


_DEFAULT_TIMEOUT = 30
_DEFAULT_TIMEOUT_VALUE = Timeout(_DEFAULT_TIMEOUT)
_HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
_HTTP_URL_ADAPTER = TypeAdapter(AnyUrl)
_HTTP_REQUEST_INPUT_SCHEMA: JSONSchema = {
    "json": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": sorted(_HTTP_METHODS),
                "description": "HTTP method to use for the request",
            },
            "url": {
                "type": "string",
                "format": "uri",
                "description": "URL to send the request to",
            },
            "headers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Optional HTTP headers as key-value pairs",
            },
            "body": {
                "type": "string",
                "description": "Optional request body as a string",
            },
            "timeout": {
                "type": "number",
                "exclusiveMinimum": 0,
                "default": _DEFAULT_TIMEOUT,
                "description": "Optional timeout in seconds (default: 30)",
            },
        },
        "required": ["method", "url"],
    }
}


@tool(name="http_request", inputSchema=_HTTP_REQUEST_INPUT_SCHEMA)
async def http_request(
    method: HttpMethod,
    url: str,
    headers: dict[str, str] | None = None,
    body: str | None = None,
    timeout: Timeout = _DEFAULT_TIMEOUT_VALUE,
) -> HttpRequestOutput:
    """Make an HTTP request to an external API.

    Supports GET, POST, PUT, DELETE, PATCH, HEAD, and OPTIONS requests. Redirects
    are followed automatically. The response body is returned as text.

    Args:
        method: HTTP method to use for the request.
        url: URL to send the request to.
        headers: Optional HTTP headers as key-value pairs.
        body: Optional request body as a string.
        timeout: Timeout in seconds (default: 30).

    Returns:
        The response status, status text, headers, and body.

    Raises:
        ValueError: If the method, URL, or timeout is invalid.
        TimeoutError: If the request exceeds the timeout.
        RuntimeError: If the request fails or returns a non-2xx response.
    """
    if method not in _HTTP_METHODS:
        raise ValueError(f"Unsupported HTTP method: {method}")
    timeout_value: object = timeout
    if isinstance(timeout_value, bool) or not isinstance(timeout_value, (int, float)):
        raise ValueError("timeout must be a number")
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise ValueError("timeout must be a finite number greater than 0")

    try:
        _HTTP_URL_ADAPTER.validate_python(url)
    except ValidationError as error:
        raise ValueError(f"Invalid URL: {url}") from error

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
            response = await asyncio.wait_for(
                client.request(method, url, headers=headers, content=body),
                timeout=timeout,
            )
    except (asyncio.TimeoutError, httpx.TimeoutException) as error:
        raise TimeoutError(f"Request timed out after {timeout} seconds: {method} {url}") from error
    except httpx.RequestError as error:
        raise RuntimeError(str(error)) from error

    response_body = response.content.decode("utf-8", errors="replace")
    if not 200 <= response.status_code < 300:
        raise RuntimeError(f"HTTP {response.status_code} {response.reason_phrase}: {method} {url}")

    return {
        "status": response.status_code,
        "status_text": response.reason_phrase,
        "headers": dict(response.headers.items()),
        "body": response_body,
    }
