"""Tests for the HTTP request tool."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from strands.vended_tools import http_request

_URL = "https://api.example.com/resource"


def _response(
    status: int = 200,
    *,
    reason: str = "OK",
    headers: list[tuple[str, str]] | None = None,
    content: bytes = b'{"success":true}',
) -> httpx.Response:
    """Build an HTTPX response with a reason phrase for the tool."""
    return httpx.Response(
        status,
        headers=headers,
        content=content,
        extensions={"reason_phrase": reason.encode()},
        request=httpx.Request("GET", _URL),
    )


@pytest.mark.parametrize(
    ("method", "status", "reason"),
    [
        ("GET", 200, "OK"),
        ("POST", 201, "Created"),
        ("PUT", 200, "OK"),
        ("DELETE", 204, "No Content"),
        ("PATCH", 200, "OK"),
        ("HEAD", 200, "OK"),
        ("OPTIONS", 200, "OK"),
    ],
)
@pytest.mark.asyncio
async def test_returns_successful_response_for_supported_methods(method, status, reason):
    response = _response(status, reason=reason, headers=[("content-type", "application/json")])

    with patch("httpx.AsyncClient.request", new=AsyncMock(return_value=response)):
        result = await http_request(method=method, url=_URL)

    assert result == {
        "status": status,
        "status_text": reason,
        "headers": {"content-type": "application/json", "content-length": str(len(response.content))},
        "body": '{"success":true}',
    }


@pytest.mark.asyncio
async def test_sends_custom_headers_and_body():
    request = AsyncMock(return_value=_response())

    with patch("httpx.AsyncClient.request", new=request):
        await http_request(
            method="POST",
            url="https://api.example.com/users",
            headers={"Content-Type": "application/json"},
            body='{"name":"test"}',
        )

    request.assert_awaited_once_with(
        "POST",
        "https://api.example.com/users",
        headers={"Content-Type": "application/json"},
        content='{"name":"test"}',
    )


@pytest.mark.asyncio
async def test_uses_none_for_unset_headers_and_body():
    request = AsyncMock(return_value=_response())

    with patch("httpx.AsyncClient.request", new=request):
        await http_request(method="GET", url=_URL)

    request.assert_awaited_once_with("GET", _URL, headers=None, content=None)


@pytest.mark.parametrize(("content", "expected"), [(b"", ""), (b"Plain text response", "Plain text response")])
@pytest.mark.asyncio
async def test_returns_response_body_as_text(content, expected):
    with patch("httpx.AsyncClient.request", new=AsyncMock(return_value=_response(content=content))):
        result = await http_request(method="GET", url=_URL)

    assert result["body"] == expected


@pytest.mark.asyncio
async def test_flattens_response_headers():
    response = _response(headers=[("content-type", "application/json"), ("x-custom-header", "value")], content=b"{}")

    with patch("httpx.AsyncClient.request", new=AsyncMock(return_value=response)):
        result = await http_request(method="GET", url=_URL)

    assert result["headers"] == {
        "content-type": "application/json",
        "x-custom-header": "value",
        "content-length": "2",
    }


@pytest.mark.parametrize(
    ("status", "reason"),
    [(301, "Moved Permanently"), (404, "Not Found"), (500, "Internal Server Error")],
)
@pytest.mark.asyncio
async def test_raises_for_non_success_statuses(status, reason):
    with patch("httpx.AsyncClient.request", new=AsyncMock(return_value=_response(status, reason=reason))):
        with pytest.raises(RuntimeError, match=rf"^HTTP {status} {reason}: GET {_URL}$"):
            await http_request(method="GET", url=_URL)


@pytest.mark.asyncio
async def test_times_out_after_configured_deadline():
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(1)
        return _response()

    with patch("httpx.AsyncClient.request", new=slow_request):
        with pytest.raises(TimeoutError, match=rf"^Request timed out after 0.01 seconds: GET {_URL}$"):
            await http_request(method="GET", url=_URL, timeout=0.01)


@pytest.mark.asyncio
async def test_wraps_network_failures():
    error = httpx.ConnectError("Failed to connect", request=httpx.Request("GET", _URL))

    with patch("httpx.AsyncClient.request", new=AsyncMock(side_effect=error)):
        with pytest.raises(RuntimeError, match="^Failed to connect$") as caught:
            await http_request(method="GET", url=_URL)

    assert caught.value.__cause__ is error


@pytest.mark.parametrize("url", ["not-a-url", "/relative"])
@pytest.mark.asyncio
async def test_rejects_invalid_http_urls(url):
    with pytest.raises(ValueError, match="^Invalid URL"):
        await http_request(method="GET", url=url)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("-inf"), float("nan")])
@pytest.mark.asyncio
async def test_rejects_non_positive_timeout(timeout):
    with pytest.raises(ValueError, match="timeout must be a finite number greater than 0"):
        await http_request(method="GET", url=_URL, timeout=timeout)


@pytest.mark.parametrize("timeout", ["1", True, False])
@pytest.mark.asyncio
async def test_tool_validation_rejects_coercible_timeout_values(timeout):
    events = [
        event
        async for event in http_request.stream(
            {"toolUseId": "test", "name": "http_request", "input": {"method": "GET", "url": _URL, "timeout": timeout}},
            {},
        )
    ]

    assert events[-1].tool_result["status"] == "error"


@pytest.mark.parametrize("timeout", ["1", True, False])
@pytest.mark.asyncio
async def test_direct_call_rejects_non_numeric_timeout_without_request(timeout):
    request = AsyncMock(return_value=_response())

    with patch("httpx.AsyncClient.request", new=request):
        with pytest.raises(ValueError, match="timeout must be a number"):
            await http_request(method="GET", url=_URL, timeout=timeout)

    request.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejects_unsupported_method_for_direct_calls():
    with pytest.raises(ValueError, match="Unsupported HTTP method"):
        await http_request(method="TRACE", url=_URL)


def test_tool_metadata():
    schema = http_request.tool_spec["inputSchema"]["json"]

    assert http_request.tool_name == "http_request"
    assert schema["required"] == ["method", "url"]
    assert set(schema["properties"]["method"]["enum"]) == {
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
        "HEAD",
        "OPTIONS",
    }
    assert schema["properties"]["url"]["format"] == "uri"
    assert schema["properties"]["timeout"]["exclusiveMinimum"] == 0
    assert schema["properties"]["timeout"]["default"] == 30


@pytest.mark.asyncio
async def test_client_follows_redirects_and_uses_total_deadline():
    response = _response()

    with (
        patch("httpx.AsyncClient.__init__", return_value=None) as init,
        patch(
            "httpx.AsyncClient.__aenter__",
            new=AsyncMock(return_value=AsyncMock(request=AsyncMock(return_value=response))),
        ),
        patch("httpx.AsyncClient.__aexit__", new=AsyncMock(return_value=None)),
    ):
        await http_request(method="GET", url=_URL)

    init.assert_called_once_with(follow_redirects=True, timeout=None)
