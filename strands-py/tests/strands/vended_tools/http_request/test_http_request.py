"""Tests for the HTTP request tool."""

import asyncio
import traceback
from unittest.mock import AsyncMock, patch

import httpx
import jsonschema
import pytest
from pydantic import AnyHttpUrl, TypeAdapter

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
        httpx.URL("https://api.example.com/users"),
        headers={"Content-Type": "application/json"},
        content='{"name":"test"}',
    )


@pytest.mark.asyncio
async def test_uses_none_for_unset_headers_and_body():
    request = AsyncMock(return_value=_response())

    with patch("httpx.AsyncClient.request", new=request):
        await http_request(method="GET", url=_URL)

    request.assert_awaited_once_with("GET", httpx.URL(_URL), headers=None, content=None)


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


@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "/relative",
        "ftp://example.com/resource",
        "mailto:user@example.com",
        "file:///tmp/resource",
        "http:///path",
        "http:example.com",
        "https:/example.com",
        "http://example.com\nfoo",
        "http://example.com: ",
        "http://user:pass@example.com/resource",
        r"http://good.example\user:secret@evil.example/resource",
        r"http://example.com\@evil.com",
        r"http://127.0.0.1:8000\@127.0.0.1:8001/private",
    ],
)
@pytest.mark.asyncio
async def test_rejects_non_http_urls_without_request(url):
    with patch("httpx.AsyncClient") as client:
        with pytest.raises(ValueError, match="^Invalid URL"):
            await http_request(method="GET", url=url)

    client.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/resource",
        "https://example.com/resource",
        "http://[::1]/health",
        "http://[0:0:0:0:0:0:0:1]/health",
        "http://[::FFFF:127.0.0.1]/health",
        "http://[2001:0db8:0000:0000:0000:ff00:0042:8329]/health",
        "https://éxample.com/resource",
        "https://xn--xample-9ua.com/resource",
    ],
)
@pytest.mark.asyncio
async def test_accepts_http_and_https_urls(url):
    request = AsyncMock(return_value=_response())

    with patch("httpx.AsyncClient.request", new=request):
        await http_request(method="GET", url=url)

    request.assert_awaited_once()
    called_url = request.await_args.args[1]
    assert isinstance(called_url, httpx.URL)
    assert called_url == httpx.URL(str(TypeAdapter(AnyHttpUrl).validate_python(url)))


@pytest.mark.parametrize(
    ("url", "expected_url"),
    [("HTTP://EXAMPLE.COM/a b", "http://example.com/a%20b")],
)
@pytest.mark.asyncio
async def test_uses_validated_url_for_request_destination(url, expected_url):
    request = AsyncMock(return_value=_response())

    with patch("httpx.AsyncClient.request", new=request):
        await http_request(method="GET", url=url)

    request.assert_awaited_once_with(
        "GET",
        httpx.URL(expected_url),
        headers=None,
        content=None,
    )


@pytest.mark.asyncio
async def test_rejects_credentials_without_disclosing_them():
    url = r"https://secret-user:secret-password\@example.com/resource"

    with patch("httpx.AsyncClient") as client:
        with pytest.raises(ValueError) as caught:
            await http_request(method="GET", url=url)

    assert "secret-user" not in str(caught.value)
    assert "secret-password" not in str(caught.value)
    assert "without embedded credentials" in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is not None
    assert caught.value.__suppress_context__ is True
    formatted = "".join(traceback.format_exception(caught.value))
    assert "secret-user" not in formatted
    assert "secret-password" not in formatted
    client.assert_not_called()


@pytest.mark.asyncio
async def test_tool_error_does_not_disclose_url_credentials():
    url = r"https://secret-user:secret-password\\@example.com/resource"

    with patch("httpx.AsyncClient") as client:
        events = [
            event
            async for event in http_request.stream(
                {"toolUseId": "test", "name": "http_request", "input": {"method": "GET", "url": url}},
                {},
            )
        ]

    result = events[-1].tool_result
    assert result["status"] == "error"
    assert "secret-user" not in str(result)
    assert "secret-password" not in str(result)
    exception = events[-1].exception
    assert isinstance(exception, ValueError)
    assert exception.__cause__ is None
    assert exception.__suppress_context__ is True
    formatted = "".join(traceback.format_exception(exception))
    assert "secret-user" not in formatted
    assert "secret-password" not in formatted
    client.assert_not_called()


@pytest.mark.parametrize("timeout", [0, -1, 10**309, float("inf"), float("-inf"), float("nan")])
@pytest.mark.asyncio
async def test_rejects_non_positive_timeout(timeout):
    with patch("httpx.AsyncClient") as client:
        with pytest.raises(ValueError, match="timeout must be a finite number greater than 0"):
            await http_request(method="GET", url=_URL, timeout=timeout)

    client.assert_not_called()


@pytest.mark.parametrize("timeout", ["1", True, False])
@pytest.mark.asyncio
async def test_tool_validation_rejects_coercible_timeout_values(timeout):
    request = AsyncMock(return_value=_response())

    with patch("httpx.AsyncClient.request", new=request):
        events = [
            event
            async for event in http_request.stream(
                {
                    "toolUseId": "test",
                    "name": "http_request",
                    "input": {"method": "GET", "url": _URL, "timeout": timeout},
                },
                {},
            )
        ]

    assert events[-1].tool_result["status"] == "error"
    request.assert_not_awaited()


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
    assert schema["properties"]["url"]["pattern"] == r"^[hH][tT][tT][pP][sS]?://[^/]"
    for url in ("ftp://example.com", "mailto:user@example.com", "file:///tmp/resource"):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"method": "GET", "url": url}, schema)
    jsonschema.validate({"method": "GET", "url": "https://example.com"}, schema)
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
