"""Web fetch tool: fetch a URL and return clean markdown for a model to read.

Distinct from the http_request tool, which returns raw response bodies for API
calls. This tool is intentionally narrow: HTTP(S) GET, decoded body, HTML to
markdown. It is not a general-purpose scraper.
"""

from __future__ import annotations

import asyncio
from http.client import HTTPException
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ...tools.decorator import tool
from ._extract import html_to_markdown
from .types import WEB_FETCH_DESCRIPTION

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool

_HEADERS = {
    "User-Agent": "strands-agents-web-fetch/1.0",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "identity",
    "Connection": "close",
}
_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB


def _fetch_once(url: str, timeout: float, max_bytes: int) -> tuple[str, str]:
    """Perform one HTTP(S) request, returning ``(content_type, body_text)``.

    Raises:
        ValueError: When the URL scheme is not http(s) or the response body exceeds ``max_bytes``.
    """
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"web_fetch only supports http(s) URLs, got {url!r}.")
    with urlopen(Request(url, headers=_HEADERS), timeout=timeout) as response:
        encoding = response.headers.get("Content-Encoding", "").strip().lower()
        if encoding and encoding != "identity":
            raise ValueError(
                f"Server returned Content-Encoding: {encoding!r} despite "
                "requesting uncompressed content. This URL cannot be fetched "
                "with this tool."
            )
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError(f"Response body exceeded max_bytes={max_bytes}. Refusing to buffer more.")
        charset = response.headers.get_content_charset() or "utf-8"
        try:
            raw = body.decode(charset, errors="replace")
        except LookupError:
            raw = body.decode("utf-8", errors="replace")
        content_type = response.headers.get("Content-Type", "")
    return content_type, raw


def make_web_fetch(
    *,
    name: str = "web_fetch",
    description: str = WEB_FETCH_DESCRIPTION,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: int = _DEFAULT_TIMEOUT,
) -> DecoratedFunctionTool:
    """Create a web fetch tool.

    Args:
        name: Tool name. Defaults to ``"web_fetch"``.
        description: Tool description shown to the model.
        max_bytes: Maximum response body size, in bytes. Larger responses are
            rejected without buffering the entire body.
        timeout: Total wall-clock timeout in seconds for each request.

    Returns:
        A decorated tool that fetches a URL and returns the extracted markdown.
    """
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")

    @tool(name=name, description=description)
    async def web_fetch_tool(url: str) -> str:
        """Fetches an HTTP(S) URL and returns readable markdown.

        Only ``http://`` and ``https://`` URLs are accepted. Response body size
        is capped before buffering. Raises ``TimeoutError`` if the request does
        not complete within the configured timeout.

        Args:
            url: The URL to fetch. Must be ``http://`` or ``https://``.
        """
        try:
            content_type, raw = await asyncio.wait_for(
                asyncio.to_thread(_fetch_once, url, float(timeout), max_bytes),
                timeout=float(timeout),
            )
        except asyncio.TimeoutError as error:
            raise TimeoutError(f"web_fetch exceeded total timeout of {timeout}s for {url!r}") from error
        except (HTTPError, URLError, HTTPException, ValueError) as exc:
            raise ValueError(f"Failed to fetch {url}: {exc}") from exc

        is_markup = "html" in content_type.lower() or "xml" in content_type.lower()
        markdown = html_to_markdown(raw) if is_markup else raw
        # Extraction failed; fall back to raw text so the model receives the content.
        if not markdown and is_markup:
            markdown = raw
        return markdown

    return web_fetch_tool


web_fetch = make_web_fetch()
"""Default web fetch tool."""
