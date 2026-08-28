"""Tests for the web_fetch tool."""

from __future__ import annotations

import asyncio
import importlib
from unittest.mock import MagicMock

import pytest

# The tool named ``web_fetch`` (re-exported by the package ``__init__``) shadows
# the submodule of the same name in the package namespace. Load the submodule
# explicitly so monkeypatch can rebind attributes on it.
web_fetch_module = importlib.import_module("strands.vended_tools.web_fetch.web_fetch")

from bs4 import BeautifulSoup  # noqa: E402

from strands.vended_tools.web_fetch import _extract as extract_module  # noqa: E402
from strands.vended_tools.web_fetch import make_web_fetch, web_fetch  # noqa: E402
from strands.vended_tools.web_fetch._extract import _tag_attribute, html_to_markdown  # noqa: E402
from strands.vended_tools.web_fetch.types import WEB_FETCH_DESCRIPTION  # noqa: E402


class TestToolMetadata:
    """Tool name, description, and factory validation."""

    def test_default_name(self):
        assert web_fetch.tool_name == "web_fetch"

    def test_default_description(self):
        assert web_fetch.tool_spec["description"] == WEB_FETCH_DESCRIPTION

    def test_custom_name(self):
        assert make_web_fetch(name="fetch_page").tool_name == "fetch_page"

    def test_custom_description(self):
        assert make_web_fetch(description="custom").tool_spec["description"] == "custom"

    @pytest.mark.parametrize("max_bytes", [0, -1])
    def test_rejects_non_positive_max_bytes(self, max_bytes):
        with pytest.raises(ValueError, match="max_bytes"):
            make_web_fetch(max_bytes=max_bytes)

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_rejects_non_positive_timeout(self, timeout):
        with pytest.raises(ValueError, match="timeout"):
            make_web_fetch(timeout=timeout)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/foo",
        "javascript:alert(1)",
        "data:text/html,<script>1</script>",
    ],
)
def test__fetch_once_rejects_non_http_schemes(url):
    with pytest.raises(ValueError, match="only supports http"):
        web_fetch_module._fetch_once(url, timeout=5.0, max_bytes=1024)


def _raising_beautiful_soup(*args, **kwargs):
    raise RuntimeError("boom")


def _fake_urlopen(body: bytes, content_type: str = "text/html", charset: str = "utf-8"):
    """Return a context-manager mock that urlopen can be patched with."""
    inner = MagicMock()
    inner.read.return_value = body
    inner.headers = MagicMock()
    inner.headers.get_content_charset.return_value = charset
    inner.headers.get.side_effect = lambda key, default="": content_type if key == "Content-Type" else default
    inner.status = 200

    outer = MagicMock()
    outer.__enter__ = MagicMock(return_value=inner)
    outer.__exit__ = MagicMock(return_value=False)
    return outer


def test__fetch_once_returns_content_type_and_body(monkeypatch):
    monkeypatch.setattr(web_fetch_module, "urlopen", lambda *a, **kw: _fake_urlopen(b"hello", "text/plain"))
    tru_content_type, tru_raw = web_fetch_module._fetch_once("https://example.com/", timeout=5.0, max_bytes=1024)
    assert (tru_content_type, tru_raw) == ("text/plain", "hello")


def test__fetch_once_rejects_body_over_cap(monkeypatch):
    big = b"x" * 100
    monkeypatch.setattr(web_fetch_module, "urlopen", lambda *a, **kw: _fake_urlopen(big))

    with pytest.raises(ValueError, match="max_bytes"):
        web_fetch_module._fetch_once("https://example.com/", timeout=5.0, max_bytes=50)


class TestWebFetchToolCall:
    """End-to-end tool behavior with the network layer stubbed out."""

    @pytest.mark.asyncio
    async def test_html_response_returns_markdown(self, monkeypatch):
        def stub(url, timeout, max_bytes):
            return "text/html; charset=utf-8", "<html><head><title>T</title></head><body><h1>Hi</h1></body></html>"

        monkeypatch.setattr(web_fetch_module, "_fetch_once", stub)
        tru_result = await web_fetch(url="https://example.com/")
        assert "# T" in tru_result
        assert "# Hi" in tru_result

    @pytest.mark.asyncio
    async def test_non_html_response_returns_body(self, monkeypatch):
        def stub(url, timeout, max_bytes):
            return "text/plain", "plain text response"

        monkeypatch.setattr(web_fetch_module, "_fetch_once", stub)
        tru_result = await web_fetch(url="https://example.com/robots.txt")
        assert tru_result == "plain text response"

    @pytest.mark.asyncio
    async def test_xml_content_type_is_converted_to_markdown(self, monkeypatch):
        def stub(url, timeout, max_bytes):
            return "application/xhtml+xml", "<html><body><p>xhtml</p></body></html>"

        monkeypatch.setattr(web_fetch_module, "_fetch_once", stub)
        tru_result = await web_fetch(url="https://example.com/page.xhtml")
        assert "xhtml" in tru_result

    @pytest.mark.asyncio
    async def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="Failed to fetch"):
            await web_fetch(url="file:///etc/passwd")

    @pytest.mark.asyncio
    async def test_http_exception_is_wrapped_as_value_error(self, monkeypatch):
        # http.client.BadStatusLine (and other HTTPException subclasses) are not
        # subclasses of URLError, so they must be caught explicitly.
        from http.client import BadStatusLine

        def boom(url, timeout, max_bytes):
            raise BadStatusLine("garbage")

        monkeypatch.setattr(web_fetch_module, "_fetch_once", boom)
        with pytest.raises(ValueError, match="Failed to fetch"):
            await web_fetch(url="https://example.com/")

    @pytest.mark.asyncio
    async def test_total_timeout_wraps_slow_transport(self, monkeypatch):
        def slow(url, timeout, max_bytes):
            raise asyncio.TimeoutError()

        monkeypatch.setattr(web_fetch_module, "_fetch_once", slow)
        with pytest.raises(TimeoutError, match="total timeout"):
            await web_fetch(url="https://example.com/")

    @pytest.mark.asyncio
    async def test_html_extraction_failure_falls_back_to_raw(self, monkeypatch):
        # When html_to_markdown returns "" (extraction failed), the tool falls
        # back to raw text so the model receives content rather than nothing.
        def stub(url, timeout, max_bytes):
            return "text/html", "<p>raw content</p>"

        monkeypatch.setattr(web_fetch_module, "_fetch_once", stub)
        monkeypatch.setattr(extract_module, "BeautifulSoup", _raising_beautiful_soup)
        tru_result = await web_fetch(url="https://example.com/")
        assert tru_result == "<p>raw content</p>"


class TestHtmlToMarkdown:
    """Extraction strips noise and preserves structure."""

    def test_strips_script_and_style(self):
        html = """
        <html><head><title>Hi</title>
        <style>body{color:red}</style>
        </head><body>
        <p>Hello world.</p>
        <script>alert('xss')</script>
        <p>After script.</p>
        </body></html>
        """
        md = html_to_markdown(html)
        assert "# Hi" in md
        assert "alert" not in md
        assert "color:red" not in md
        assert "Hello world." in md
        assert "After script." in md

    def test_strips_data_uri_images(self):
        blob = "A" * 200
        html = f'<p>text</p><img src="data:image/png;base64,{blob}" alt="alt text">'
        md = html_to_markdown(html)
        assert blob not in md
        assert "data:" not in md
        assert "alt text" in md

    def test_preserves_regular_images(self):
        html = '<img src="https://example.com/pic.png" alt="pic">'
        md = html_to_markdown(html)
        assert "![pic](https://example.com/pic.png)" in md

    def test_javascript_href_is_dropped(self):
        html = '<a href="javascript:alert(1)">click</a>'
        md = html_to_markdown(html)
        assert "javascript:" not in md
        assert "click" in md

    def test_javascript_img_src_is_dropped(self):
        html = '<img src="javascript:alert(1)" alt="x">'
        md = html_to_markdown(html)
        assert "javascript:" not in md

    @pytest.mark.parametrize("prefix", [" ", "\t", "\u200b", "\ufeff", "\u00ad"])
    def test_javascript_href_with_invisible_prefix_is_dropped(self, prefix):
        # Invisible/whitespace prefixes before "javascript:" must not bypass the check.
        html = f'<a href="{prefix}javascript:alert(1)">click</a>'
        md = html_to_markdown(html)
        assert "javascript:" not in md
        assert "click" in md

    @pytest.mark.parametrize("prefix", [" ", "\u200b", "\ufeff"])
    def test_javascript_img_src_with_invisible_prefix_is_dropped(self, prefix):
        html = f'<img src="{prefix}javascript:alert(1)" alt="x">'
        md = html_to_markdown(html)
        assert "javascript:" not in md

    def test_preserves_headings_lists_and_links(self):
        html = """
        <h1>Title</h1>
        <p>Intro paragraph with a <a href="https://ex.com/x">link</a>.</p>
        <ul><li>one</li><li>two</li></ul>
        <ol><li>first</li><li>second</li></ol>
        """
        md = html_to_markdown(html)
        assert "# Title" in md
        assert "[link](https://ex.com/x)" in md
        assert "- one" in md
        assert "- two" in md
        assert "1. first" in md
        assert "2. second" in md

    def test_preserves_code_blocks(self):
        html = "<pre><code>def f():\n    return 1</code></pre>"
        md = html_to_markdown(html)
        assert "```" in md
        assert "def f():" in md
        assert "return 1" in md

    def test_returns_empty_string_on_parser_exception(self, monkeypatch):
        monkeypatch.setattr(extract_module, "BeautifulSoup", _raising_beautiful_soup)
        assert html_to_markdown("<p>anything</p>") == ""


def test__tag_attribute_joins_list_values():
    # BS4 returns ``class`` as a list; _tag_attribute must join it.
    tag = BeautifulSoup('<div class="foo bar">', "html.parser").div
    assert _tag_attribute(tag, "class") == "foo bar"
