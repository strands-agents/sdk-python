"""Tests for fetch_doc MCP tool."""

from unittest.mock import patch

import pytest

from strands_mcp_server.server import fetch_doc
from strands_mcp_server.utils.doc_fetcher import Page


@patch("strands_mcp_server.server.cache")
class TestFetchDocTocMode:
    """Tests for fetch_doc TOC mode (no section param)."""

    def test_returns_toc_for_large_doc(self, mock_cache, api_reference_doc):
        mock_cache.ensure_page.return_value = Page(
            url="https://strandsagents.com/test.md",
            title="Test Doc",
            content=api_reference_doc,
        )

        tru_result = fetch_doc(uri="https://strandsagents.com/test.md")

        assert "sections" in tru_result
        assert len(tru_result["sections"]) == 3
        assert tru_result["title"] == "Test Doc"

        # Internal fields must not leak into tool responses
        for section in tru_result["sections"]:
            for key in section:
                assert not key.startswith("_"), f"Internal field '{key}' leaked"

    def test_small_doc_returns_full_content(self, mock_cache, small_doc):
        mock_cache.ensure_page.return_value = Page(
            url="https://strandsagents.com/small.md",
            title="Small Doc",
            content=small_doc,
        )

        tru_result = fetch_doc(uri="https://strandsagents.com/small.md")

        assert tru_result["document_small"] is True
        assert tru_result["reason"] == "size"
        assert "content" in tru_result
        assert "sections" not in tru_result

    def test_small_doc_ignores_section_param(self, mock_cache, small_doc):
        mock_cache.ensure_page.return_value = Page(
            url="https://strandsagents.com/small.md",
            title="Small Doc",
            content=small_doc,
        )

        tru_result = fetch_doc(uri="https://strandsagents.com/small.md", section="1")

        # Section param should be ignored for small docs
        assert tru_result["document_small"] is True
        assert tru_result["reason"] == "size"
        assert "content" in tru_result
        assert "section_id" not in tru_result

    def test_toc_includes_preamble(self, mock_cache, api_reference_doc):
        mock_cache.ensure_page.return_value = Page(
            url="https://strandsagents.com/test.md",
            title="Test Doc",
            content=api_reference_doc,
        )

        tru_result = fetch_doc(uri="https://strandsagents.com/test.md")

        assert "preamble" in tru_result
        assert "Experimental hook events" in tru_result["preamble"]

    def test_no_h2_headers_returns_full_content(self, mock_cache, no_h2_doc):
        mock_cache.ensure_page.return_value = Page(
            url="https://strandsagents.com/no-h2.md",
            title="No H2 Doc",
            content=no_h2_doc,
        )

        tru_result = fetch_doc(uri="https://strandsagents.com/no-h2.md")

        # No ## sections means fallback to full content
        assert tru_result["document_small"] is True
        assert tru_result["reason"] == "no_sections"
        assert "content" in tru_result
        assert "sections" not in tru_result

    @pytest.mark.parametrize("kwargs", [{}, {"uri": ""}], ids=["no-args", "empty-uri"])
    def test_omitted_uri_returns_url_catalog(self, mock_cache, kwargs):
        mock_cache.get_url_titles.return_value = {"https://strandsagents.com/a.md": "Doc A"}

        tru_result = fetch_doc(**kwargs)

        assert tru_result == {"urls": [{"url": "https://strandsagents.com/a.md", "title": "Doc A"}]}


@patch("strands_mcp_server.server.cache")
class TestFetchDocSectionMode:
    """Tests for fetch_doc section mode."""

    def test_returns_section_content(self, mock_cache, api_reference_doc):
        mock_cache.ensure_page.return_value = Page(
            url="https://strandsagents.com/test.md",
            title="Test Doc",
            content=api_reference_doc,
        )

        tru_result = fetch_doc(uri="https://strandsagents.com/test.md", section="1")

        assert tru_result["section_id"] == "1"
        assert "content" in tru_result
        assert "sections" not in tru_result


@patch("strands_mcp_server.server.cache")
class TestFetchDocErrors:
    """Tests for fetch_doc error handling."""

    @pytest.mark.parametrize(
        "malicious_uri",
        [
            "https://strandsagents.com.evil.com/path",
            "https://strandsagents.com@evil.com/path",
            "http://strandsagents.com/path",
            "ftp://strandsagents.com/path",
            "https://evil.com/hack",
        ],
    )
    def test_ssrf_bypass_vectors_rejected(self, mock_cache, malicious_uri):
        tru_result = fetch_doc(uri=malicious_uri)

        assert "error" in tru_result
        assert "strandsagents.com" in tru_result["error"]

    def test_invalid_section_returns_error(self, mock_cache, api_reference_doc):
        mock_cache.ensure_page.return_value = Page(
            url="https://strandsagents.com/test.md",
            title="Test Doc",
            content=api_reference_doc,
        )

        tru_result = fetch_doc(uri="https://strandsagents.com/test.md", section="99")

        assert "error" in tru_result

    def test_fetch_failure_returns_error(self, mock_cache):
        mock_cache.ensure_page.return_value = None

        tru_result = fetch_doc(uri="https://strandsagents.com/missing.md")

        assert tru_result["error"] == "fetch failed"
