"""Shared types and constants for the web fetch tool."""

from typing import TypedDict


class WebFetchOutput(TypedDict):
    """Output of a web fetch.

    Attributes:
        status: HTTP status code of the final response.
        content_type: Content-Type header of the final response (may be empty).
        markdown: Extracted, cleaned markdown suitable for a model to read.
            The page title is prepended when present.
    """

    status: int
    content_type: str
    markdown: str


WEB_FETCH_DESCRIPTION = (
    "Fetches an HTTP(S) URL and returns its content as markdown. "
    "HTML pages are converted to markdown with the page title prepended; "
    "other content types are returned as-is. "
    "Scripts, styles, and non-content noise are stripped. "
    "Returns status, content_type, and markdown. "
    "Only http:// and https:// URLs are allowed. Private, loopback, and link-local "
    "addresses are refused."
)
"""Description for the web fetch tool."""
