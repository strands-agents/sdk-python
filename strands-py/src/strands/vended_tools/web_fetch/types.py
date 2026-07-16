"""Shared types and constants for the web fetch tool."""

from typing import TypedDict


class WebFetchOutput(TypedDict):
    """Output of a web fetch.

    Attributes:
        url: The final URL after redirects.
        status: HTTP status code of the final response.
        content_type: Content-Type header of the final response (may be empty).
        title: Extracted document title, or an empty string when none was found.
        markdown: Extracted, cleaned markdown suitable for a model to read.
    """

    url: str
    status: int
    content_type: str
    title: str
    markdown: str


WEB_FETCH_DESCRIPTION = (
    "Fetches an HTTP(S) URL and returns its readable content as markdown, suitable "
    "for a model to read. Scripts, styles, and other non-content noise are stripped. "
    "Only http:// and https:// URLs are allowed. Private, loopback, and link-local "
    "addresses are refused."
)
"""Description for the web fetch tool."""
