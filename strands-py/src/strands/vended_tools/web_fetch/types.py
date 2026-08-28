"""Shared types and constants for the web fetch tool."""

WEB_FETCH_DESCRIPTION = (
    "Fetches an HTTP(S) URL and returns its content as markdown. "
    "HTML pages are converted to markdown with the page title prepended; "
    "other content types are returned as-is. "
    "Scripts, styles, and non-content noise are stripped."
)
"""Description for the web fetch tool."""
