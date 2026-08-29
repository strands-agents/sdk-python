"""Shared types and constants for the web fetch tool."""

WEB_FETCH_DESCRIPTION = (
    "Fetches an HTTP(S) URL and returns its content as markdown. "
    "HTML pages are converted to markdown with the page title prepended; "
    "other content types are returned as-is. "
    "Scripts, styles, and non-content noise are stripped. "
    "When prompt is provided, an analyst agent answers it over the page "
    "content so only the answer (not the full page) enters the conversation."
)
"""Description for the web fetch tool."""
