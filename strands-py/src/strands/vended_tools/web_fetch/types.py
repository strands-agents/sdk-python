"""Shared types and constants for the web fetch tool."""

WEB_FETCH_DESCRIPTION = (
    "Fetches an HTTP(S) URL and returns its content. "
    "Use mode='markdown' (default) to receive the page as clean markdown; "
    "scripts, styles, and non-content noise are stripped. "
    "if the page cannot be converted, the content is returned raw; "
    "the prompt is ignored. "
    "Use mode='agentic' with a prompt to get a targeted answer about the page; "
    "the full page never enters the conversation."
)
"""Description for the web fetch tool."""
