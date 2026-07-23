"""Shared types for the HTTP request tool."""

from typing import TypedDict


class HttpRequestOutput(TypedDict):
    """Output of an HTTP request.

    Attributes:
        status: HTTP status code.
        status_text: HTTP status text.
        headers: Response headers as key-value pairs.
        body: Response body as text.
    """

    status: int
    status_text: str
    headers: dict[str, str]
    body: str
