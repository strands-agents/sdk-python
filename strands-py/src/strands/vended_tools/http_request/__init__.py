"""HTTP request tool for making raw HTTP calls to external APIs.

The tool ships with a strict default posture: it rejects requests to
non-public destinations (loopback, RFC1918, link-local, multicast, reserved,
cloud-metadata endpoints), refuses non-http(s) schemes, caps redirect chains
and response body size, and rejects model-supplied ``Authorization`` /
``Cookie`` / ``Proxy-Authorization`` headers unless the tool operator has
opted the target host in. On a redirect that changes origin (scheme, host,
or port), those same headers are also stripped so the credential never
travels to the new origin.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import http_request

    agent = Agent(tools=[http_request])
    ```

    Custom posture (e.g. allow a known internal host):
    ```python
    from strands.vended_tools import make_http_request

    tool = make_http_request(
        allow_private_hosts=["metrics.internal.example.com"],
        allow_auth_for_hosts=["api.example.com"],
    )
    agent = Agent(tools=[tool])
    ```
"""

from .http_request import HttpRequestConfig, HttpRequestError, http_request, make_http_request
from .types import HttpMethod, HttpRequestOutput

__all__ = [
    "HttpMethod",
    "HttpRequestConfig",
    "HttpRequestError",
    "HttpRequestOutput",
    "http_request",
    "make_http_request",
]
