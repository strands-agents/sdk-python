HTTP request tool for making raw HTTP calls to external APIs.

Provides :func:`make_http_request` (a factory that lets the tool operator supply a custom HTTP client) and :data:`http_request` (a default instance).

The tool is a thin shim over `httpx.AsyncClient`. It delegates all networking to the client instance provided by the operator, giving full control over transport configuration, authentication, proxies, timeouts, redirects, and connection pooling.

The parent agent’s cancel signal (`Agent._cancel_signal`) is propagated so an in-flight fetch aborts when the agent is cancelled. Cancellation is signalled with :class:`asyncio.CancelledError`.

## HttpRequestError

```python
class HttpRequestError(RuntimeError)
```

Defined in: [src/strands/vended\_tools/http\_request/http\_request.py:32](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/http_request/http_request.py#L32)

Raised when a request fails.

#### make\_http\_request

```python
def make_http_request(
        *,
        name: str = "http_request",
        description: str = DEFAULT_HTTP_REQUEST_DESCRIPTION,
        client: httpx.AsyncClient | None = None) -> DecoratedFunctionTool
```

Defined in: [src/strands/vended\_tools/http\_request/http\_request.py:36](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/http_request/http_request.py#L36)

Create an HTTP request tool.

**Arguments**:

-   `name` - Tool name shown to the model.
-   `description` - Tool description shown to the model.
-   `client` - Optional `httpx.AsyncClient` instance to use for requests. When provided, the tool uses this client directly — timeouts, redirects, proxies, auth, and all other transport settings are controlled by the client. The tool will not close it. When `None`, a new client is created per request with httpx defaults.

**Returns**:

A decorated tool that makes HTTP requests.

#### http\_request

Default HTTP request tool.