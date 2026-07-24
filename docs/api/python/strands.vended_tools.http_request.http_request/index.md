HTTP request tool for making raw HTTP calls to external APIs.

Provides :func:`make_http_request` (a factory that lets the tool operator configure allowlists and limits) and :data:`http_request` (a default instance with the safe defaults applied).

The tool is a thin shim over `httpx.AsyncClient`. Its job is to guard the network boundary the model can reach:

-   reject non-http(s) schemes;
-   reject hostnames that end in the SSRF-spec denylist suffixes (`.internal`, `.local`, `.corp`, `.onion`, …);
-   resolve the target host to an IP and refuse non-public destinations (RFC1918, loopback, link-local, multicast, reserved, IPv4-mapped-private IPv6, cloud-metadata addresses) unless the operator has explicitly allowlisted the host at construction time;
-   re-validate every redirect hop against the same policy;
-   strip cross-origin auth (`Authorization` / `Cookie` / `Proxy-Authorization`) on every redirect that changes origin — a scheme downgrade, a port change, or a host change all strip;
-   cap the total request time, the redirect count, the response body size, and the total response-header size;
-   reject model-supplied `Authorization` / `Cookie` / `Proxy-Authorization` headers unless the operator’s config permits them for the target host;
-   propagate the parent agent’s cancel signal (`Agent._cancel_signal`) so an in-flight fetch aborts when the agent is cancelled. Cancellation is signalled with :class:`asyncio.CancelledError`, matching the :func:`asyncio.sleep`\-style vended-tool convention.

DNS-rebinding note. The check-then-connect pattern here leaves a small TOCTOU window: an attacker-controlled name server can return a public address to `getaddrinfo` and a private one to httpx’s own resolve at connect time. Pinning the resolved IP through httpx’s transport is impractical without reaching into httpcore internals; we accept this residual risk against dynamic DNS and rely on the guard for static DNS. The rest of the controls still hold: metadata addresses and non-http(s) schemes remain blocked, and redirects are re-validated.

These controls are set by the tool operator at construction time and are never controllable by the model.

#### Resolver

Function that maps a hostname to a list of IP-address strings.

## HttpRequestError

```python
class HttpRequestError(RuntimeError)
```

Defined in: [src/strands/vended\_tools/http\_request/http\_request.py:128](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/http_request/http_request.py#L128)

Raised when a request is rejected or fails.

## HttpRequestConfig

```python
@dataclass(frozen=True)
class HttpRequestConfig()
```

Defined in: [src/strands/vended\_tools/http\_request/http\_request.py:133](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/http_request/http_request.py#L133)

Operator-controlled configuration for :func:`make_http_request`.

Every field is set by the tool operator at construction time. Nothing here is under the model’s control.

**Attributes**:

-   `allow_private_hosts` - Hostnames (case-insensitive) that are permitted to resolve to a private-network IP. An entry `"internal.example.com"` allows exactly that host; there is no wildcard matching. Empty by
-   `default` - private destinations are denied by default.
-   `allow_auth_for_hosts` - Hostnames for which the model may set `Authorization` / `Cookie` / `Proxy-Authorization` headers. Any request to a host not in this set is rejected outright if it carries one of those headers; on a cross-origin redirect away from an allowlisted host, the same headers are stripped so the credential never travels to the new origin.
-   `max_response_bytes` - Hard cap on response body size in bytes.
-   `max_response_headers_bytes` - Hard cap on total response-header size in bytes. Response headers are returned to the model, so this bounds what an oversized `Set-Cookie` (or similar) can push through.
-   `max_redirects` - Hard cap on redirect chain length. A value of `0` raises `HttpRequestError` if a 3xx is encountered.
-   `default_timeout_seconds` - Timeout used when the model does not supply one (also acts as an upper bound on any timeout the model asks for).

#### make\_http\_request

```python
def make_http_request(
    *,
    allow_private_hosts: Iterable[str] | None = None,
    allow_auth_for_hosts: Iterable[str] | None = None,
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
    max_response_headers_bytes: int = _DEFAULT_MAX_RESPONSE_HEADERS_BYTES,
    max_redirects: int = _DEFAULT_MAX_REDIRECTS,
    default_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    name: str = "http_request",
    description: str = DEFAULT_HTTP_REQUEST_DESCRIPTION,
    resolver: Resolver | None = None,
    transport: httpx.AsyncBaseTransport | None = None
) -> DecoratedFunctionTool
```

Defined in: [src/strands/vended\_tools/http\_request/http\_request.py:187](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/http_request/http_request.py#L187)

Create an HTTP request tool configured for a given security posture.

The safe default posture (no arguments) denies every private-network destination, denies model-supplied `Authorization` / `Cookie`, caps the response body at 10 MiB, follows at most five redirects, and times out after 30 seconds.

**Arguments**:

-   `allow_private_hosts` - Hostnames whose DNS answers may point at private IPs (loopback / RFC1918 / link-local / etc.). Use only for hosts you fully trust the model to reach on the internal network.
-   `allow_auth_for_hosts` - Hostnames for which the model is allowed to set `Authorization` / `Cookie` / `Proxy-Authorization` headers.
-   `max_response_bytes` - Hard cap on response body size.
-   `max_response_headers_bytes` - Hard cap on total response-header size (default 64 KiB).
-   `max_redirects` - Hard cap on redirect chain length. `0` raises `HttpRequestError` if a 3xx is encountered instead of following.
-   `default_timeout_seconds` - Default and upper-bound per-request timeout.
-   `name` - Tool name shown to the model.
-   `description` - Tool description shown to the model.
-   `resolver` - Injection point for the DNS resolver. Callers can pass a custom resolver (e.g. to force a specific mode); tests pass a stub. Defaults to a resolver backed by `socket.getaddrinfo`.
-   `transport` - Injection point for the `httpx` transport. Callers can pass a mock transport in tests. Defaults to the standard connection-based transport.

**Returns**:

A decorated tool that makes HTTP requests within the configured policy.

#### http\_request

Default HTTP request tool with the safe posture applied.