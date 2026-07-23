# http_request

Vended tool for making raw HTTP calls from an agent, with a strict-by-default posture. Thin shim over httpx.AsyncClient.

## Usage

```python
from strands import Agent
from strands.vended_tools import http_request

agent = Agent(tools=[http_request])
```

Custom posture (operator opts specific hosts back in):

```python
from strands.vended_tools import make_http_request

tool = make_http_request(
    allow_private_hosts=["metrics.internal.example.com"],
    allow_auth_for_hosts=["api.example.com"],
    max_response_bytes=2 * 1024 * 1024,
    max_redirects=3,
    default_timeout_seconds=15.0,
)
agent = Agent(tools=[tool])
```

Model-facing inputs: `method`, `url`, `headers?`, `body?`, `timeout?`. Model-facing output: `status`, `status_text`, `headers`, `body`.

## Defaults

The tool operator sets every control at construction time; the model cannot loosen them.

- Only http and https schemes are accepted.
- Hostnames ending in `.internal`, `.local`, `.localhost`, `.corp`, `.home`, `.lan`, `.intranet`, `.private`, `.i2p`, or `.onion` are refused before DNS runs. A trailing dot on a rooted FQDN is stripped before matching.
- Loopback, RFC1918, link-local, multicast, reserved, unspecified, CGNAT, and site-local ranges are refused. IPv4-mapped IPv6 addresses are unwrapped first so `is_global` on Python 3.10 and 3.11 cannot smuggle a private address through.
- Cloud metadata endpoints (169.254.169.254, `fd00:ec2::254`, `metadata.google.internal`, `metadata`, 100.100.100.200, 192.0.0.192) are rejected before DNS.
- Redirects are walked manually, every hop is re-validated, and cross-origin hops drop Authorization, Cookie, and Proxy-Authorization. Origin is compared as (scheme, host, port-or-default), so an HTTPS to HTTP downgrade or a port change on the same host also strips. A 303 response forces GET with an empty body. Chain length defaults to five; `max_redirects=0` raises if a 3xx is encountered.
- Response body defaults to a ten-mebibyte cap; response headers default to a sixty-four-kibibyte cap. Oversize responses raise and the connection is aborted.
- Model-supplied `timeout` must be positive and is capped at the operator-configured default (thirty seconds); a zero or negative value is rejected.
- Authorization, Cookie, and Proxy-Authorization are rejected unless the target host is in `allow_auth_for_hosts`. Host and other hop-by-hop headers are rejected unconditionally.
- The parent agent's cancel signal is read via the injected `ToolContext`. A set signal raises `asyncio.CancelledError` before the next redirect hop and between response-body chunks.

## DNS rebinding

The guard resolves the host up front via `getaddrinfo` and re-validates every redirect. Between check-time and connect-time, httpx performs its own resolve; an attacker-controlled name server can return a public address to the first call and a private one to the second. Pinning the resolved IP through httpx is impractical without reaching into httpcore, so the tool accepts this residual risk against dynamic DNS. The guard still protects against static DNS pointing at private space, and metadata endpoints are blocked by name before DNS runs.
