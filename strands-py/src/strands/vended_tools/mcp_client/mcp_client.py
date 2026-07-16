"""Agent-callable MCP client tool.

Exposes a single tool with four operations — ``connect``, ``list_tools``, ``call_tool``,
``disconnect`` — that lets an agent, at runtime, open a connection to a Model Context
Protocol server, discover its tools, invoke one, and tear the connection down. The tool
is a thin shim over :class:`~strands.tools.mcp.MCPClient`; all transport, session, and
protocol handling lives in that class.

**Security model.** This is one of the highest-risk tools to vend: an MCP server can
implement arbitrary logic, so the model must never be allowed to point the tool at an
arbitrary URL. The factory takes a developer-set allowlist of exact URLs at construction;
the tool rejects every ``connect`` that does not match one. In addition:

- Only ``http://`` and ``https://`` URLs are accepted; the tool uses the streamable-http
  MCP transport. Stdio, SSE, and WebSocket are out of scope: stdio expands the attack
  surface substantially (subprocess spawn); SSE is being deprecated by the MCP spec in
  favour of streamable-http; WebSocket is not a documented MCP client transport.
- The URL's host is rejected outright when it matches a DNS suffix denylist
  (``.internal``, ``.local``, ``.localhost``, ``.corp``, ``.home``, ``.lan``,
  ``.intranet``, ``.private``, ``.i2p``, ``.onion``) before any DNS lookup.
- The URL's host is then resolved and every returned address is checked with
  :attr:`ipaddress._BaseAddress.is_global` for the private/loopback/CGNAT/shared-
  address baseline, layered with **explicit** ``is_multicast``, ``is_reserved``,
  ``is_unspecified``, ``is_link_local``, and IPv6-site-local (``fec0::/10``)
  rejections: on all supported CPython versions (3.10–3.14) ``is_global`` is
  ``True`` for IPv4 and IPv6 multicast, and ``fec0::/10`` slips through on some
  releases, so ``is_global`` alone is not sufficient. IPv4-mapped IPv6
  (``::ffff:a.b.c.d``, including the fully expanded form) is unwrapped first so
  the underlying IPv4 category is what gets checked. A short list of
  cloud-metadata addresses is also spelt out explicitly for defence in depth.
- The httpx client that MCP uses under the hood is configured with
  ``follow_redirects=False`` so an allowlisted URL cannot be 3xx'd to a private
  endpoint the guard never saw.
- Session IDs are generated with :mod:`secrets` and scoped to the agent that opened
  them via a :func:`weakref.ref`; another agent using the same tool instance cannot
  use a session it did not open, and a garbage-collected agent's sessions become
  unreachable and are dropped from the session table on the next access or connect.
- The number of live sessions the tool may hold is capped (default: 8), reserved
  synchronously at connect entry so racing connects cannot overshoot the cap.
- Blocking MCP client calls (``start``, ``list_tools_sync``, ``stop``) are pushed
  off the event loop with :func:`asyncio.to_thread` so a slow handshake does not
  stall other tools.
- The tool set exposed by the server is fetched once at connect and cached on
  the session; ``list_tools`` reads that cache and ``call_tool`` validates the
  requested name against it locally. The cache can go stale if the server
  changes its tool set mid-session, which is not a supported MCP flow today.
- Tool-call text and ``structured_content`` are size-capped before being returned to
  the model.

**Limits.** The public-host classification is a check-time judgment. Between the
guard's ``getaddrinfo`` and the MCP transport's own ``getaddrinfo``, an
attacker-controlled resolver could return a public IP first and a private IP second
(DNS rebinding). MCP's transport layer does not expose a hook for pinned-IP connects,
so this shim relies on the allowlist pointing at endpoints whose operator does not
serve time-varying DNS. Do not allowlist a hostname you do not control DNS for.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import secrets
import socket
import weakref
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlparse

import httpx
from mcp.client.streamable_http import streamablehttp_client

from ...tools.decorator import tool
from ...tools.mcp.mcp_client import MCPClient
from ...types.tools import ToolContext

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool

logger = logging.getLogger(__name__)


_MCP_CLIENT_DESCRIPTION = (
    "Connects to Model Context Protocol (MCP) servers at runtime. Supports four operations: "
    "'connect' (open a session to an allowlisted server URL), 'list_tools' (list the tools "
    "the connected server exposes), 'call_tool' (invoke a tool on a connected server), and "
    "'disconnect' (close a session). URLs must be on a developer-set allowlist."
)

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_SCHEME_PORTS = {"http": 80, "https": 443}
_DEFAULT_SESSION_LIMIT = 8
_DEFAULT_STARTUP_TIMEOUT = 30
_DEFAULT_CALL_TIMEOUT_SECONDS = 60
_MAX_RESULT_TEXT_CHARS = 100_000
_MAX_STRUCTURED_CONTENT_BYTES = 100_000

# Hostname suffixes we reject before DNS. Names on these zones are intended for
# internal use even when they happen to resolve to a public IP for some resolver.
_BLOCKED_HOST_SUFFIXES: tuple[str, ...] = (
    ".internal",
    ".local",
    ".localhost",
    ".corp",
    ".home",
    ".lan",
    ".intranet",
    ".private",
    ".i2p",
    ".onion",
)

# Site-local IPv6 (`fec0::/10`, deprecated in RFC 3879 but not caught by
# `is_global` on some CPython releases). Layer an explicit rejection.
_IPV6_SITE_LOCAL = ipaddress.IPv6Network("fec0::/10")

# Cloud-metadata endpoints. `is_global` already rejects link-local (169.254/16) and
# the unique-local IPv6 space, but naming them explicitly guards against a future
# refactor that weakens a generic predicate.
_BLOCKED_METADATA_ADDRESSES: frozenset[str] = frozenset(
    {
        "169.254.169.254",  # AWS / Azure / DigitalOcean
        "fd00:ec2::254",  # AWS IPv6
        "100.100.100.200",  # Alibaba
        "192.0.0.192",  # Oracle Cloud
    }
)

Op = Literal["connect", "list_tools", "call_tool", "disconnect"]


class _Session:
    """A single live MCP connection owned by exactly one agent."""

    __slots__ = ("client", "url", "owner_ref", "tools")

    def __init__(
        self,
        client: MCPClient,
        url: str,
        owner_ref: weakref.ref,
        tools: dict[str, Any],
    ) -> None:
        self.client = client
        self.url = url
        self.owner_ref = owner_ref
        # Tool set cached at connect time. Keyed by tool name; the value is the
        # server-provided metadata used to render `list_tools` and validate the
        # `tool_name` passed to `call_tool`. Matches the TypeScript side, which
        # also caches at connect so both SDKs surface an "unknown tool" error
        # locally instead of forwarding an unadvertised name to the server.
        self.tools = tools


def make_mcp_client(
    *,
    allowed_urls: list[str],
    name: str = "mcp_client",
    description: str = _MCP_CLIENT_DESCRIPTION,
    session_limit: int = _DEFAULT_SESSION_LIMIT,
    startup_timeout: int = _DEFAULT_STARTUP_TIMEOUT,
) -> DecoratedFunctionTool:
    """Create an agent-callable MCP client tool bound to a developer-set URL allowlist.

    The returned tool exposes ``connect``, ``list_tools``, ``call_tool``, and ``disconnect``
    operations under a single tool name. Every ``connect`` is checked against ``allowed_urls``;
    only URLs that appear verbatim (after normalisation) are accepted.

    Args:
        allowed_urls: Exact URLs the tool may connect to. The list is normalised (scheme
            lowercased, trailing slash stripped) and each entry must use ``http`` or
            ``https``. Must not be empty; the tool is deliberately useless without an
            explicit allowlist.
        name: Tool name. Defaults to ``"mcp_client"``.
        description: Tool description shown to the model.
        session_limit: Maximum number of concurrent live sessions across all agents using
            this tool instance. Defaults to 8.
        startup_timeout: Seconds to wait for the initial MCP handshake before treating a
            connection as failed. Defaults to 30.

    Returns:
        A decorated tool that manages MCP sessions.

    Raises:
        ValueError: If ``allowed_urls`` is empty or contains an entry that is not a
            valid, http-family URL.
    """
    if not isinstance(session_limit, int) or isinstance(session_limit, bool) or session_limit <= 0:
        raise ValueError(f"`session_limit` must be a positive integer, got {session_limit!r}")

    normalised_allowlist = _normalise_allowlist(allowed_urls)
    sessions: dict[str, _Session] = {}
    # Reserved-slot counter. Incremented synchronously before any `await` in
    # `_handle_connect`, so racing connects can never overshoot the cap even
    # while the blocking `client.start()` is off the event loop in a worker
    # thread. Held in a single-element list so nested functions can mutate it.
    reserved: list[int] = [0]

    @tool(name=name, description=description, context="tool_context")
    async def mcp_client_tool(
        op: Op,
        tool_context: ToolContext,
        server_url: str | None = None,
        session_id: str | None = None,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        timeout: int = _DEFAULT_CALL_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Manage a runtime MCP client connection.

        Args:
            op: The operation to perform: ``connect``, ``list_tools``, ``call_tool``, ``disconnect``.
            tool_context: Injected by the framework. Not user-facing.
            server_url: Server URL for ``connect``. Must match the developer-set allowlist verbatim.
            session_id: Session identifier returned by ``connect``, required for the other three ops.
            tool_name: Tool name to invoke, required for ``call_tool``.
            arguments: Arguments to pass to the invoked tool, for ``call_tool``.
            timeout: Per-call timeout in seconds. Defaults to 60. Applies to ``call_tool`` only.
        """
        agent = tool_context.agent

        if op == "connect":
            if not server_url:
                raise ValueError("`server_url` is required for op='connect'")
            return await _handle_connect(
                sessions,
                reserved,
                allowlist=normalised_allowlist,
                session_limit=session_limit,
                startup_timeout=startup_timeout,
                server_url=server_url,
                agent=agent,
            )

        if op == "list_tools":
            session = _resolve_session(sessions, session_id, agent)
            return _handle_list_tools(session)

        if op == "call_tool":
            if not tool_name:
                raise ValueError("`tool_name` is required for op='call_tool'")
            session = _resolve_session(sessions, session_id, agent)
            return await _handle_call_tool(session, tool_name, arguments, timeout)

        if op == "disconnect":
            session = _resolve_session(sessions, session_id, agent)
            return await _handle_disconnect(sessions, cast(str, session_id), session)

        raise ValueError(f"Unknown op: {op}")

    return mcp_client_tool


# ---- URL validation ----------------------------------------------------------------


def _normalise_allowlist(urls: list[str]) -> frozenset[str]:
    """Normalise the developer-set allowlist and validate each entry."""
    if not urls:
        raise ValueError("`allowed_urls` must not be empty; the mcp_client tool requires an explicit allowlist")

    normalised: set[str] = set()
    for raw in urls:
        parsed = urlparse(raw)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            raise ValueError(
                f"Allowlist entry {raw!r} has unsupported scheme {parsed.scheme!r}; only http and https are supported"
            )
        if not parsed.hostname:
            raise ValueError(f"Allowlist entry {raw!r} has no host")
        _reject_userinfo_and_fragment(raw, parsed)
        normalised.add(_canonicalise_url(raw))
    return frozenset(normalised)


def _reject_userinfo_and_fragment(raw: str, parsed: Any) -> None:
    """Reject URLs carrying credentials or fragments.

    Both are stripped by :func:`_canonicalise_url`, so leaving them accepted would
    let ``https://user:pass@host/path`` canonicalise to the same string as
    ``https://host/path`` and bypass the verbatim allowlist match.
    """
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"URL {raw!r} carries credentials; strip the userinfo before allowlisting or connecting")
    if parsed.fragment:
        raise ValueError(f"URL {raw!r} carries a fragment; strip it before allowlisting or connecting")


def _canonicalise_url(url: str) -> str:
    """Canonicalise a URL for allowlist matching.

    Lowercase scheme/host, strip the trailing slash on the path, drop the port when it
    matches the scheme default (so ``https://host`` and ``https://host:443`` match), and
    drop the fragment. Userinfo is dropped too; :func:`_reject_userinfo_and_fragment`
    refuses URLs carrying credentials or fragments before this function ever sees them.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    default_port = _DEFAULT_SCHEME_PORTS.get(scheme)
    port = f":{parsed.port}" if parsed.port and parsed.port != default_port else ""
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{host}{port}{path}{query}"


def _assert_public_host(url: str) -> None:
    """Reject URLs whose host is on the suffix denylist or resolves to non-public space.

    Applied on top of the allowlist as defence in depth: even if the allowlist entry
    itself points at an internal address, the guard still refuses.

    The classification is a check-time judgment. A DNS-rebinding resolver can return a
    different address to the MCP transport's own ``getaddrinfo`` at connect time; the
    MCP client SDK does not expose a hook for pinned-IP connects, so this guard alone
    does not close the TOCTOU window. Do not allowlist a hostname whose DNS you do not
    control.
    """
    host = (urlparse(url).hostname or "").lower()
    if host.endswith("."):
        host = host[:-1]

    for suffix in _BLOCKED_HOST_SUFFIXES:
        bare = suffix.lstrip(".")
        if host == bare or host.endswith(suffix):
            raise ValueError(f"Refusing to connect to {url!r}: hostname {host!r} matches blocked suffix {suffix!r}")

    try:
        ip = ipaddress.ip_address(host)
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as e:
            raise ValueError(f"Could not resolve host {host!r}: {e}") from e
        addresses = []
        for info in infos:
            sockaddr = info[4]
            try:
                addresses.append(ipaddress.ip_address(sockaddr[0]))
            except ValueError:
                continue

    if not addresses:
        raise ValueError(f"Could not resolve host {host!r} to any IP address")

    for addr in addresses:
        # Unwrap IPv4-mapped IPv6 first so subsequent checks apply to the embedded v4.
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        if str(addr) in _BLOCKED_METADATA_ADDRESSES:
            raise ValueError(f"Refusing to connect to {url!r}: host {host!r} resolves to metadata address {addr}")
        # `is_global` is True for multicast on every supported CPython (3.10–3.14); the
        # site-local `fec0::/10` range also slips through on some releases. Layer
        # explicit rejections on top so the guard doesn't inherit those quirks.
        if addr.is_multicast or addr.is_reserved or addr.is_unspecified or addr.is_link_local:
            raise ValueError(f"Refusing to connect to {url!r}: host {host!r} resolves to non-public address {addr}")
        if isinstance(addr, ipaddress.IPv6Address) and addr in _IPV6_SITE_LOCAL:
            raise ValueError(f"Refusing to connect to {url!r}: host {host!r} resolves to non-public address {addr}")
        if not addr.is_global:
            raise ValueError(f"Refusing to connect to {url!r}: host {host!r} resolves to non-public address {addr}")


# ---- Operation handlers ------------------------------------------------------------


async def _handle_connect(
    sessions: dict[str, _Session],
    reserved: list[int],
    *,
    allowlist: frozenset[str],
    session_limit: int,
    startup_timeout: int,
    server_url: str,
    agent: Any,
) -> dict[str, Any]:
    parsed = urlparse(server_url)
    _reject_userinfo_and_fragment(server_url, parsed)

    canonical = _canonicalise_url(server_url)
    if canonical not in allowlist:
        raise ValueError(f"URL {server_url!r} is not on the developer-set allowlist")

    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        # Defence in depth: the allowlist already screens this.
        raise ValueError(f"unsupported scheme {scheme!r}")

    _assert_public_host(server_url)

    # Drop sessions whose owning agent has been garbage-collected before enforcing
    # the cap. Left in place they'd count toward the limit forever and pin the
    # underlying transport open.
    _purge_dead_sessions(sessions)

    if len(sessions) + reserved[0] >= session_limit:
        raise RuntimeError(
            f"Refusing to open a new MCP session: "
            f"{len(sessions) + reserved[0]}/{session_limit} concurrent sessions in use"
        )

    transport_callable = _make_http_transport(server_url)

    # Reserve a slot synchronously before any `await`. Without this a burst of
    # concurrent connects would all pass the cap check, then race to `start()`
    # in worker threads and overshoot.
    reserved[0] += 1
    client = MCPClient(transport_callable, startup_timeout=startup_timeout)
    try:
        # `MCPClient.start()` blocks the calling thread until the MCP handshake
        # completes or `startup_timeout` elapses. Push it off the event loop so
        # concurrent invocations of the tool are not serialised.
        await asyncio.to_thread(client.start)
        try:
            # Cache the tool list at connect time. `call_tool` validates the
            # requested name against this cache so an unadvertised name fails
            # locally instead of being forwarded to the server. TypeScript
            # does the same; both SDKs accept that the cache goes stale if
            # the server changes its tool set mid-session, which is not a
            # supported MCP flow today.
            tools = await asyncio.to_thread(_list_tools_via_client, client)
        except BaseException:
            # A failure past `start()` leaves the client connected. Tear it down
            # so we do not leak the transport.
            try:
                await asyncio.to_thread(client.stop, None, None, None)
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("failed to tear down MCP client after connect failure", exc_info=True)
            raise
    finally:
        reserved[0] -= 1

    session_id = secrets.token_urlsafe(24)
    sessions[session_id] = _Session(
        client=client,
        url=canonical,
        owner_ref=weakref.ref(agent),
        tools=tools,
    )
    logger.debug("session_id=<%s>, url=<%s> | opened MCP session", session_id, canonical)

    return {"session_id": session_id, "server_url": canonical}


def _list_tools_via_client(client: MCPClient) -> dict[str, dict[str, Any]]:
    """Pull the full tool list from an MCP client and return it keyed by name."""
    tools: dict[str, dict[str, Any]] = {}
    pagination_token: str | None = None
    while True:
        paginated = client.list_tools_sync(pagination_token)
        for agent_tool in paginated:
            mcp_tool = agent_tool.mcp_tool
            entry: dict[str, Any] = {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "",
                "input_schema": mcp_tool.inputSchema,
            }
            if mcp_tool.outputSchema:
                entry["output_schema"] = mcp_tool.outputSchema
            tools[mcp_tool.name] = entry
        pagination_token = paginated.pagination_token
        if pagination_token is None:
            break
    return tools


def _handle_list_tools(session: _Session) -> dict[str, Any]:
    return {"tools": list(session.tools.values())}


async def _handle_call_tool(
    session: _Session,
    tool_name: str,
    arguments: dict[str, Any] | None,
    timeout: int,
) -> dict[str, Any]:
    if tool_name not in session.tools:
        raise ValueError(f"Tool {tool_name!r} is not exposed by the connected server")

    tool_use_id = secrets.token_urlsafe(12)
    result = await session.client.call_tool_async(
        tool_use_id=tool_use_id,
        name=tool_name,
        arguments=arguments,
        read_timeout_seconds=timedelta(seconds=timeout),
    )

    truncated = False
    text_parts: list[str] = []
    for content in result.get("content", []):
        text = content.get("text") if isinstance(content, dict) else None
        if isinstance(text, str):
            text_parts.append(text)

    joined = "\n".join(text_parts)
    if len(joined) > _MAX_RESULT_TEXT_CHARS:
        joined = joined[:_MAX_RESULT_TEXT_CHARS]
        truncated = True

    response: dict[str, Any] = {
        "status": result.get("status", "error"),
        "text": joined,
    }
    if truncated:
        response["truncated"] = True
    if "structuredContent" in result:
        response["structured_content"] = _cap_structured_content(result["structuredContent"], response)
    if result.get("isError"):
        response["is_error"] = True
    return response


def _cap_structured_content(value: Any, response: dict[str, Any]) -> Any:
    """Serialise ``value`` to JSON and, if it exceeds the size cap, return a marker.

    Left as-is when small; when too large, replaced with an object announcing the
    truncation so the model can see the content was dropped rather than silently
    receiving a shortened blob.
    """
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):
        # Non-JSON-serialisable — reject rather than smuggle an opaque object through.
        response["truncated"] = True
        return {"__truncated__": True, "reason": "structured_content was not JSON-serialisable"}
    encoded_bytes = len(encoded.encode("utf-8"))
    if encoded_bytes <= _MAX_STRUCTURED_CONTENT_BYTES:
        return value
    response["truncated"] = True
    # `size` reports the same unit the cap is measured in (bytes) so a non-ASCII
    # payload's reported size matches the check.
    return {"__truncated__": True, "reason": "structured_content exceeded size cap", "size": encoded_bytes}


async def _handle_disconnect(
    sessions: dict[str, _Session],
    session_id: str,
    session: _Session,
) -> dict[str, Any]:
    try:
        # `stop` blocks until the transport tears down. Push it off the event loop.
        await asyncio.to_thread(session.client.stop, None, None, None)
    finally:
        sessions.pop(session_id, None)
    return {"disconnected": True}


def _purge_dead_sessions(sessions: dict[str, _Session]) -> None:
    """Remove sessions whose owning agent has been garbage-collected.

    A dead weakref means the agent that opened the session no longer exists, so
    the session is permanently unreachable via :func:`_resolve_session`. Left in
    place these entries would count against the ``session_limit`` forever and
    pin the underlying MCP transport open. Best-effort ``stop`` for the same
    reason — failures are logged, not raised.
    """
    dead: list[str] = [sid for sid, s in sessions.items() if s.owner_ref() is None]
    for sid in dead:
        session = sessions.pop(sid, None)
        if session is None:
            continue
        try:
            session.client.stop(None, None, None)
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("failed to stop MCP client for GC'd session %s", sid, exc_info=True)


def _resolve_session(sessions: dict[str, _Session], session_id: str | None, agent: Any) -> _Session:
    """Look up a session, rejecting missing ids and cross-agent access.

    The owner is held as a weakref: if the agent that opened the session has been
    garbage-collected, ``owner_ref()`` returns ``None`` and the session is
    unreachable — the caller gets the same "no active session" error as an
    unknown id. When we observe a dead owner we also drop the session and
    best-effort stop its client so the resource doesn't leak.
    """
    if not session_id:
        raise ValueError("`session_id` is required")
    session = sessions.get(session_id)
    if session is None:
        raise ValueError(f"No active session for id {session_id!r}")
    owner = session.owner_ref()
    if owner is None:
        # Dead weakref: reap this specific session as we walk past it. Callers see
        # the same "no active session" error as an unknown id.
        sessions.pop(session_id, None)
        try:
            session.client.stop(None, None, None)
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("failed to stop MCP client for GC'd session %s", session_id, exc_info=True)
        raise ValueError(f"No active session for id {session_id!r}")
    if owner is not agent:
        raise ValueError(f"No active session for id {session_id!r}")
    return session


# ---- Transports --------------------------------------------------------------------


def _no_redirect_httpx_client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """MCP-shaped ``httpx.AsyncClient`` factory that refuses to follow redirects.

    MCP's default factory sets ``follow_redirects=True``, which would allow an
    allowlisted URL to be 3xx'd to a private endpoint the SSRF guard never saw. We
    override to ``False`` so a redirect surfaces as an ``httpx.HTTPStatusError`` (or a
    3xx status the caller can see) instead of a silent hop to the new location.
    """
    kwargs: dict[str, Any] = {"follow_redirects": False}
    if timeout is None:
        kwargs["timeout"] = httpx.Timeout(30.0, read=300.0)
    else:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


def _make_http_transport(url: str):  # type: ignore[no-untyped-def]
    """Return a zero-arg callable that opens a streamable-http transport for ``url``."""
    return lambda: streamablehttp_client(
        url=url,
        httpx_client_factory=_no_redirect_httpx_client_factory,
    )
