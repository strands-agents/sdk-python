# MCP Client Tool

Agent-callable Model Context Protocol (MCP) client. Lets the model, at runtime, open a
connection to an MCP server on a developer-set allowlist, discover its tools, invoke one,
and disconnect. Thin shim over the SDK's `McpClient` — all transport and protocol
handling lives there.

The tool is safe only when the developer has decided, at construction time, which servers
the model may reach. That decision is made in code — the tool does not, and should not,
prompt for interactive consent at runtime. If you want a human-in-the-loop confirmation,
use the SDK's HITL intervention on top of this tool; do not weaken the allowlist.

## Usage

```typescript
import { Agent } from '@strands-agents/sdk'
import { makeMcpClient } from '@strands-agents/sdk/vended-tools/mcp-client'

const mcpClient = makeMcpClient({
  allowedUrls: ['https://mcp.example.com/mcp', 'https://tools.internal-org.com/mcp'],
})

const agent = new Agent({ tools: [mcpClient] })
```

## Operations

The tool exposes four operations under a single tool name, selected by an `op` field:

| Op           | Required inputs                                            | Returns                                                            |
| ------------ | ---------------------------------------------------------- | ------------------------------------------------------------------ |
| `connect`    | `server_url`                                               | `{ session_id, server_url }`                                       |
| `list_tools` | `session_id`                                               | `{ tools: [{ name, description, input_schema, output_schema? }] }` |
| `call_tool`  | `session_id`, `tool_name`, optional `arguments`, `timeout` | `{ status, text, truncated?, is_error?, structured_content? }`     |
| `disconnect` | `session_id`                                               | `{ disconnected: true }`                                           |

## Security model

The allowlist is the primary control. Each `allowedUrls` entry is canonicalised (scheme
and host lowercased, trailing slash stripped, default port for the scheme dropped, and
fragment removed) and each `connect` URL is matched verbatim against that set; anything
else is rejected before any network I/O. URLs carrying credentials in the userinfo
section or a `#` fragment are rejected outright so they cannot bypass the verbatim
match. The allowlist must be non-empty and every entry must use `http:` or `https:`.

Layered on top of the allowlist, an SSRF guard rejects hosts whose name matches a
denylist of internal-use suffixes (`.internal`, `.local`, `.localhost`, `.corp`,
`.home`, `.lan`, `.intranet`, `.private`, `.i2p`, `.onion`) before DNS is consulted,
then resolves the host and rejects any address in private (RFC1918), loopback,
link-local, site-local (`fec0::/10`), CGNAT (`100.64.0.0/10`), multicast, reserved,
IPv6 documentation (`2001:db8::/32`), or discard-only (`100::/64`) space. IPv4-mapped
IPv6 (both compressed `::ffff:a.b.c.d` and fully-expanded forms) is unwrapped before
classification. Cloud-metadata endpoints (AWS, Azure, Alibaba, Oracle) are spelt out
explicitly for defence in depth.

The classification is a check-time judgment. A resolver that returns a public IP
first and a private IP second (DNS rebinding) is not fully closed off — the MCP
client SDK does not expose a hook for pinned-IP connects. Only allowlist hostnames
whose DNS you control.

Both SDKs reject HTTP redirects rather than following them. In Python, the httpx
client MCP uses is constructed with `follow_redirects=False`, so a redirect surfaces
as an `httpx.HTTPStatusError` or a 3xx status the caller sees. In TypeScript, the
MCP transport's `fetch` is replaced with a shim that sets `redirect: 'error'`, so
the platform `fetch` throws on any 3xx. Manual redirect walking is not implemented;
if you allowlist an origin whose canonical path is a redirect, allowlist the final
URL instead.

Only `http:` and `https:` (MCP streamable-http transport) are supported. Stdio, SSE,
and WebSocket transports are deliberately out of scope: stdio would let the model
choose between developer-approved binaries at runtime, expanding the subprocess attack
surface for negligible value over the developer-wired path; SSE is being deprecated by
the MCP spec in favour of streamable-http; WebSocket is not a documented MCP client
transport in the spec.

Session identifiers are `crypto.randomUUID()` values held in a `WeakRef` keyed on the
agent that opened them. Another agent using the same tool instance cannot use a
session it did not open, and a garbage-collected agent's sessions become unreachable
and are dropped from the session table (with a best-effort disconnect) the next time
that entry is touched or the next connect walks the table. Both cases produce the
same "no active session" error, so probing for other agents' ids is not informative.
The number of concurrent live sessions across all agents sharing the tool instance is
bounded by `sessionLimit` (default eight), with a pending-slot reservation that bumps
the count synchronously at connect entry so a burst of concurrent connects cannot
overshoot the cap.

`sessionLimit` must be a positive integer; zero, negative, or non-integer values are
rejected at construction. The tool set exposed by the server is fetched once at
connect and cached on the session — `list_tools` reads that cache, and `call_tool`
validates the requested name against it locally so an unadvertised name fails with
a clear error instead of being forwarded to the server. The cache can go stale if
the server changes its tool set mid-session, which is not a supported MCP flow today.

Tool-call output is size-capped before being returned to the model. Text content is
concatenated and truncated to 100,000 characters; `structured_content` is
JSON-serialised and capped at 100,000 bytes, and on overflow is replaced with a
`{ __truncated__: true }` marker. When any cap fires, the response includes
`truncated: true`.

## API

### `makeMcpClient(options)`

Returns an `InvokableTool` that manages MCP sessions.

**Options:**

| Field          | Type       | Required | Description                                                    |
| -------------- | ---------- | -------- | -------------------------------------------------------------- |
| `allowedUrls`  | `string[]` | Yes      | Exact URLs the tool may connect to. Non-empty, `http`/`https`. |
| `name`         | `string`   | No       | Tool name. Defaults to `mcp_client`.                           |
| `description`  | `string`   | No       | Tool description.                                              |
| `sessionLimit` | `number`   | No       | Maximum concurrent live sessions. Defaults to 8.               |

### Inputs

| Field        | Type                                                       | When required           | Description                                         |
| ------------ | ---------------------------------------------------------- | ----------------------- | --------------------------------------------------- |
| `op`         | `'connect' \| 'list_tools' \| 'call_tool' \| 'disconnect'` | always                  | The operation to perform.                           |
| `server_url` | `string`                                                   | `op='connect'`          | URL on the developer-set allowlist.                 |
| `session_id` | `string`                                                   | any op except `connect` | Session identifier returned by `connect`.           |
| `tool_name`  | `string`                                                   | `op='call_tool'`        | Name of the MCP tool to invoke.                     |
| `arguments`  | `Record<string, unknown>`                                  | optional                | Arguments for the invoked tool.                     |
| `timeout`    | `number` (seconds)                                         | optional                | Per-call timeout. Defaults to 60. `call_tool` only. |

### Outputs

Discriminated by `op`, see the table above.
