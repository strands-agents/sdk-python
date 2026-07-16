import { z } from 'zod'
import { promises as dns } from 'node:dns'
import { isIP } from 'node:net'
import { Buffer } from 'node:buffer'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import { tool } from '../../tools/tool-factory.js'
import { McpClient, type McpTransport } from '../../mcp/client.js'
import type { McpTool } from '../../tools/mcp-tool.js'
import type { InvokableTool } from '../../tools/tool.js'
import type { JSONValue } from '../../types/json.js'
import type {
  McpClientCallToolResult,
  McpClientConnectResult,
  McpClientDisconnectResult,
  McpClientListToolsResult,
  McpClientToolSpec,
} from './types.js'

const ALLOWED_SCHEMES = new Set(['http:', 'https:'])
const DEFAULT_SCHEME_PORTS: Readonly<Record<string, string>> = { 'http:': '80', 'https:': '443' }
const DEFAULT_SESSION_LIMIT = 8
const DEFAULT_CALL_TIMEOUT_SECONDS = 60
const MAX_RESULT_TEXT_CHARS = 100_000
const MAX_STRUCTURED_CONTENT_BYTES = 100_000

// Hostname suffixes rejected before DNS lookup. Names in these zones are intended
// for internal use even when they happen to resolve to a public IP for some resolver.
const BLOCKED_HOST_SUFFIXES: readonly string[] = [
  '.internal',
  '.local',
  '.localhost',
  '.corp',
  '.home',
  '.lan',
  '.intranet',
  '.private',
  '.i2p',
  '.onion',
] as const

// Well-known cloud-metadata addresses. `isNonPublicIPv4` already rejects link-local
// (169.254/16) and unique-local IPv6, but naming them explicitly guards against a
// future refactor that weakens a generic predicate.
const BLOCKED_METADATA_ADDRESSES: ReadonlySet<string> = new Set([
  '169.254.169.254', // AWS / Azure / DigitalOcean
  'fd00:ec2::254', // AWS IPv6
  '100.100.100.200', // Alibaba
  '192.0.0.192', // Oracle Cloud
])

/**
 * Options for building an agent-callable MCP client tool.
 */
export interface MakeMcpClientOptions {
  /**
   * Exact URLs the tool may connect to. Each entry must use `http:` or `https:`. Empty
   * arrays are rejected: the tool is deliberately useless without an explicit allowlist.
   */
  allowedUrls: string[]
  /** Tool name reported to the model. Defaults to `mcp_client`. */
  name?: string
  /** Tool description shown to the model. */
  description?: string
  /**
   * Maximum number of concurrent live sessions this tool instance may hold, across all
   * agents that share it. Defaults to 8.
   */
  sessionLimit?: number
  /**
   * Hook for injecting a custom MCP client factory in tests. Not part of the public API;
   * do not use in production code.
   * @internal
   */
  clientFactory?: (url: string) => McpClient
  /**
   * Hook for overriding DNS resolution in tests.
   * @internal
   */
  resolveHost?: (host: string) => Promise<string[]>
}

interface Session {
  client: McpClient
  tools: Map<string, McpTool>
  url: string
  ownerId: WeakRef<object>
}

const inputSchema = z
  .object({
    op: z
      .enum(['connect', 'list_tools', 'call_tool', 'disconnect'])
      .describe(
        'Operation: "connect" (open a session), "list_tools" (list tools on a connected server), ' +
          '"call_tool" (invoke a tool), or "disconnect" (close a session).'
      ),
    server_url: z
      .string()
      .optional()
      .describe('Server URL for op="connect". Must appear verbatim on the developer-set allowlist.'),
    session_id: z
      .string()
      .optional()
      .describe('Session identifier returned by op="connect". Required for the other three ops.'),
    tool_name: z.string().optional().describe('Name of the MCP tool to invoke. Required for op="call_tool".'),
    arguments: z
      .record(z.string(), z.unknown())
      .optional()
      .describe('Arguments to pass to the MCP tool for op="call_tool".'),
    timeout: z
      .number()
      .positive()
      .optional()
      .describe(`Per-call timeout in seconds (default: ${DEFAULT_CALL_TIMEOUT_SECONDS}). Applies to op="call_tool".`),
  })
  .refine(
    (data) => {
      if (data.op === 'connect') return typeof data.server_url === 'string' && data.server_url.length > 0
      return true
    },
    { message: '`server_url` is required when op="connect"' }
  )
  .refine(
    (data) => {
      if (data.op === 'call_tool') return typeof data.tool_name === 'string' && data.tool_name.length > 0
      return true
    },
    { message: '`tool_name` is required when op="call_tool"' }
  )
  .refine(
    (data) => {
      if (data.op === 'list_tools' || data.op === 'call_tool' || data.op === 'disconnect') {
        return typeof data.session_id === 'string' && data.session_id.length > 0
      }
      return true
    },
    { message: '`session_id` is required for op="list_tools", "call_tool", and "disconnect"' }
  )

/**
 * Creates an agent-callable MCP client tool bound to a developer-set URL allowlist.
 *
 * The tool exposes four operations under a single name — `connect`, `list_tools`,
 * `call_tool`, `disconnect` — selected by the `op` field. It is a thin shim over
 * {@link McpClient}: URL validation, session lifecycle, and result truncation live here;
 * every other concern (transport, protocol, tool invocation) is delegated.
 *
 * **Security model.** MCP servers can implement arbitrary logic, so the model must never
 * be able to point the tool at an arbitrary URL:
 *
 * - Every `connect` URL must appear verbatim on the developer-set `allowedUrls` list
 *   (after normalisation).
 * - Only `http:` and `https:` are supported. Stdio, SSE, and WebSocket transports are
 *   deliberately out of scope — see the accompanying `README.md` for the rationale.
 * - The URL's host is rejected outright if it matches a DNS suffix denylist
 *   (`.internal`, `.local`, `.localhost`, `.corp`, `.home`, `.lan`, `.intranet`,
 *   `.private`, `.i2p`, `.onion`) before any DNS lookup.
 * - The URL's host is then resolved and every returned address is checked; hostnames
 *   that resolve to private, loopback, link-local, site-local, multicast, reserved,
 *   documentation (`2001:db8::/32`, `100::/64`, `fec0::/10`), CGNAT, or
 *   IPv4-mapped-into-any-of-those space are rejected.
 * - The MCP transport's `fetch` is replaced with one that sets `redirect: 'error'`,
 *   so an allowlisted URL cannot be 3xx'd to a private endpoint the SSRF guard never
 *   saw. Redirects surface as errors rather than silent hops.
 * - Session IDs are unguessable (`crypto.randomUUID()`) and scoped to the agent that
 *   opened them via `WeakRef`; another agent using the same tool instance cannot use
 *   a session it did not open, and a garbage-collected agent's sessions become
 *   unreachable and are dropped from the session table on the next access or connect.
 * - The number of concurrent live sessions is capped (default: 8), reserved atomically
 *   at connect time so a burst of concurrent connects cannot overshoot the cap.
 * - Tool-call text and `structured_content` are size-capped before being returned to
 *   the model.
 *
 * **Limits.** The classification is a check-time judgment. Between the guard's
 * `dns.lookup` and the MCP transport's own resolve, an attacker-controlled resolver
 * could return a public IP first and a private IP second (DNS rebinding). The MCP
 * client SDK does not expose a hook for pinned-IP connects, so this shim relies on
 * the allowlist pointing at endpoints whose operator does not serve time-varying DNS.
 * Do not allowlist a hostname whose DNS you do not control.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { makeMcpClient } from '@strands-agents/sdk/vended-tools/mcp-client'
 *
 * const mcpClient = makeMcpClient({ allowedUrls: ['https://mcp.example.com/mcp'] })
 * const agent = new Agent({ tools: [mcpClient] })
 * ```
 *
 * @param options - Factory options; `allowedUrls` is required and must be non-empty.
 * @returns An {@link InvokableTool} that manages MCP sessions.
 */
export function makeMcpClient(options: MakeMcpClientOptions): InvokableTool<z.infer<typeof inputSchema>, JSONValue> {
  const allowlist = normaliseAllowlist(options.allowedUrls)
  const sessionLimit = options.sessionLimit ?? DEFAULT_SESSION_LIMIT
  if (!Number.isInteger(sessionLimit) || sessionLimit <= 0) {
    throw new Error(`\`sessionLimit\` must be a positive integer, got ${String(options.sessionLimit)}`)
  }
  const clientFactory = options.clientFactory ?? defaultClientFactory
  const resolveHost = options.resolveHost ?? defaultResolveHost
  const sessions = new Map<string, Session>()
  // Reserved slot count. Incremented synchronously before any `await`, so racing
  // connects can never overshoot the cap even while `client.connect()` is in flight.
  let reserved = 0

  return tool({
    name: options.name ?? 'mcp_client',
    description:
      options.description ??
      'Connects to Model Context Protocol (MCP) servers at runtime. Supports four operations: ' +
        '"connect" (open a session to an allowlisted server URL), "list_tools" (list the tools ' +
        'the connected server exposes), "call_tool" (invoke a tool on a connected server), and ' +
        '"disconnect" (close a session). URLs must be on a developer-set allowlist.',
    inputSchema,
    callback: async (input, context) => {
      if (!context) {
        throw new Error('Tool context is required')
      }
      const owner = context.agent

      switch (input.op) {
        case 'connect': {
          const url = input.server_url!
          rejectUserinfoAndFragment(url)
          const canonical = canonicaliseUrl(url)
          if (!allowlist.has(canonical)) {
            throw new Error(`URL "${url}" is not on the developer-set allowlist`)
          }
          // Drop sessions whose owning agent has been garbage-collected before
          // enforcing the cap. Left in place they would count against the limit
          // forever and pin the underlying transport open.
          purgeDeadSessions(sessions)
          if (sessions.size + reserved >= sessionLimit) {
            throw new Error(
              `Refusing to open a new MCP session: ${sessions.size + reserved}/${sessionLimit} concurrent sessions in use`
            )
          }
          reserved += 1
          try {
            await assertPublicHost(url, resolveHost)
            const client = clientFactory(url)
            await client.connect()
            let tools
            try {
              tools = await client.listTools()
            } catch (err) {
              // `connect()` succeeded but the follow-up threw. Tear the client
              // down so we don't leak the transport / open socket.
              await client.disconnect().catch(() => {})
              throw err
            }
            const sessionId = globalThis.crypto.randomUUID()
            const toolMap = new Map(tools.map((t) => [t.name, t]))
            sessions.set(sessionId, {
              client,
              tools: toolMap,
              url: canonical,
              ownerId: new WeakRef(owner as object),
            })
            const result: McpClientConnectResult = { session_id: sessionId, server_url: canonical }
            return result as unknown as JSONValue
          } finally {
            reserved -= 1
          }
        }

        case 'list_tools': {
          const session = resolveSession(sessions, input.session_id, owner as object)
          const specs: McpClientToolSpec[] = []
          for (const t of session.tools.values()) {
            const spec: McpClientToolSpec = {
              name: t.name,
              description: t.description || '',
              input_schema: t.toolSpec.inputSchema ?? {},
            }
            if (t.toolSpec.outputSchema !== undefined) {
              spec.output_schema = t.toolSpec.outputSchema
            }
            specs.push(spec)
          }
          const result: McpClientListToolsResult = { tools: specs }
          return result as unknown as JSONValue
        }

        case 'call_tool': {
          const session = resolveSession(sessions, input.session_id, owner as object)
          const t = session.tools.get(input.tool_name!)
          if (!t) {
            throw new Error(`Tool "${input.tool_name}" is not exposed by the connected server`)
          }
          const timeoutSeconds = input.timeout ?? DEFAULT_CALL_TIMEOUT_SECONDS
          const timeoutSignal = AbortSignal.timeout(timeoutSeconds * 1000)
          const signal = context.agent ? AbortSignal.any([timeoutSignal, context.agent.cancelSignal]) : timeoutSignal
          const rawResult = await session.client.callTool(t, (input.arguments ?? {}) as JSONValue, { signal })
          return summariseCallToolResult(rawResult) as unknown as JSONValue
        }

        case 'disconnect': {
          const session = resolveSession(sessions, input.session_id, owner as object)
          try {
            await session.client.disconnect()
          } finally {
            sessions.delete(input.session_id!)
          }
          const result: McpClientDisconnectResult = { disconnected: true }
          return result as unknown as JSONValue
        }
      }
    },
  })
}

function normaliseAllowlist(urls: string[]): Set<string> {
  if (!urls || urls.length === 0) {
    throw new Error('`allowedUrls` must not be empty; the mcp_client tool requires an explicit allowlist')
  }
  const normalised = new Set<string>()
  for (const raw of urls) {
    let parsed: URL
    try {
      parsed = new URL(raw)
    } catch {
      throw new Error(`Allowlist entry "${raw}" is not a valid URL`)
    }
    if (!ALLOWED_SCHEMES.has(parsed.protocol)) {
      throw new Error(
        `Allowlist entry "${raw}" has unsupported scheme "${parsed.protocol}"; only http and https are supported`
      )
    }
    if (!parsed.hostname) {
      throw new Error(`Allowlist entry "${raw}" has no host`)
    }
    rejectUserinfoAndFragment(raw, parsed)
    normalised.add(canonicaliseUrl(raw))
  }
  return normalised
}

/**
 * Reject URLs that carry credentials in the userinfo section or a `#` fragment.
 *
 * Both are stripped by `canonicaliseUrl`, so leaving them accepted would let
 * `https://user:pass@host/path` canonicalise to the same string as
 * `https://host/path` and bypass the verbatim allowlist match.
 */
function rejectUserinfoAndFragment(raw: string, parsed?: URL): void {
  let parsedUrl: URL
  try {
    parsedUrl = parsed ?? new URL(raw)
  } catch {
    // Callers that pass an unparseable URL surface that error elsewhere; nothing
    // to reject here.
    return
  }
  if (parsedUrl.username || parsedUrl.password) {
    throw new Error(`URL "${raw}" carries credentials; strip the userinfo before allowlisting or connecting`)
  }
  if (parsedUrl.hash) {
    throw new Error(`URL "${raw}" carries a fragment; strip it before allowlisting or connecting`)
  }
}

function canonicaliseUrl(url: string): string {
  const parsed = new URL(url)
  const scheme = parsed.protocol.toLowerCase()
  const host = parsed.hostname.toLowerCase()
  // Drop the port when it matches the scheme default so `https://host` and
  // `https://host:443` canonicalise to the same string. Explicit non-default
  // ports are preserved verbatim.
  const defaultPort = DEFAULT_SCHEME_PORTS[scheme]
  const port = parsed.port && parsed.port !== defaultPort ? `:${parsed.port}` : ''
  const path = parsed.pathname.replace(/\/+$/, '')
  const query = parsed.search
  return `${scheme}//${host}${port}${path}${query}`
}

async function defaultResolveHost(host: string): Promise<string[]> {
  const literal = isIP(host)
  if (literal !== 0) return [host]
  const results = await dns.lookup(host, { all: true })
  return results.map((r) => r.address)
}

async function assertPublicHost(url: string, resolveHost: (host: string) => Promise<string[]>): Promise<void> {
  const parsed = new URL(url)
  let host = parsed.hostname.toLowerCase()
  if (host.endsWith('.')) host = host.slice(0, -1)

  for (const suffix of BLOCKED_HOST_SUFFIXES) {
    const bare = suffix.slice(1)
    if (host === bare || host.endsWith(suffix)) {
      throw new Error(`Refusing to connect to "${url}": hostname "${host}" matches blocked suffix "${suffix}"`)
    }
  }

  let addresses: string[]
  try {
    addresses = await resolveHost(host)
  } catch (err) {
    throw new Error(`Could not resolve host "${host}": ${(err as Error).message}`, { cause: err })
  }
  if (addresses.length === 0) {
    throw new Error(`Could not resolve host "${host}" to any IP address`)
  }
  for (const addr of addresses) {
    const canonical = canonicaliseIpForMetadataCheck(addr)
    if (BLOCKED_METADATA_ADDRESSES.has(canonical)) {
      throw new Error(`Refusing to connect to "${url}": host "${host}" resolves to metadata address ${addr}`)
    }
    if (isNonPublic(addr)) {
      throw new Error(`Refusing to connect to "${url}": host "${host}" resolves to non-public address ${addr}`)
    }
  }
}

/**
 * Lowercase an IP literal and unwrap `::ffff:` prefixes so the metadata-address set can
 * be matched against both bare and mapped forms.
 */
function canonicaliseIpForMetadataCheck(address: string): string {
  const lower = address.toLowerCase()
  const mapped = extractIpv4Mapped(lower)
  return mapped ?? lower
}

/**
 * Returns true when the address is in RFC1918 private space, loopback, link-local,
 * multicast, reserved, or the unspecified address. Rough but sufficient for the
 * defence-in-depth SSRF guard applied on top of the allowlist.
 */
function isNonPublic(address: string): boolean {
  const kind = isIP(address)
  if (kind === 4) return isNonPublicIPv4(address)
  if (kind === 6) return isNonPublicIPv6(address)
  // Unknown format — reject conservatively.
  return true
}

function isNonPublicIPv4(address: string): boolean {
  const parts = address.split('.').map((p) => Number(p))
  if (parts.length !== 4 || parts.some((p) => Number.isNaN(p) || p < 0 || p > 255)) return true
  const [a, b, c] = parts as [number, number, number, number]
  if (a === 10) return true // 10.0.0.0/8
  if (a === 127) return true // loopback
  if (a === 0) return true // unspecified / 0.0.0.0/8
  if (a === 169 && b === 254) return true // link-local
  if (a === 172 && b >= 16 && b <= 31) return true // 172.16.0.0/12
  if (a === 192 && b === 168) return true // 192.168.0.0/16
  if (a >= 224) return true // multicast + reserved (224.0.0.0/4, 240.0.0.0/4)
  if (a === 100 && b >= 64 && b <= 127) return true // CGNAT 100.64.0.0/10
  if (a === 192 && b === 0 && c === 0) return true // 192.0.0.0/24 (IETF reserved incl. 192.0.0.192)
  if (a === 192 && b === 0 && c === 2) return true // 192.0.2.0/24 documentation
  if (a === 198 && (b === 18 || b === 19)) return true // 198.18.0.0/15 benchmarking
  if (a === 198 && b === 51 && c === 100) return true // 198.51.100.0/24 documentation
  if (a === 203 && b === 0 && c === 113) return true // 203.0.113.0/24 documentation
  return false
}

function isNonPublicIPv6(address: string): boolean {
  const normalised = address.toLowerCase()
  if (normalised === '::' || normalised === '::1') return true
  // Link-local fe80::/10 → first 10 bits `1111 1110 10`, first hextet `fe80`–`febf`.
  if (/^fe[89ab][0-9a-f]:/.test(normalised) || /^fe[89ab][0-9a-f]::/.test(normalised)) return true
  // Site-local fec0::/10 → first 10 bits `1111 1110 11`, first hextet `fec0`–`feff`.
  // Deprecated by RFC 3879 but still worth blocking as internal namespace.
  if (/^fe[cdef][0-9a-f]:/.test(normalised) || /^fe[cdef][0-9a-f]::/.test(normalised)) return true
  // Unique local fc00::/7 → first byte fc or fd.
  if (/^f[cd][0-9a-f]{2}:/.test(normalised) || /^f[cd]::/.test(normalised)) return true
  // Multicast ff00::/8.
  if (normalised.startsWith('ff')) return true
  // Discard-only 100::/64 — compressed form is `100::` (optionally with a suffix);
  // fully-expanded form is `0(1)00:0000:0000:0000:...`. Both are rejected.
  if (normalised === '100::' || normalised.startsWith('100::')) return true
  if (/^0*100:0+:0+:0+:/.test(normalised)) return true
  // Documentation range 2001:db8::/32 (compressed `2001:db8:` or expanded `2001:0db8:`).
  if (normalised.startsWith('2001:db8:') || normalised.startsWith('2001:0db8:')) return true
  const mapped = extractIpv4Mapped(normalised)
  if (mapped !== null) {
    return isIP(mapped) === 4 ? isNonPublicIPv4(mapped) : true
  }
  return false
}

/**
 * Return the embedded IPv4 dotted-quad from an IPv4-mapped IPv6 literal, in either
 * compressed (`::ffff:a.b.c.d`) or fully-expanded (`0:0:0:0:0:ffff:a.b.c.d`) form.
 * Returns `null` when the input is not IPv4-mapped.
 */
function extractIpv4Mapped(address: string): string | null {
  const lower = address.toLowerCase()
  // Compressed: leading `::ffff:`, optionally with an extra `0:` group (`::ffff:0:0:a.b.c.d`).
  const compressed = /^::ffff:(?:0:)?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$/.exec(lower)
  if (compressed) return compressed[1] ?? null
  // Expanded: seven groups of `0:` (or `0000:`) then `ffff:` then the dotted quad.
  const expanded = /^(?:0{1,4}:){5}ffff:(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$/.exec(lower)
  if (expanded) return expanded[1] ?? null
  return null
}

/**
 * A `fetch` shim that refuses to follow HTTP redirects.
 *
 * The MCP SDK's `StreamableHTTPClientTransport` accepts a custom `fetch` in its options
 * bag. Wiring `redirect: 'error'` there forces the platform `fetch` to throw on any 3xx
 * response, so an allowlisted URL cannot be redirected to a private endpoint the SSRF
 * guard never saw. The MCP SDK does not expose a manual redirect walker, so this is
 * strictly a reject-on-3xx policy.
 *
 * @internal exported so tests can assert the redirect flag is actually set.
 */
export function noRedirectFetch(input: string | URL | Request, init?: RequestInit): Promise<Response> {
  const merged: RequestInit = { ...(init ?? {}), redirect: 'error' }
  return globalThis.fetch(input as Parameters<typeof globalThis.fetch>[0], merged)
}

/**
 * Default MCP client factory: constructs a {@link McpClient} whose underlying
 * `StreamableHTTPClientTransport` uses {@link noRedirectFetch}, so redirects surface as
 * errors instead of silently landing on a new host.
 */
function defaultClientFactory(url: string): McpClient {
  const parsed = new URL(url)
  const transport = new StreamableHTTPClientTransport(parsed, {
    fetch: noRedirectFetch as unknown as (input: string | URL, init?: RequestInit) => Promise<Response>,
  })
  return new McpClient({ transport: transport as unknown as McpTransport })
}

function resolveSession(sessions: Map<string, Session>, sessionId: string | undefined, owner: object): Session {
  if (!sessionId) {
    throw new Error('`session_id` is required')
  }
  const session = sessions.get(sessionId)
  if (!session) {
    throw new Error(`No active session for id "${sessionId}"`)
  }
  const ownerRef = session.ownerId.deref()
  if (ownerRef === undefined) {
    // Owning agent has been garbage-collected. Reap this specific session so the
    // slot is returned to the pool and the transport is torn down; callers see
    // the same "no active session" error as an unknown id.
    sessions.delete(sessionId)
    session.client.disconnect().catch(() => {})
    throw new Error(`No active session for id "${sessionId}"`)
  }
  // A foreign agent produces the same "no session" error as a missing id —
  // probing for other agents' ids is not informative.
  if (ownerRef !== owner) {
    throw new Error(`No active session for id "${sessionId}"`)
  }
  return session
}

/**
 * Drop sessions whose owning agent has been garbage-collected and tear down their
 * clients on a best-effort basis. Called before enforcing the session cap so a
 * stream of connect-and-forget agents cannot pin the cap at zero.
 */
function purgeDeadSessions(sessions: Map<string, Session>): void {
  for (const [sid, session] of sessions) {
    if (session.ownerId.deref() === undefined) {
      sessions.delete(sid)
      session.client.disconnect().catch(() => {})
    }
  }
}

function summariseCallToolResult(raw: unknown): McpClientCallToolResult {
  if (!raw || typeof raw !== 'object') {
    return { status: 'error', text: '' }
  }
  const record = raw as Record<string, unknown>
  const content = Array.isArray(record.content) ? record.content : []
  const textParts: string[] = []
  for (const item of content) {
    if (item && typeof item === 'object') {
      const t = (item as Record<string, unknown>).text
      if (typeof t === 'string') textParts.push(t)
    }
  }
  let text = textParts.join('\n')
  let truncated = false
  if (text.length > MAX_RESULT_TEXT_CHARS) {
    text = text.slice(0, MAX_RESULT_TEXT_CHARS)
    truncated = true
  }
  const result: McpClientCallToolResult = {
    status: record.isError ? 'error' : 'success',
    text,
  }
  if (record.structuredContent !== undefined) {
    const capped = capStructuredContent(record.structuredContent)
    result.structured_content = capped.value
    if (capped.truncated) truncated = true
  }
  if (truncated) result.truncated = true
  if (record.isError) result.is_error = true
  return result
}

/**
 * Serialise `value` to JSON and, when it exceeds the size cap, swap it for a marker
 * object announcing the truncation. Non-JSON-serialisable input is also swapped so the
 * model never receives an opaque or oversized blob.
 */
function capStructuredContent(value: unknown): { value: JSONValue; truncated: boolean } {
  let encoded: string
  try {
    encoded = JSON.stringify(value)
  } catch {
    return {
      value: { __truncated__: true, reason: 'structured_content was not JSON-serialisable' } as JSONValue,
      truncated: true,
    }
  }
  if (encoded === undefined) {
    return {
      value: { __truncated__: true, reason: 'structured_content was not JSON-serialisable' } as JSONValue,
      truncated: true,
    }
  }
  const byteLength = Buffer.byteLength(encoded, 'utf8')
  if (byteLength <= MAX_STRUCTURED_CONTENT_BYTES) {
    return { value: value as JSONValue, truncated: false }
  }
  return {
    value: {
      __truncated__: true,
      reason: 'structured_content exceeded size cap',
      size: byteLength,
    } as JSONValue,
    truncated: true,
  }
}
