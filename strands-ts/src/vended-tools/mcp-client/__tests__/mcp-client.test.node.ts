import { describe, expect, it, vi } from 'vitest'
import { makeMcpClient, noRedirectFetch } from '../mcp-client.js'
import { McpClient } from '../../../mcp/client.js'
import { McpTool } from '../../../tools/mcp-tool.js'
import type { ToolContext } from '../../../tools/tool.js'
import type { McpClientCallToolResult, McpClientConnectResult, McpClientListToolsResult } from '../types.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'

type FakeMcpClient = Partial<McpClient> & {
  connect: ReturnType<typeof vi.fn>
  disconnect: ReturnType<typeof vi.fn>
  listTools: ReturnType<typeof vi.fn>
  callTool: ReturnType<typeof vi.fn>
}

interface FakeClientOptions {
  tools?: McpTool[]
  callToolResult?: unknown
}

function fakeMcpClient(options: FakeClientOptions = {}): FakeMcpClient {
  const client: FakeMcpClient = {
    connect: vi.fn().mockResolvedValue(undefined),
    disconnect: vi.fn().mockResolvedValue(undefined),
    listTools: vi.fn().mockResolvedValue(options.tools ?? []),
    callTool: vi.fn().mockResolvedValue(options.callToolResult ?? { content: [{ type: 'text', text: 'ok' }] }),
  }
  return client
}

function makeMcpToolStub(name: string): McpTool {
  return new McpTool({
    name,
    description: `${name} tool`,
    inputSchema: { type: 'object', properties: { msg: { type: 'string' } } },
    client: {} as McpClient,
  })
}

function buildContext(agent = createMockAgent()): ToolContext {
  return {
    toolUse: { name: 'mcp_client', toolUseId: 'test-id', input: {} },
    agent,
    invocationState: {},
    interrupt: () => {
      throw new Error('interrupt not available in mock context')
    },
  } as unknown as ToolContext
}

const PUBLIC_IP = '93.184.216.34'

describe('makeMcpClient', () => {
  describe('allowlist enforcement', () => {
    it('rejects a URL that is not on the allowlist', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://evil.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/not on the developer-set allowlist/)
    })

    it('matches allowlist entries case-insensitively on scheme and host', async () => {
      const fake = fakeMcpClient()
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fake as unknown as McpClient,
      })
      const result = (await t.invoke(
        { op: 'connect', server_url: 'HTTPS://MCP.EXAMPLE.COM/mcp/' },
        buildContext()
      )) as unknown as McpClientConnectResult
      expect(result.session_id).toBeTruthy()
      expect(result.server_url).toBe('https://mcp.example.com/mcp')
      expect(fake.connect).toHaveBeenCalledOnce()
    })

    it('rejects an empty allowlist at construction time', () => {
      expect(() => makeMcpClient({ allowedUrls: [] })).toThrow(/must not be empty/)
    })

    it('rejects an unsupported scheme in the allowlist at construction time', () => {
      expect(() => makeMcpClient({ allowedUrls: ['file:///etc/passwd'] })).toThrow(/unsupported scheme/)
    })

    it('rejects a malformed URL in the allowlist at construction time', () => {
      expect(() => makeMcpClient({ allowedUrls: ['not a url'] })).toThrow(/not a valid URL/)
    })

    it('rejects an allowlist entry that carries credentials', () => {
      // A canonicaliser that dropped userinfo would let `https://user:pass@host/x`
      // collide with `https://host/x` on the allowlist.
      expect(() => makeMcpClient({ allowedUrls: ['https://user:pass@mcp.example.com/mcp'] })).toThrow(/credentials/)
    })

    it('rejects an allowlist entry that carries a fragment', () => {
      expect(() => makeMcpClient({ allowedUrls: ['https://mcp.example.com/mcp#anchor'] })).toThrow(/fragment/)
    })

    it('rejects a connect URL that carries credentials', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://user:pass@mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/credentials/)
    })

    it('rejects a connect URL that carries a fragment', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp#x' }, buildContext())
      ).rejects.toThrow(/fragment/)
    })

    it('matches an allowlist entry with a default port against a connect URL without one', async () => {
      // `https://host` and `https://host:443` are the same URL; the canonicaliser
      // must treat them as equal so a well-known-port allowlist entry still lets
      // a model connect with the bare form (and vice versa).
      const fake = fakeMcpClient()
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com:443/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fake as unknown as McpClient,
      })
      const result = (await t.invoke(
        { op: 'connect', server_url: 'https://mcp.example.com/mcp' },
        buildContext()
      )) as unknown as McpClientConnectResult
      expect(result.session_id).toBeTruthy()
    })
  })

  describe('sessionLimit validation', () => {
    it.each([0, -1, 1.5, Number.NaN])('rejects non-positive-integer sessionLimit %s', (bad) => {
      expect(() => makeMcpClient({ allowedUrls: ['https://mcp.example.com/mcp'], sessionLimit: bad })).toThrow(
        /positive integer/
      )
    })
  })

  describe('no-redirect fetch', () => {
    it('forces redirect: "error" on the outgoing request even when the caller omits it', async () => {
      // The MCP transport's default `fetch` follows 3xx, which would let an
      // allowlisted URL be redirected onto a private endpoint the SSRF guard
      // never saw. Assert the shim we wire into the transport actually flips
      // the flag on the outgoing request.
      let observedInit: RequestInit | undefined
      const originalFetch = globalThis.fetch
      globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
        observedInit = init
        return Promise.resolve(new Response('', { status: 200 }))
      }) as typeof globalThis.fetch
      try {
        await noRedirectFetch('https://mcp.example.com/mcp', { method: 'POST' })
      } finally {
        globalThis.fetch = originalFetch
      }
      expect(observedInit?.redirect).toBe('error')
    })

    it('overrides a caller-provided redirect: "follow"', async () => {
      let observedInit: RequestInit | undefined
      const originalFetch = globalThis.fetch
      globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
        observedInit = init
        return Promise.resolve(new Response('', { status: 200 }))
      }) as typeof globalThis.fetch
      try {
        await noRedirectFetch('https://mcp.example.com/mcp', { redirect: 'follow' })
      } finally {
        globalThis.fetch = originalFetch
      }
      expect(observedInit?.redirect).toBe('error')
    })
  })

  // Direct end-to-end coverage of dead-session purging lives on the Python side
  // where `gc.collect()` is deterministic. Node.js has no test hook to force
  // `WeakRef` targets to be reclaimed. The dead-ref branch in `resolveSession`
  // is exercised via the type system and mirror-tested in the Python suite.

  describe('SSRF guard', () => {
    it('rejects a hostname that resolves to a private IPv4', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => ['10.0.0.5'],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/non-public address/)
    })

    it('rejects a hostname that resolves to loopback', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => ['127.0.0.1'],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/non-public address/)
    })

    it('rejects a hostname that resolves to link-local (metadata endpoint)', async () => {
      const t = makeMcpClient({
        allowedUrls: ['http://169.254.169.254/mcp'],
        resolveHost: async (host) => [host],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'http://169.254.169.254/mcp' }, buildContext())
      ).rejects.toThrow(/metadata address|non-public address/)
    })

    it('rejects a hostname that resolves to an IPv6 loopback', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => ['::1'],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/non-public address/)
    })

    it('rejects a CGNAT IPv4', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => ['100.64.0.1'],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/non-public address/)
    })

    it('rejects IPv4-mapped IPv6 CGNAT (compressed form)', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => ['::ffff:100.64.0.1'],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/non-public address/)
    })

    it('rejects IPv4-mapped IPv6 CGNAT (expanded form)', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => ['0:0:0:0:0:ffff:100.64.0.1'],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/non-public address/)
    })

    it('rejects IPv4-mapped IPv6 private (::ffff:10.0.0.5)', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => ['::ffff:10.0.0.5'],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/non-public address/)
    })

    it.each([
      'svc.internal',
      'db.corp',
      'printer.local',
      'gateway.home',
      'onboarding.i2p',
      'secret.onion',
      'svc.internal.',
    ])('rejects hostname %s on the suffix denylist before DNS', async (host) => {
      const t = makeMcpClient({
        allowedUrls: [`https://${host}/mcp`],
        // resolveHost intentionally omitted — should never be called.
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(t.invoke({ op: 'connect', server_url: `https://${host}/mcp` }, buildContext())).rejects.toThrow(
        /blocked suffix/
      )
    })

    it.each([
      ['IPv6 multicast', 'ff02::1'],
      ['IPv6 site-local (fec0::/10)', 'fec0::1'],
      ['IPv6 site-local upper edge (feff::)', 'feff::1'],
      ['IPv6 documentation (2001:db8::/32) compressed', '2001:db8::1'],
      ['IPv6 documentation (2001:db8::/32) expanded', '2001:0db8:85a3::8a2e:0370:7334'],
      ['IPv6 discard-only 100::/64', '100::1'],
    ])('rejects %s', async (_label, ip) => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [ip],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/non-public address/)
    })

    it('rejects an unresolvable host', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => {
          throw new Error('ENOTFOUND')
        },
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/Could not resolve host/)
    })
  })

  describe('session scoping', () => {
    it('rejects a session id belonging to another agent', async () => {
      const fake = fakeMcpClient()
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fake as unknown as McpClient,
      })
      const agentA = createMockAgent()
      const agentB = createMockAgent()
      const connect = (await t.invoke(
        { op: 'connect', server_url: 'https://mcp.example.com/mcp' },
        buildContext(agentA)
      )) as unknown as McpClientConnectResult
      await expect(
        t.invoke({ op: 'list_tools', session_id: connect.session_id }, buildContext(agentB))
      ).rejects.toThrow(/No active session/)
    })

    it('rejects a missing session id', async () => {
      const t = makeMcpClient({ allowedUrls: ['https://mcp.example.com/mcp'] })
      await expect(t.invoke({ op: 'list_tools' } as never, buildContext())).rejects.toThrow(/session_id/i)
    })

    it('rejects an unknown session id', async () => {
      const t = makeMcpClient({ allowedUrls: ['https://mcp.example.com/mcp'] })
      await expect(t.invoke({ op: 'list_tools', session_id: 'does-not-exist' }, buildContext())).rejects.toThrow(
        /No active session/
      )
    })
  })

  describe('session limit', () => {
    it('rejects further connects once the limit is reached', async () => {
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        sessionLimit: 1,
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fakeMcpClient() as unknown as McpClient,
      })
      await t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/concurrent sessions/)
    })
  })

  describe('full lifecycle', () => {
    it('supports connect -> list_tools -> call_tool -> disconnect', async () => {
      const echoTool = makeMcpToolStub('echo')
      const fake = fakeMcpClient({
        tools: [echoTool],
        callToolResult: { content: [{ type: 'text', text: 'hello' }] },
      })
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fake as unknown as McpClient,
      })
      const agent = createMockAgent()

      const connect = (await t.invoke(
        { op: 'connect', server_url: 'https://mcp.example.com/mcp' },
        buildContext(agent)
      )) as unknown as McpClientConnectResult

      const listed = (await t.invoke(
        { op: 'list_tools', session_id: connect.session_id },
        buildContext(agent)
      )) as unknown as McpClientListToolsResult
      expect(listed.tools).toHaveLength(1)
      expect(listed.tools[0]?.name).toBe('echo')

      const called = (await t.invoke(
        {
          op: 'call_tool',
          session_id: connect.session_id,
          tool_name: 'echo',
          arguments: { msg: 'hi' },
        },
        buildContext(agent)
      )) as unknown as McpClientCallToolResult
      expect(called.status).toBe('success')
      expect(called.text).toBe('hello')
      expect(fake.callTool).toHaveBeenCalledOnce()

      const disconnected = await t.invoke({ op: 'disconnect', session_id: connect.session_id }, buildContext(agent))
      expect(disconnected).toEqual({ disconnected: true })
      expect(fake.disconnect).toHaveBeenCalledOnce()

      // Session is gone after disconnect.
      await expect(t.invoke({ op: 'list_tools', session_id: connect.session_id }, buildContext(agent))).rejects.toThrow(
        /No active session/
      )
    })

    it('rejects call_tool for a tool the server did not expose', async () => {
      const fake = fakeMcpClient({ tools: [makeMcpToolStub('echo')] })
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fake as unknown as McpClient,
      })
      const agent = createMockAgent()
      const connect = (await t.invoke(
        { op: 'connect', server_url: 'https://mcp.example.com/mcp' },
        buildContext(agent)
      )) as unknown as McpClientConnectResult
      await expect(
        t.invoke({ op: 'call_tool', session_id: connect.session_id, tool_name: 'unknown' }, buildContext(agent))
      ).rejects.toThrow(/not exposed by the connected server/)
    })
  })

  describe('result truncation', () => {
    it('truncates oversized text results', async () => {
      const big = 'x'.repeat(250_000)
      const fake = fakeMcpClient({
        tools: [makeMcpToolStub('echo')],
        callToolResult: { content: [{ type: 'text', text: big }] },
      })
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fake as unknown as McpClient,
      })
      const agent = createMockAgent()
      const connect = (await t.invoke(
        { op: 'connect', server_url: 'https://mcp.example.com/mcp' },
        buildContext(agent)
      )) as unknown as McpClientConnectResult
      const called = (await t.invoke(
        { op: 'call_tool', session_id: connect.session_id, tool_name: 'echo' },
        buildContext(agent)
      )) as unknown as McpClientCallToolResult
      expect(called.truncated).toBe(true)
      expect(called.text.length).toBeLessThan(big.length)
    })
  })

  describe('call_tool result mapping', () => {
    it('flags server-reported errors on the result', async () => {
      const fake = fakeMcpClient({
        tools: [makeMcpToolStub('echo')],
        callToolResult: { content: [{ type: 'text', text: 'boom' }], isError: true },
      })
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fake as unknown as McpClient,
      })
      const agent = createMockAgent()
      const connect = (await t.invoke(
        { op: 'connect', server_url: 'https://mcp.example.com/mcp' },
        buildContext(agent)
      )) as unknown as McpClientConnectResult
      const called = (await t.invoke(
        { op: 'call_tool', session_id: connect.session_id, tool_name: 'echo' },
        buildContext(agent)
      )) as unknown as McpClientCallToolResult
      expect(called.status).toBe('error')
      expect(called.is_error).toBe(true)
      expect(called.text).toBe('boom')
    })

    it('passes structured content through', async () => {
      const fake = fakeMcpClient({
        tools: [makeMcpToolStub('echo')],
        callToolResult: { content: [], structuredContent: { answer: 42 } },
      })
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fake as unknown as McpClient,
      })
      const agent = createMockAgent()
      const connect = (await t.invoke(
        { op: 'connect', server_url: 'https://mcp.example.com/mcp' },
        buildContext(agent)
      )) as unknown as McpClientConnectResult
      const called = (await t.invoke(
        { op: 'call_tool', session_id: connect.session_id, tool_name: 'echo' },
        buildContext(agent)
      )) as unknown as McpClientCallToolResult
      expect(called.structured_content).toEqual({ answer: 42 })
    })

    it('caps oversized structured_content and flags truncation', async () => {
      const big = { payload: 'y'.repeat(250_000) }
      const fake = fakeMcpClient({
        tools: [makeMcpToolStub('echo')],
        callToolResult: { content: [{ type: 'text', text: 'ok' }], structuredContent: big },
      })
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fake as unknown as McpClient,
      })
      const agent = createMockAgent()
      const connect = (await t.invoke(
        { op: 'connect', server_url: 'https://mcp.example.com/mcp' },
        buildContext(agent)
      )) as unknown as McpClientConnectResult
      const called = (await t.invoke(
        { op: 'call_tool', session_id: connect.session_id, tool_name: 'echo' },
        buildContext(agent)
      )) as unknown as McpClientCallToolResult
      expect(called.truncated).toBe(true)
      expect(called.structured_content).not.toEqual(big)
      expect((called.structured_content as { __truncated__?: boolean }).__truncated__).toBe(true)
    })
  })

  describe('connect cleanup on failure', () => {
    it('disconnects the client if listTools throws after connect succeeds', async () => {
      const fake: FakeMcpClient = {
        connect: vi.fn().mockResolvedValue(undefined),
        disconnect: vi.fn().mockResolvedValue(undefined),
        listTools: vi.fn().mockRejectedValue(new Error('listTools boom')),
        callTool: vi.fn().mockResolvedValue({}),
      }
      const t = makeMcpClient({
        allowedUrls: ['https://mcp.example.com/mcp'],
        resolveHost: async () => [PUBLIC_IP],
        clientFactory: () => fake as unknown as McpClient,
      })
      await expect(
        t.invoke({ op: 'connect', server_url: 'https://mcp.example.com/mcp' }, buildContext())
      ).rejects.toThrow(/listTools boom/)
      // Without the cleanup path the client would stay connected forever;
      // asserting `disconnect` was invoked proves the transport is torn down.
      expect(fake.disconnect).toHaveBeenCalledOnce()
    })
  })

  describe('input validation', () => {
    it('rejects op=connect without server_url', async () => {
      const t = makeMcpClient({ allowedUrls: ['https://mcp.example.com/mcp'] })
      await expect(t.invoke({ op: 'connect' } as never, buildContext())).rejects.toThrow()
    })

    it('rejects op=call_tool without tool_name', async () => {
      const t = makeMcpClient({ allowedUrls: ['https://mcp.example.com/mcp'] })
      await expect(t.invoke({ op: 'call_tool', session_id: 'x' } as never, buildContext())).rejects.toThrow()
    })
  })
})
