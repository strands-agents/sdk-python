import { Buffer } from 'buffer'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { a2aClient, makeA2AClient } from '../a2a-client.js'
import { UrlNotAllowedError, validateUrl } from '../url-guard.js'

// Injectable DNS behavior — the mock module below reads `dnsMock.lookup` on
// every call, so tests can swap it out per-test without re-mocking the module.
const dnsMock: { lookup: (host: string, opts?: unknown) => Promise<Array<{ address: string; family: number }>> } = {
  lookup: async () => [{ address: '93.184.216.34', family: 4 }],
}

vi.mock('node:dns/promises', () => ({
  lookup: (host: string, opts?: unknown) => dnsMock.lookup(host, opts),
}))

// Mock A2AAgent so tests never hit the network. The tool calls
// `getAgentCard()` before sending the message; the mock reads the next queued
// card off `pendingCards`, or returns the last-set one.
const mockInvoke = vi.fn()
interface MockAgentInstance {
  card?: { url?: string; name?: string; description?: string }
  name?: string
  description?: string
  config: { url: string; [k: string]: unknown }
}
const mockAgentInstances: MockAgentInstance[] = []
const pendingCards: Array<{ url?: string; name?: string; description?: string }> = []

vi.mock('../../../a2a/a2a-agent.js', () => ({
  A2AAgent: class MockA2AAgent implements MockAgentInstance {
    card?: { url?: string; name?: string; description?: string }
    name?: string
    description?: string
    constructor(public config: { url: string; [k: string]: unknown }) {
      mockAgentInstances.push(this)
      const next = pendingCards.shift()
      if (next) {
        this.card = next
        if (next.name !== undefined) this.name = next.name
        if (next.description !== undefined) this.description = next.description
      }
    }
    async getAgentCard(): Promise<{ url?: string; name?: string; description?: string }> {
      return this.card ?? {}
    }
    async invoke(msg: string): Promise<unknown> {
      return mockInvoke(msg)
    }
  },
}))

/** Helper: set what the next-constructed mock A2AAgent will report as its card. */
function setNextCard(card: { url?: string; name?: string; description?: string }): void {
  pendingCards.push(card)
  mockInvoke.mockImplementation(async () => ({ toString: () => 'remote response', stopReason: 'endTurn' }))
}

// Fake DNS: default to a public IP for any hostname the guard resolves,
// so URL validation succeeds unless a specific test overrides it.
beforeEach(() => {
  dnsMock.lookup = async () => [{ address: '93.184.216.34', family: 4 }]
  mockInvoke.mockReset()
  mockAgentInstances.length = 0
  pendingCards.length = 0
})

afterEach(() => {
  vi.restoreAllMocks()
})

// =====================================================================
// URL guard — SSRF surface
// =====================================================================

describe('validateUrl', () => {
  it.each(['ftp://example.com', 'file:///etc/passwd', 'javascript:alert(1)', 'gopher://example.com', ''])(
    'rejects non-http scheme: %s',
    async (url) => {
      await expect(validateUrl(url)).rejects.toBeInstanceOf(UrlNotAllowedError)
    }
  )

  it.each(['http://127.0.0.1', 'http://127.1.2.3', 'http://[::1]', 'http://localhost', 'https://LOCALHOST'])(
    'rejects loopback: %s',
    async (url) => {
      await expect(validateUrl(url)).rejects.toBeInstanceOf(UrlNotAllowedError)
    }
  )

  it.each(['http://10.0.0.1', 'http://192.168.1.1', 'http://172.16.0.1', 'http://100.64.0.1'])(
    'rejects RFC1918 / CGNAT ip: %s',
    async (url) => {
      await expect(validateUrl(url)).rejects.toBeInstanceOf(UrlNotAllowedError)
    }
  )

  it('rejects the AWS metadata IP', async () => {
    await expect(validateUrl('http://169.254.169.254')).rejects.toBeInstanceOf(UrlNotAllowedError)
    await expect(validateUrl('http://169.254.169.254/latest/meta-data/')).rejects.toBeInstanceOf(UrlNotAllowedError)
  })

  it('rejects link-local ip', async () => {
    await expect(validateUrl('http://169.254.1.1')).rejects.toThrow(/link-local/)
  })

  it.each(['http://foo.internal', 'http://bar.corp', 'http://svc.local', 'http://api.home'])(
    'rejects blocked suffix: %s',
    async (url) => {
      await expect(validateUrl(url)).rejects.toBeInstanceOf(UrlNotAllowedError)
    }
  )

  it('allows a public DNS name', async () => {
    const host = await validateUrl('https://example.com')
    expect(host).toBe('example.com')
  })

  it('allows a public IP literal without DNS lookup', async () => {
    dnsMock.lookup = async () => {
      throw new Error('should not be called')
    }
    const host = await validateUrl('https://8.8.8.8')
    expect(host).toBe('8.8.8.8')
  })

  it('rejects when DNS resolves to a private IP (DNS rebinding defense)', async () => {
    dnsMock.lookup = async () => [{ address: '10.0.0.1', family: 4 }]
    await expect(validateUrl('https://sneaky.example.com')).rejects.toThrow(/private/)
  })

  it('rejects when DNS resolves to a link-local IPv6 address', async () => {
    dnsMock.lookup = async () => [{ address: 'fe80::1', family: 6 }]
    await expect(validateUrl('https://sneaky6.example.com')).rejects.toThrow(/link-local/)
  })

  it.each(['http://[fec0::1]', 'http://[fed0::1]', 'http://[fee0::1]', 'http://[fef0::1]'])(
    'rejects IPv6 site-local across the full fec0::/10 range: %s',
    async (url) => {
      await expect(validateUrl(url)).rejects.toThrow(/site-local/)
    }
  )

  it('rejects when DNS returns no addresses', async () => {
    dnsMock.lookup = async () => []
    await expect(validateUrl('https://empty.example.com')).rejects.toThrow(/resolved to no addresses/)
  })

  it('rejects when DNS lookup fails', async () => {
    dnsMock.lookup = async () => {
      throw new Error('gaierror')
    }
    await expect(validateUrl('https://nx.example.com')).rejects.toThrow(/could not resolve/)
  })

  it('honors the developer allowlist', async () => {
    await expect(validateUrl('https://api.example.com/v1', ['https://api.example.com/'])).resolves.toBe(
      'api.example.com'
    )
    await expect(validateUrl('https://evil.example.com', ['https://api.example.com/'])).rejects.toThrow(/allowlist/)
  })

  it.each([
    'http://[::ffff:127.0.0.1]',
    'http://[::ffff:7f00:1]',
    'http://[0:0:0:0:0:ffff:127.0.0.1]',
    'http://[::ffff:169.254.169.254]',
  ])('rejects IPv4-mapped IPv6: %s', async (url) => {
    await expect(validateUrl(url)).rejects.toBeInstanceOf(UrlNotAllowedError)
  })

  it('strips trailing dot before suffix match', async () => {
    await expect(validateUrl('http://foo.internal.')).rejects.toThrow(/blocked suffix/)
  })

  it.each([
    // Standard 4-hextet multicast (all IPv6 hostnames from URL.hostname).
    'http://[ff02::1]',
    'http://[ff05::1:3]',
    'http://[ff0e::abcd]',
    // Multicast expressed with a full IPv4-in-IPv6 tail.
    'http://[ff02:0:0:0:0:0:0:1]',
    // Multicast reached through the fully-expanded eight-hextet form.
    'http://[ff01:0:0:0:0:0:0:1]',
  ])('rejects IPv6 multicast literal: %s', async (url) => {
    await expect(validateUrl(url)).rejects.toThrow(/multicast/)
  })

  it.each(['http://239.255.255.250', 'http://224.0.0.1'])('rejects IPv4 multicast literal: %s', async (url) => {
    await expect(validateUrl(url)).rejects.toThrow(/multicast/)
  })

  it('rejects bare metadata hostname before DNS', async () => {
    dnsMock.lookup = async () => {
      throw new Error('should not be called for bare metadata label')
    }
    await expect(validateUrl('http://metadata')).rejects.toThrow(/metadata label/)
    await expect(validateUrl('http://METADATA/some/path')).rejects.toThrow(/metadata label/)
  })
})

// =====================================================================
// Tool: input validation + oversize rejection
// =====================================================================

describe('a2aClient input validation', () => {
  it('rejects non-http URL', async () => {
    await expect(a2aClient.invoke({ url: 'ftp://example.com', message: 'hi' })).rejects.toThrow(/scheme/)
  })

  it('rejects loopback', async () => {
    await expect(a2aClient.invoke({ url: 'http://localhost', message: 'hi' })).rejects.toThrow(/blocked suffix/)
  })

  it('rejects private IP', async () => {
    await expect(a2aClient.invoke({ url: 'http://192.168.1.1', message: 'hi' })).rejects.toThrow(/private/)
  })

  it('rejects metadata IP', async () => {
    await expect(a2aClient.invoke({ url: 'http://169.254.169.254', message: 'hi' })).rejects.toBeDefined()
  })

  it('rejects oversized message', async () => {
    const big = 'x'.repeat(64 * 1024 + 1)
    await expect(a2aClient.invoke({ url: 'https://example.com', message: big })).rejects.toThrow(/limit is/)
  })

  it('developer allowlist rejects off-list URL', async () => {
    const bounded = makeA2AClient({ allowedUrlPrefixes: ['https://api.example.com/'] })
    await expect(bounded.invoke({ url: 'https://other.example.com', message: 'hi' })).rejects.toThrow(/allowlist/)
  })

  it('rejects when multiagent depth reaches the cap', async () => {
    const mockContext = { invocationState: { multiagentDepth: 3 } } as never
    await expect(a2aClient.invoke({ url: 'https://example.com', message: 'hi' }, mockContext)).rejects.toThrow(
      /multiagentDepth=3/
    )
  })
})

// =====================================================================
// Tool: oversize card / response
// =====================================================================

describe('a2aClient oversize guards', () => {
  it('rejects an oversized agent card', async () => {
    // Craft a card whose serialized JSON blows past the default cap.
    const bigDesc = 'x'.repeat(300 * 1024)
    setNextCard({ url: 'https://example.com', name: 'remote', description: bigDesc })
    await expect(a2aClient.invoke({ url: 'https://example.com', message: 'hi' })).rejects.toThrow(/agent card is/)
  })

  it('truncates an oversized response text', async () => {
    setNextCard({ url: 'https://example.com', name: 'remote', description: 'd' })
    mockInvoke.mockImplementation(async () => {
      const big = 'y'.repeat(300 * 1024)
      return { toString: () => big, stopReason: 'endTurn' }
    })
    const out = await a2aClient.invoke({ url: 'https://example.com', message: 'hi' })
    expect(out.output).toMatch(/\[truncated\]$/)
    expect(Buffer.byteLength(out.output, 'utf-8')).toBeLessThanOrEqual(256 * 1024)
  })

  it('truncates on a code-point boundary when the cap falls mid-multibyte', async () => {
    // The response is a run of a four-byte-encoded code point (U+1F600).
    // If the truncator sliced on an arbitrary byte offset it would emit
    // U+FFFD replacement characters and the resulting string, re-encoded,
    // could exceed the cap.
    setNextCard({ url: 'https://example.com', name: 'remote', description: 'd' })
    const bounded = makeA2AClient({ maxResponseBytes: 100 })
    const grinning = '\u{1F600}'
    const big = grinning.repeat(200)
    mockInvoke.mockImplementation(async () => ({ toString: () => big, stopReason: 'endTurn' }))
    const out = await bounded.invoke({ url: 'https://example.com', message: 'hi' })
    expect(out.output).toMatch(/\[truncated\]$/)
    // Re-encoded, the output must not exceed the cap.
    expect(Buffer.byteLength(out.output, 'utf-8')).toBeLessThanOrEqual(100)
    // And must not contain the U+FFFD replacement character, which is what
    // the pre-fix truncator emitted when it split a code point.
    expect(out.output).not.toMatch(/�/)
  })

  it('rejects when the card URL points to a private host', async () => {
    setNextCard({ url: 'http://10.0.0.1', name: 'evil', description: 'x' })
    await expect(a2aClient.invoke({ url: 'https://example.com', message: 'hi' })).rejects.toThrow(/disallowed url/)
  })

  it('rejects when the card URL falls outside the developer allowlist', async () => {
    // Developer pins the tool to one remote; the remote's card advertises a
    // different public host. The allowlist is re-applied to `card.url`, so
    // the send is rejected even though the advertised host would pass the
    // SSRF checks on its own.
    const bounded = makeA2AClient({ allowedUrlPrefixes: ['https://agents.example.com/'] })
    setNextCard({ url: 'https://other.example.com/', name: 'remote', description: 'd' })
    await expect(bounded.invoke({ url: 'https://agents.example.com/one', message: 'hi' })).rejects.toThrow(
      /disallowed url/
    )
    // The message must not have been sent.
    expect(mockInvoke).not.toHaveBeenCalled()
  })
})

// =====================================================================
// Tool: happy path
// =====================================================================

describe('a2aClient happy path', () => {
  it('returns the response and agent info', async () => {
    setNextCard({ url: 'https://example.com', name: 'remote-agent', description: 'A test agent' })
    const out = await a2aClient.invoke({ url: 'https://example.com', message: 'hi remote' })
    expect(out.status).toBe('success')
    expect(out.output).toBe('remote response')
    expect(out.remoteCard).toEqual({
      name: 'remote-agent',
      description: 'A test agent',
      url: 'https://example.com',
    })
    expect(typeof out.executionTimeMs).toBe('number')
    expect(out.executionTimeMs).toBeGreaterThanOrEqual(0)
    expect(mockInvoke).toHaveBeenCalledWith('hi remote')
  })

  it('passes developer agentConfig through to A2AAgent', async () => {
    setNextCard({ url: 'https://example.com', name: 'x', description: 'y' })
    const customFactory = { createFromUrl: vi.fn() }
    const bounded = makeA2AClient({ agentConfig: { clientFactory: customFactory as never } })
    await bounded.invoke({ url: 'https://example.com', message: 'hi' })
    const inst = mockAgentInstances[mockAgentInstances.length - 1] as unknown as {
      config: { url: string; clientFactory: unknown; fetchImpl?: unknown }
    }
    expect(inst.config).toMatchObject({
      url: 'https://example.com',
      clientFactory: customFactory,
    })
    // Developer-supplied factory is expected to carry its own fetch discipline,
    // so we don't clobber it with our guarded fetch.
    expect(inst.config.fetchImpl).toBeUndefined()
  })

  it('injects a guarded fetchImpl when no clientFactory is supplied', async () => {
    setNextCard({ url: 'https://example.com', name: 'x', description: 'y' })
    await a2aClient.invoke({ url: 'https://example.com', message: 'hi' })
    const inst = mockAgentInstances[mockAgentInstances.length - 1] as unknown as {
      config: { fetchImpl?: unknown }
    }
    expect(typeof inst.config.fetchImpl).toBe('function')
  })

  it('guarded fetchImpl rejects a 302 redirect that lands on a metadata URL', async () => {
    // Grab the guarded fetch by invoking the tool once and pulling it off the
    // mocked A2AAgent. Then drive it directly against a stubbed global fetch
    // so we can prove redirect walking re-runs the URL guard on the target.
    setNextCard({ url: 'https://example.com', name: 'x', description: 'y' })
    await a2aClient.invoke({ url: 'https://example.com', message: 'hi' })
    const inst = mockAgentInstances[mockAgentInstances.length - 1] as unknown as {
      config: { fetchImpl?: typeof globalThis.fetch }
    }
    const guarded = inst.config.fetchImpl
    if (typeof guarded !== 'function') throw new Error('expected a guarded fetch')

    const originalFetch = globalThis.fetch
    const stub = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' || input instanceof URL ? input.toString() : input.url
      // First hop returns 302 pointing at the AWS metadata IP.
      if (url === 'https://public.example.com/') {
        return new Response(null, {
          status: 302,
          headers: { location: 'http://169.254.169.254/latest/meta-data/' },
        })
      }
      throw new Error(`stub fetch should not have been called for ${url}`)
    })
    globalThis.fetch = stub as unknown as typeof globalThis.fetch
    try {
      await expect(guarded('https://public.example.com/')).rejects.toBeInstanceOf(UrlNotAllowedError)
      // First hop hit, second hop rejected pre-flight — no second network call.
      expect(stub).toHaveBeenCalledTimes(1)
    } finally {
      globalThis.fetch = originalFetch
    }
  })

  it('guarded fetchImpl rejects a 302 that lands outside the developer allowlist', async () => {
    // Pin the tool to one prefix; the first hop 302s to a public host that
    // isn't on the allowlist. The guarded fetch must re-apply the allowlist
    // to the redirect target, not just to the initial URL.
    const bounded = makeA2AClient({ allowedUrlPrefixes: ['https://agents.example.com/'] })
    setNextCard({ url: 'https://agents.example.com/one', name: 'x', description: 'y' })
    await bounded.invoke({ url: 'https://agents.example.com/one', message: 'hi' })
    const inst = mockAgentInstances[mockAgentInstances.length - 1] as unknown as {
      config: { fetchImpl?: typeof globalThis.fetch }
    }
    const guarded = inst.config.fetchImpl
    if (typeof guarded !== 'function') throw new Error('expected a guarded fetch')

    const originalFetch = globalThis.fetch
    const stub = vi.fn(
      async () => new Response(null, { status: 302, headers: { location: 'https://other.example.com/two' } })
    )
    globalThis.fetch = stub as unknown as typeof globalThis.fetch
    try {
      await expect(guarded('https://agents.example.com/one')).rejects.toThrow(/allowlist/)
      expect(stub).toHaveBeenCalledTimes(1)
    } finally {
      globalThis.fetch = originalFetch
    }
  })

  it('guarded fetchImpl preserves Request method and headers when the caller passes a Request', async () => {
    setNextCard({ url: 'https://example.com', name: 'x', description: 'y' })
    await a2aClient.invoke({ url: 'https://example.com', message: 'hi' })
    const inst = mockAgentInstances[mockAgentInstances.length - 1] as unknown as {
      config: { fetchImpl?: typeof globalThis.fetch }
    }
    const guarded = inst.config.fetchImpl
    if (typeof guarded !== 'function') throw new Error('expected a guarded fetch')

    const originalFetch = globalThis.fetch
    const seenInit: Array<{ url: string; method: string; auth: string | null }> = []
    const stub = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' || input instanceof URL ? input.toString() : input.url
      const headers = new Headers(init?.headers ?? {})
      seenInit.push({ url, method: init?.method ?? 'GET', auth: headers.get('authorization') })
      return new Response('ok', { status: 200 })
    })
    globalThis.fetch = stub as unknown as typeof globalThis.fetch
    try {
      const req = new Request('https://public.example.com/', {
        method: 'POST',
        headers: { authorization: 'Bearer x' },
      })
      await guarded(req)
      expect(seenInit).toHaveLength(1)
      expect(seenInit[0]!.method).toBe('POST')
      expect(seenInit[0]!.auth).toBe('Bearer x')
    } finally {
      globalThis.fetch = originalFetch
    }
  })

  it('guarded fetchImpl forces redirect: "manual" even when init overrides it', async () => {
    // A caller-supplied `redirect: 'follow'` must not defeat manual redirect
    // walking; otherwise the runtime would follow 3xx transparently and bypass
    // the per-hop validateUrl check.
    setNextCard({ url: 'https://example.com', name: 'x', description: 'y' })
    await a2aClient.invoke({ url: 'https://example.com', message: 'hi' })
    const inst = mockAgentInstances[mockAgentInstances.length - 1] as unknown as {
      config: { fetchImpl?: typeof globalThis.fetch }
    }
    const guarded = inst.config.fetchImpl
    if (typeof guarded !== 'function') throw new Error('expected a guarded fetch')

    const originalFetch = globalThis.fetch
    const observed: RequestRedirect[] = []
    const stub = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      observed.push(init?.redirect ?? 'follow')
      return new Response('ok', { status: 200 })
    })
    globalThis.fetch = stub as unknown as typeof globalThis.fetch
    try {
      // String branch: caller passes redirect: 'follow'.
      await guarded('https://foo.example.com/', { redirect: 'follow' })
      // Request branch: caller passes redirect: 'follow' via init.
      await guarded(new Request('https://bar.example.com/'), { redirect: 'follow' })
      expect(observed).toHaveLength(2)
      expect(observed[0]).toBe('manual')
      expect(observed[1]).toBe('manual')
    } finally {
      globalThis.fetch = originalFetch
    }
  })

  it('guarded fetchImpl strips Authorization on a scheme-only origin change', async () => {
    setNextCard({ url: 'https://example.com', name: 'x', description: 'y' })
    await a2aClient.invoke({ url: 'https://example.com', message: 'hi' })
    const inst = mockAgentInstances[mockAgentInstances.length - 1] as unknown as {
      config: { fetchImpl?: typeof globalThis.fetch }
    }
    const guarded = inst.config.fetchImpl
    if (typeof guarded !== 'function') throw new Error('expected a guarded fetch')

    const originalFetch = globalThis.fetch
    const seen: Array<{ url: string; auth: string | null }> = []
    const stub = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' || input instanceof URL ? input.toString() : input.url
      const headers = new Headers(init?.headers ?? {})
      seen.push({ url, auth: headers.get('authorization') })
      if (url === 'https://foo.example.com/') {
        return new Response(null, { status: 302, headers: { location: 'http://foo.example.com/' } })
      }
      return new Response('ok', { status: 200 })
    })
    globalThis.fetch = stub as unknown as typeof globalThis.fetch
    try {
      await guarded('https://foo.example.com/', { headers: { authorization: 'Bearer x' } })
      // First hop kept the header, second hop dropped it because scheme changed.
      expect(seen[0]!.auth).toBe('Bearer x')
      expect(seen[1]!.auth).toBeNull()
    } finally {
      globalThis.fetch = originalFetch
    }
  })

  it('guarded fetchImpl caps redirects at 5 hops', async () => {
    setNextCard({ url: 'https://example.com', name: 'x', description: 'y' })
    await a2aClient.invoke({ url: 'https://example.com', message: 'hi' })
    const inst = mockAgentInstances[mockAgentInstances.length - 1] as unknown as {
      config: { fetchImpl?: typeof globalThis.fetch }
    }
    const guarded = inst.config.fetchImpl
    if (typeof guarded !== 'function') throw new Error('expected a guarded fetch')

    const originalFetch = globalThis.fetch
    let hop = 0
    const stub = vi.fn(async () => {
      hop += 1
      return new Response(null, {
        status: 302,
        headers: { location: `https://hop-${hop}.example.com/` },
      })
    })
    globalThis.fetch = stub as unknown as typeof globalThis.fetch
    try {
      await expect(guarded('https://hop-0.example.com/')).rejects.toThrow(/redirect cap/)
    } finally {
      globalThis.fetch = originalFetch
    }
  })
})

// =====================================================================
// Tool: timeout + cancellation
// =====================================================================

describe('a2aClient timeout and cancellation', () => {
  it('surfaces a timeout when the underlying invoke never resolves', async () => {
    // Card set is unused because we never resolve; return a promise that hangs.
    mockInvoke.mockImplementation(() => new Promise(() => {}))
    const bounded = makeA2AClient({ timeoutSeconds: 0.05 })
    await expect(bounded.invoke({ url: 'https://example.com', message: 'hi' })).rejects.toMatchObject({
      name: 'AbortError',
    })
  })

  it('surfaces a cancellation when the parent agent signal fires', async () => {
    mockInvoke.mockImplementation(() => new Promise(() => {}))
    const controller = new AbortController()
    const bounded = makeA2AClient({ timeoutSeconds: 60 })
    // Fire cancellation after a short delay.
    setTimeout(() => controller.abort(), 30)
    // The tool checks context.agent.cancelSignal; simulate the InvokableTool context.
    const mockContext = {
      agent: { cancelSignal: controller.signal },
    } as never
    await expect(bounded.invoke({ url: 'https://example.com', message: 'hi' }, mockContext)).rejects.toMatchObject({
      name: 'AbortError',
    })
  })

  it('rejects with AbortError when the parent cancelSignal is already aborted', async () => {
    setNextCard({ url: 'https://example.com', name: 'x', description: 'y' })
    const controller = new AbortController()
    controller.abort()
    const mockContext = {
      agent: { cancelSignal: controller.signal },
    } as never
    await expect(a2aClient.invoke({ url: 'https://example.com', message: 'hi' }, mockContext)).rejects.toMatchObject({
      name: 'AbortError',
    })
  })
})

// =====================================================================
// Tool metadata
// =====================================================================

describe('a2aClient tool metadata', () => {
  it('has the default name', () => {
    expect(a2aClient.name).toBe('a2a_client')
  })

  it('accepts a custom name', () => {
    const t = makeA2AClient({ name: 'remote_agent' })
    expect(t.name).toBe('remote_agent')
  })

  it('advertises only url and message on the schema', () => {
    const schema = a2aClient.toolSpec.inputSchema as { properties?: Record<string, unknown> }
    const props = schema.properties ?? {}
    expect(Object.keys(props).sort()).toEqual(['message', 'url'])
  })
})
