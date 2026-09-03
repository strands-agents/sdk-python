import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import type { ToolContext } from '../../../tools/tool.js'
import type { LocalAgent } from '../../../types/agent.js'
import * as agentModule from '../../../agent/agent.js'
import { htmlToMarkdown } from '../extract.js'
import { makeWebFetch, webFetch, DEFAULT_MAX_BYTES, DEFAULT_MAX_CONTENT_CHARS } from '../web-fetch.js'
import { WEB_FETCH_DESCRIPTION_MARKDOWN, WEB_FETCH_DESCRIPTION_AGENTIC } from '../types.js'

describe('webFetch tool', () => {
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  function mockFetch(
    body: string,
    options: { status?: number; statusText?: string; contentType?: string; contentLength?: string } = {}
  ): void {
    const { status = 200, statusText = 'OK', contentType = 'text/plain', contentLength } = options
    const headers = new Map<string, string>([['content-type', contentType]])
    if (contentLength !== undefined) headers.set('content-length', contentLength)
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: status < 400,
      status,
      statusText,
      headers: { get: (key: string) => headers.get(key) ?? null },
      text: async () => body,
    })
  }

  function makeContext(overrides: Partial<ToolContext> = {}): ToolContext {
    return {
      toolUse: { name: 'web_fetch', toolUseId: 'test-id', input: {} },
      agent: { model: undefined } as unknown as LocalAgent,
      invocationState: {},
      cancelSignal: new AbortController().signal,
      interrupt: vi.fn() as ToolContext['interrupt'],
      ...overrides,
    }
  }

  describe('htmlToMarkdown', () => {
    it('strips script and style content', () => {
      const html = `
        <html><head><title>Hi</title>
        <style>body{color:red}</style>
        </head><body>
        <p>Hello world.</p>
        <script>alert('xss')</script>
        <p>After script.</p>
        </body></html>
      `
      const result = htmlToMarkdown(html)
      expect(result).toMatch(/# Hi/)
      expect(result).not.toMatch(/alert/)
      expect(result).not.toMatch(/color:red/)
      expect(result).toMatch(/Hello world\./)
      expect(result).toMatch(/After script\./)
    })

    it('strips data-URI image blobs', () => {
      const bigBlob = 'A'.repeat(10000)
      const result = htmlToMarkdown(`<p>text</p><img src="data:image/png;base64,${bigBlob}" alt="alt text">`)
      expect(result).not.toContain(bigBlob)
      expect(result).not.toMatch(/data:/)
      expect(result).toMatch(/alt text/)
    })

    it('preserves regular images and relative hrefs', () => {
      expect(htmlToMarkdown('<img src="https://example.com/pic.png" alt="pic">')).toContain(
        '![pic](https://example.com/pic.png)'
      )
      expect(htmlToMarkdown('<a href="/about">About</a>')).toContain('[About](/about)')
    })

    it('drops javascript: hrefs but keeps link text', () => {
      const result = htmlToMarkdown('<a href="javascript:alert(1)">click</a>')
      expect(result).not.toMatch(/javascript:/)
      expect(result).toMatch(/click/)
    })

    it('drops javascript: img srcs', () => {
      expect(htmlToMarkdown('<img src="javascript:alert(1)" alt="x">')).not.toMatch(/javascript:/)
    })

    it('preserves headings, lists, and links', () => {
      const html = `
        <h1>Title</h1>
        <p>Intro paragraph with a <a href="https://ex.com/x">link</a>.</p>
        <ul><li>one</li><li>two</li></ul>
        <ol><li>first</li><li>second</li></ol>
      `
      const result = htmlToMarkdown(html)
      expect(result).toMatch(/# Title/)
      expect(result).toContain('[link](https://ex.com/x)')
      expect(result).toMatch(/- +one/)
      expect(result).toMatch(/1\. +first/)
    })

    it('preserves code blocks', () => {
      const result = htmlToMarkdown('<pre><code>def f():\n    return 1</code></pre>')
      expect(result).toContain('```')
      expect(result).toContain('def f():')
    })

    it('survives malformed HTML', () => {
      const result = htmlToMarkdown('<p>ok <b>bold <em>and italic</p>')
      expect(result).toContain('ok')
      expect(result).toContain('bold')
    })

    it('preserves blockquote prefix on nested block content', () => {
      const result = htmlToMarkdown('<blockquote><p>quoted line one.</p><p>quoted line two.</p></blockquote>')
      const bqLines = result.split('\n').filter((line) => line.includes('quoted line'))
      expect(bqLines).toHaveLength(2)
      for (const line of bqLines) {
        expect(line.startsWith('> ')).toBe(true)
      }
    })

    it('void tag inside a dropped element does not swallow following content', () => {
      expect(htmlToMarkdown('<form><input></form><p>after</p>')).toContain('after')
    })

    it('embeds the page title as a top-level heading', () => {
      const result = htmlToMarkdown('<html><head><title>My Page</title></head><body><p>body</p></body></html>')
      expect(result).toMatch(/^# My Page\n/)
      expect(result).toContain('body')
    })

    it('produces no heading when there is no title', () => {
      expect(htmlToMarkdown('<p>no title here</p>')).not.toMatch(/^#/)
    })

    it('returns original input on conversion failure', () => {
      expect(htmlToMarkdown(null as unknown as string)).toBe(null)
    })
  })

  describe('makeWebFetch factory', () => {
    it('defaults to agentic mode', () => {
      expect(makeWebFetch().description).toBe(WEB_FETCH_DESCRIPTION_AGENTIC)
      expect(webFetch.name).toBe('web_fetch')
    })

    it('markdown mode uses markdown description', () => {
      expect(makeWebFetch({ mode: 'markdown' }).description).toBe(WEB_FETCH_DESCRIPTION_MARKDOWN)
    })

    it('custom name and description override defaults', () => {
      const t = makeWebFetch({ name: 'fetch_page', description: 'custom desc' })
      expect(t.name).toBe('fetch_page')
      expect(t.description).toBe('custom desc')
    })

    it('rejects non-positive maxBytes', () => {
      expect(() => makeWebFetch({ maxBytes: 0 })).toThrow(/maxBytes/)
      expect(() => makeWebFetch({ maxBytes: -1 })).toThrow(/maxBytes/)
    })

    it('rejects non-positive maxContentChars', () => {
      expect(() => makeWebFetch({ maxContentChars: 0 })).toThrow(/maxContentChars/)
      expect(() => makeWebFetch({ maxContentChars: -1 })).toThrow(/maxContentChars/)
    })

    it('exports correct default constants', () => {
      expect(DEFAULT_MAX_BYTES).toBe(5 * 1024 * 1024)
      expect(DEFAULT_MAX_CONTENT_CHARS).toBe(50_000)
    })
  })

  describe('markdown mode', () => {
    it('html response is converted to markdown', async () => {
      mockFetch('<html><head><title>T</title></head><body><h1>Hi</h1></body></html>', {
        contentType: 'text/html; charset=utf-8',
      })
      const result = await makeWebFetch({ mode: 'markdown' }).invoke({ url: 'https://example.com/' })
      expect(result).toContain('# T')
      expect(result).toContain('# Hi')
    })

    it('xml content type is also converted to markdown', async () => {
      mockFetch('<html><body><p>xhtml</p></body></html>', { contentType: 'application/xhtml+xml' })
      const result = await makeWebFetch({ mode: 'markdown' }).invoke({ url: 'https://example.com/page.xhtml' })
      expect(result).toContain('xhtml')
    })

    it('non-html response is returned as-is', async () => {
      mockFetch('plain text response', { contentType: 'text/plain' })
      const result = await makeWebFetch({ mode: 'markdown' }).invoke({ url: 'https://example.com/robots.txt' })
      expect(result).toBe('plain text response')
    })

    it('content is truncated at maxContentChars', async () => {
      mockFetch('x'.repeat(200), { contentType: 'text/plain' })
      const result = await makeWebFetch({ mode: 'markdown', maxContentChars: 50 }).invoke({
        url: 'https://example.com/',
      })
      expect(result).toContain('x'.repeat(50))
      expect(result).not.toContain('x'.repeat(51))
      expect(result).toContain('[content truncated]')
    })

    it('rejects 4xx status', async () => {
      mockFetch('Not Found', { status: 404, statusText: 'Not Found' })
      await expect(makeWebFetch({ mode: 'markdown' }).invoke({ url: 'https://example.com/missing' })).rejects.toThrow(
        'HTTP 404'
      )
    })

    it('rejects response exceeding content-length header', async () => {
      mockFetch('short', { contentType: 'text/plain', contentLength: String(DEFAULT_MAX_BYTES + 1) })
      await expect(makeWebFetch({ mode: 'markdown' }).invoke({ url: 'https://example.com/' })).rejects.toThrow(
        /max_bytes/
      )
    })

    it('wraps network errors', async () => {
      globalThis.fetch = vi.fn().mockRejectedValue(new Error('connection refused'))
      await expect(makeWebFetch({ mode: 'markdown' }).invoke({ url: 'https://example.com/' })).rejects.toThrow(
        /fetch failed/
      )
    })

    it('throws on abort signal', async () => {
      globalThis.fetch = vi.fn().mockRejectedValue(Object.assign(new Error('aborted'), { name: 'AbortError' }))
      await expect(makeWebFetch({ mode: 'markdown' }).invoke({ url: 'https://example.com/' })).rejects.toThrow(
        'Web fetch tool request cancelled'
      )
    })

    it('sends correct user-agent header and passes cancel signal', async () => {
      mockFetch('ok')
      const controller = new AbortController()
      await makeWebFetch({ mode: 'markdown' }).invoke(
        { url: 'https://example.com/' },
        makeContext({ cancelSignal: controller.signal })
      )
      expect(globalThis.fetch).toHaveBeenCalledWith(
        'https://example.com/',
        expect.objectContaining({
          headers: expect.objectContaining({ 'User-Agent': 'strands-agents-web-fetch/1.0' }),
          signal: controller.signal,
        })
      )
    })
  })

  describe('agentic mode', () => {
    it('requires a non-empty prompt', async () => {
      mockFetch('<p>content</p>', { contentType: 'text/html' })
      await expect(
        makeWebFetch({ mode: 'agentic', model: {} as agentModule.Agent['model'] }).invoke({
          url: 'https://example.com/',
          prompt: '   ',
        })
      ).rejects.toThrow('agentic mode requires a non-empty prompt')
    })

    it('requires a model when no context agent', async () => {
      mockFetch('<p>content</p>', { contentType: 'text/html' })
      await expect(
        makeWebFetch({ mode: 'agentic' }).invoke({ url: 'https://example.com/', prompt: 'Summarize' })
      ).rejects.toThrow('agentic mode requires a model')
    })

    it('uses factory model when provided', async () => {
      mockFetch('<p>page content</p>', { contentType: 'text/html' })
      const fakeModel = {} as agentModule.Agent['model']
      const receivedModel: unknown[] = []
      vi.spyOn(agentModule, 'Agent').mockImplementation(function (config) {
        receivedModel.push(config?.model)
        return { invoke: vi.fn().mockResolvedValue({ toString: () => 'the answer' }) } as unknown as agentModule.Agent
      })
      await makeWebFetch({ mode: 'agentic', model: fakeModel }).invoke({
        url: 'https://example.com/',
        prompt: 'What is this?',
      })
      expect(receivedModel[0]).toBe(fakeModel)
    })

    it('falls back to host agent model', async () => {
      mockFetch('<p>page content</p>', { contentType: 'text/html' })
      const hostModel = {} as agentModule.Agent['model']
      const receivedModel: unknown[] = []
      vi.spyOn(agentModule, 'Agent').mockImplementation(function (config) {
        receivedModel.push(config?.model)
        return { invoke: vi.fn().mockResolvedValue({ toString: () => 'host answer' }) } as unknown as agentModule.Agent
      })
      await makeWebFetch({ mode: 'agentic' }).invoke(
        { url: 'https://example.com/', prompt: 'Summarize' },
        makeContext({ agent: { model: hostModel } as unknown as LocalAgent })
      )
      expect(receivedModel[0]).toBe(hostModel)
    })

    it('passes prompt and page content to analyst', async () => {
      mockFetch('<html><body><p>page content</p></body></html>', { contentType: 'text/html' })
      const receivedPrompts: string[] = []
      vi.spyOn(agentModule, 'Agent').mockImplementation(function () {
        return {
          invoke: vi.fn().mockImplementation(async (prompt: string) => {
            receivedPrompts.push(prompt)
            return { toString: () => 'the answer' }
          }),
        } as unknown as agentModule.Agent
      })
      const result = await makeWebFetch({ mode: 'agentic', model: {} as agentModule.Agent['model'] }).invoke({
        url: 'https://example.com/',
        prompt: 'What is this about?',
      })
      expect(result).toBe('the answer')
      expect(receivedPrompts[0]).toContain('What is this about?')
      expect(receivedPrompts[0]).toContain('page content')
    })

    it('truncates content before passing to analyst', async () => {
      mockFetch('x'.repeat(200), { contentType: 'text/plain' })
      const receivedPrompts: string[] = []
      vi.spyOn(agentModule, 'Agent').mockImplementation(function () {
        return {
          invoke: vi.fn().mockImplementation(async (prompt: string) => {
            receivedPrompts.push(prompt)
            return { toString: () => 'answer' }
          }),
        } as unknown as agentModule.Agent
      })
      await makeWebFetch({ mode: 'agentic', model: {} as agentModule.Agent['model'], maxContentChars: 50 }).invoke({
        url: 'https://example.com/',
        prompt: 'Summarize',
      })
      expect(receivedPrompts[0]).toContain('x'.repeat(50))
      expect(receivedPrompts[0]).not.toContain('x'.repeat(51))
      expect(receivedPrompts[0]).toContain('[content truncated]')
    })

    it('wraps analyst error with url context', async () => {
      mockFetch('<p>content</p>', { contentType: 'text/html' })
      vi.spyOn(agentModule, 'Agent').mockImplementation(function () {
        return {
          invoke: vi.fn().mockRejectedValue(new Error('analyst boom')),
        } as unknown as agentModule.Agent
      })
      await expect(
        makeWebFetch({ mode: 'agentic', model: {} as agentModule.Agent['model'] }).invoke({
          url: 'https://example.com/',
          prompt: 'Summarize',
        })
      ).rejects.toThrow(/web fetch analyst failed/)
    })
  })
})
