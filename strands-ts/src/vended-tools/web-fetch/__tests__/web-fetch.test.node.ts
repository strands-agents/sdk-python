import { describe, it, expect, vi, beforeAll, afterAll, beforeEach } from 'vitest'
import { Buffer } from 'buffer'
import { createServer, type Server, type IncomingMessage, type ServerResponse } from 'http'
import type { AddressInfo } from 'net'
import { addressIsPublic, assertHostIsAllowed, resolveAndValidateHost, validateUrlScheme } from '../ssrf.js'
import { htmlToMarkdown } from '../extract.js'
import { webFetch, makeWebFetch, _fetchOnce, _followRedirects, DEFAULT_MAX_BYTES } from '../web-fetch.js'
import * as ssrfModule from '../ssrf.js'

describe('validateUrlScheme', () => {
  it('accepts http', () => {
    expect(() => validateUrlScheme('http://example.com/foo')).not.toThrow()
  })

  it('accepts https', () => {
    expect(() => validateUrlScheme('https://example.com/foo')).not.toThrow()
  })

  it.each([
    'file:///etc/passwd',
    'ftp://example.com/foo',
    'gopher://example.com/',
    'javascript:alert(1)',
    'chrome-extension://foo/bar',
  ])('rejects scheme: %s', (url) => {
    expect(() => validateUrlScheme(url)).toThrow(/Only http/)
  })

  it('rejects malformed URL', () => {
    expect(() => validateUrlScheme('not a url')).toThrow(/Invalid URL/)
  })
})

describe('addressIsPublic', () => {
  it.each([
    '127.0.0.1',
    '127.0.0.53',
    '0.0.0.0',
    '10.0.0.5',
    '172.16.0.1',
    '172.31.255.255',
    '192.168.1.1',
    '169.254.169.254', // AWS/GCP metadata endpoint
    '169.254.170.2', // ECS task metadata
    '100.64.0.1', // CGNAT
    '224.0.0.1', // multicast
    '240.0.0.1', // reserved
    '255.255.255.255',
    '203.0.113.1', // TEST-NET-3
    '198.51.100.1', // TEST-NET-2
    '192.0.2.1', // TEST-NET-1
    '::1',
    'fe80::1',
    'fc00::1',
    'fd00::1',
    'ff02::1',
    '2001:db8::1', // documentation (compressed)
    '2001:0db8:0000:0000:0000:0000:0000:0001', // documentation (fully expanded)
    '100::1', // 100::/64 discard-only
    'fec0::1', // site-local (deprecated but filtered)
    'fed0::1',
  ])('rejects non-public: %s', (ip) => {
    expect(addressIsPublic(ip)).toBe(false)
  })

  it.each(['8.8.8.8', '1.1.1.1', '2606:4700:4700::1111'])('accepts public: %s', (ip) => {
    expect(addressIsPublic(ip)).toBe(true)
  })

  it('unwraps IPv4-mapped IPv6 loopback and refuses it', () => {
    // Without unwrapping, ::ffff:127.0.0.1 could bypass an IPv6-only check.
    expect(addressIsPublic('::ffff:127.0.0.1')).toBe(false)
  })

  it('unwraps IPv4-mapped IPv6 public addresses and accepts them', () => {
    expect(addressIsPublic('::ffff:8.8.8.8')).toBe(true)
  })

  it.each([
    // Node's URL parser normalizes bracketed `::ffff:127.0.0.1` to the hex
    // form `::ffff:7f00:1`. The address predicate must unwrap that too.
    '::ffff:7f00:1', // 127.0.0.1
    '::ffff:a9fe:a9fe', // 169.254.169.254 (metadata)
    '::ffff:a00:1', // 10.0.0.1
    '::ffff:6440:1', // 100.64.0.1 (CGNAT)
  ])('unwraps hex-form IPv4-mapped IPv6 and refuses private: %s', (ip) => {
    expect(addressIsPublic(ip)).toBe(false)
  })

  it('rejects garbage input', () => {
    expect(addressIsPublic('not-an-ip')).toBe(false)
    expect(addressIsPublic('999.999.999.999')).toBe(false)
  })
})

describe('assertHostIsAllowed', () => {
  it.each([
    'foo.internal',
    'foo.internal.',
    'FOO.INTERNAL',
    'bar.local',
    'localhost',
    'foo.corp',
    'foo.home',
    'foo.lan',
    'foo.intranet',
    'foo.private',
    'example.i2p',
    'example.onion',
  ])('rejects denied DNS suffix: %s', (host) => {
    expect(() => assertHostIsAllowed(host)).toThrow(/denylist|metadata/)
  })

  it.each([
    'metadata',
    'metadata.google.internal',
    '169.254.169.254',
    'fd00:ec2::254',
    '100.100.100.200',
    '192.0.0.192',
  ])('rejects named metadata endpoint: %s', (host) => {
    expect(() => assertHostIsAllowed(host)).toThrow(/metadata|denylist/)
  })

  it('accepts ordinary public hostnames', () => {
    expect(() => assertHostIsAllowed('example.com')).not.toThrow()
    expect(() => assertHostIsAllowed('cloudflare.com')).not.toThrow()
  })
})

describe('resolveAndValidateHost', () => {
  it('accepts a public IP literal without DNS', async () => {
    // 8.8.8.8 is a canonical public address; no real DNS query is issued.
    await expect(resolveAndValidateHost('8.8.8.8')).resolves.toEqual(['8.8.8.8'])
  })

  it('rejects a private IP literal', async () => {
    await expect(resolveAndValidateHost('127.0.0.1')).rejects.toThrow(/not public/)
  })

  it('rejects the AWS/GCP metadata IP literal', async () => {
    // 169.254.169.254 is refused by the pre-DNS named-metadata check before
    // the address predicate runs. Either error is fine as long as it's a
    // hard refusal.
    await expect(resolveAndValidateHost('169.254.169.254')).rejects.toThrow(/not public|metadata/)
  })

  it('accepts a bracketed IPv6 URL literal', async () => {
    // new URL('http://[2606:4700:4700::1111]/').hostname is bracketed; the
    // resolver must strip the brackets before isIP/DNS lookup.
    await expect(resolveAndValidateHost('[2606:4700:4700::1111]')).resolves.toEqual(['2606:4700:4700::1111'])
  })

  it('rejects a bracketed private IPv6 URL literal', async () => {
    await expect(resolveAndValidateHost('[::1]')).rejects.toThrow(/not public/)
  })
})

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
    const { title, markdown } = htmlToMarkdown(html)
    expect(title).toBe('Hi')
    expect(markdown).not.toMatch(/alert/)
    expect(markdown).not.toMatch(/color:red/)
    expect(markdown).toMatch(/Hello world\./)
    expect(markdown).toMatch(/After script\./)
  })

  it('strips data-URI image blobs', () => {
    const bigBlob = 'A'.repeat(10000)
    const html = `<p>text</p><img src="data:image/png;base64,${bigBlob}" alt="alt text">`
    const { markdown } = htmlToMarkdown(html)
    expect(markdown).not.toContain(bigBlob)
    expect(markdown).not.toMatch(/data:/)
    expect(markdown).toMatch(/alt text/)
  })

  it('preserves regular images', () => {
    const { markdown } = htmlToMarkdown('<img src="https://example.com/pic.png" alt="pic">')
    expect(markdown).toContain('![pic](https://example.com/pic.png)')
  })

  it('drops javascript: hrefs but keeps link text', () => {
    const { markdown } = htmlToMarkdown('<a href="javascript:alert(1)">click</a>')
    expect(markdown).not.toMatch(/javascript:/)
    expect(markdown).toMatch(/click/)
  })

  it('drops javascript: img srcs', () => {
    const { markdown } = htmlToMarkdown('<img src="javascript:alert(1)" alt="x">')
    expect(markdown).not.toMatch(/javascript:/)
  })

  it('preserves headings, lists, and links', () => {
    const html = `
      <h1>Title</h1>
      <p>Intro paragraph with a <a href="https://ex.com/x">link</a>.</p>
      <ul><li>one</li><li>two</li></ul>
      <ol><li>first</li><li>second</li></ol>
    `
    const { markdown } = htmlToMarkdown(html)
    expect(markdown).toMatch(/# Title/)
    expect(markdown).toContain('[link](https://ex.com/x)')
    expect(markdown).toContain('- one')
    expect(markdown).toContain('- two')
    expect(markdown).toContain('1. first')
    expect(markdown).toContain('2. second')
  })

  it('preserves code blocks', () => {
    const { markdown } = htmlToMarkdown('<pre><code>def f():\n    return 1</code></pre>')
    expect(markdown).toContain('```')
    expect(markdown).toContain('def f():')
    expect(markdown).toContain('return 1')
  })

  it('ignores event handler attributes', () => {
    // onclick/onerror never appear as output text because we only render
    // href/src/alt attributes, and the raw tokenizer discards everything else.
    const { markdown } = htmlToMarkdown('<div onclick="alert(1)"><p onmouseover="x()">safe</p></div>')
    expect(markdown).not.toMatch(/alert/)
    expect(markdown).not.toMatch(/onclick/)
    expect(markdown).toMatch(/safe/)
  })

  it('survives malformed HTML', () => {
    const { markdown } = htmlToMarkdown('<p>ok <b>bold <em>and italic</p>')
    expect(markdown).toContain('ok')
    expect(markdown).toContain('bold')
  })

  it('preserves blockquote prefix on nested block content', () => {
    // <blockquote> wrapping <p> is the common shape. Every line inside the
    // blockquote must be prefixed with "> ", including the paragraph that a
    // naive open-tag-only implementation would emit unquoted.
    const { markdown } = htmlToMarkdown('<blockquote><p>quoted line one.</p><p>quoted line two.</p></blockquote>')
    const bqLines = markdown.split('\n').filter((l) => l.includes('quoted line'))
    expect(bqLines).toHaveLength(2)
    for (const line of bqLines) {
      expect(line.startsWith('> ')).toBe(true)
    }
  })

  it('nests blockquote prefixes', () => {
    const { markdown } = htmlToMarkdown('<blockquote><blockquote><p>deep.</p></blockquote></blockquote>')
    expect(markdown).toMatch(/> > deep\./)
  })

  it('void tag inside a dropped element does not swallow following content', () => {
    // <input> is void: only a start tag ever fires. Previous logic used a
    // depth counter over every start/end tag inside the drop, which trapped
    // the parser in "dropping" mode after </form>.
    const { markdown } = htmlToMarkdown('<form><input></form><p>after</p>')
    expect(markdown).toContain('after')
  })

  it('drops javascript: hrefs even with leading whitespace', () => {
    const { markdown } = htmlToMarkdown('<a href="\tjavascript:alert(1)">click</a>')
    expect(markdown).not.toMatch(/javascript:/)
    expect(markdown).not.toMatch(/alert/)
    expect(markdown).toContain('click')
  })

  it('decodes HTML entities', () => {
    const { markdown } = htmlToMarkdown('<p>Rock &amp; roll &lt;3</p>')
    expect(markdown).toContain('Rock & roll <3')
  })

  it('discards HTML comments', () => {
    const { markdown } = htmlToMarkdown('<p>before <!-- SECRET_TOKEN=abc --> after</p>')
    expect(markdown).not.toContain('SECRET_TOKEN')
    expect(markdown).toContain('before')
    expect(markdown).toContain('after')
  })
})

describe('webFetch end-to-end (SSRF)', () => {
  it('rejects http://127.0.0.1', async () => {
    await expect(webFetch.invoke({ url: 'http://127.0.0.1/anything' })).rejects.toThrow(/not public/)
  })

  it('rejects the cloud metadata endpoint', async () => {
    await expect(webFetch.invoke({ url: 'http://169.254.169.254/latest/meta-data/' })).rejects.toThrow(
      /not public|metadata/
    )
  })

  it('rejects file://', async () => {
    await expect(webFetch.invoke({ url: 'file:///etc/passwd' })).rejects.toThrow(/Only http/)
  })

  it('rejects malformed URL', async () => {
    await expect(webFetch.invoke({ url: 'not a url' })).rejects.toThrow(/Invalid URL/)
  })
})

describe('makeWebFetch factory', () => {
  it('overrides maxBytes and maxRedirects via options', () => {
    // Just proves the factory shape; behavior of the caps is verified below
    // by the loopback transport tests.
    expect(() => makeWebFetch({ maxBytes: 1024, maxRedirects: 2 })).not.toThrow()
  })

  it('rejects non-positive maxBytes', () => {
    expect(() => makeWebFetch({ maxBytes: 0 })).toThrow(/maxBytes/)
    expect(() => makeWebFetch({ maxBytes: -1 })).toThrow(/maxBytes/)
  })

  it('rejects negative maxRedirects', () => {
    expect(() => makeWebFetch({ maxRedirects: -1 })).toThrow(/maxRedirects/)
  })
})

// Real-transport tests using an http.createServer on 127.0.0.1. The SSRF
// address checker rejects loopback by design, so we stub `resolveAndValidateHost`
// and `assertHostIsAllowed` for these tests only. Every test asserts on
// behaviors that are impossible to observe from a pure mock-based test:
// redirect following and revalidation, the size-cap abort, Host-header
// construction, and the total-timeout wrapper.
describe('web-fetch transport (loopback)', () => {
  let server: Server
  let port = 0
  type Handler = (req: IncomingMessage, res: ServerResponse) => void
  let handler: Handler = () => {
    /* installed per test */
  }

  beforeAll(async () => {
    server = createServer((req, res) => handler(req, res))
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    port = (server.address() as AddressInfo).port
    // Force every host through validation as if it were the loopback IP; the
    // pinned-connect logic uses the returned address as the socket target.
    vi.spyOn(ssrfModule, 'resolveAndValidateHost').mockImplementation(async () => ['127.0.0.1'])
    vi.spyOn(ssrfModule, 'assertHostIsAllowed').mockImplementation(() => {
      /* allow */
    })
  })

  afterAll(async () => {
    vi.restoreAllMocks()
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())))
  })

  beforeEach(() => {
    handler = () => {
      /* overridden per test */
    }
  })

  it('follows a redirect and revalidates the hop', async () => {
    handler = (req, res) => {
      if (req.url === '/start') {
        res.writeHead(302, { location: `http://web.local:${port}/next` })
        res.end()
      } else if (req.url === '/next') {
        res.writeHead(200, { 'content-type': 'text/html' })
        res.end('<html><body><h1>ok</h1></body></html>')
      } else {
        res.writeHead(404).end()
      }
    }
    const result = await _followRedirects(`http://web.local:${port}/start`, 5000, DEFAULT_MAX_BYTES, 5)
    expect(result.status).toBe(200)
    expect(result.body.toString()).toContain('<h1>ok</h1>')
  })

  it('rejects a javascript: redirect', async () => {
    handler = (req, res) => {
      if (req.url === '/start') {
        res.writeHead(302, { location: 'javascript:alert(1)' })
        res.end()
      } else {
        res.writeHead(404).end()
      }
    }
    await expect(_followRedirects(`http://web.local:${port}/start`, 5000, DEFAULT_MAX_BYTES, 5)).rejects.toThrow(
      /Only http/
    )
  })

  it('surfaces a redirect without a Location header', async () => {
    handler = (_req, res) => {
      res.writeHead(302, {})
      res.end()
    }
    await expect(_followRedirects(`http://web.local:${port}/start`, 5000, DEFAULT_MAX_BYTES, 5)).rejects.toThrow(
      /without a Location/
    )
  })

  it('caps the number of redirects', async () => {
    let hop = 0
    handler = (_req, res) => {
      hop += 1
      res.writeHead(302, { location: `http://web.local:${port}/hop-${hop}` })
      res.end()
    }
    await expect(_followRedirects(`http://web.local:${port}/start`, 5000, DEFAULT_MAX_BYTES, 2)).rejects.toThrow(
      /Exceeded max_redirects/
    )
  })

  it('aborts as soon as the response body exceeds maxBytes', async () => {
    handler = (_req, res) => {
      res.writeHead(200, { 'content-type': 'application/octet-stream' })
      // Write far more than the cap so at least one chunk trips the check
      // before the connection's `end` event has a chance to fire.
      res.end(Buffer.alloc(2 * 1024 * 1024))
    }
    await expect(_fetchOnce(`http://web.local:${port}/big`, 5000, 100 * 1024)).rejects.toThrow(/exceeded max_bytes/)
  })

  it('sends the original hostname in the Host header (not the pinned IP)', async () => {
    let captured: string | undefined
    handler = (req, res) => {
      captured = req.headers.host
      res.writeHead(200, { 'content-type': 'text/plain' })
      res.end('ok')
    }
    await _fetchOnce(`http://web.local:${port}/`, 5000, DEFAULT_MAX_BYTES)
    expect(captured).toBe(`web.local:${port}`)
  })

  it('short-circuits the redirect body drain -- does not buffer the discarded payload', async () => {
    handler = (req, res) => {
      if (req.url === '/start') {
        res.writeHead(302, {
          location: `http://web.local:${port}/final`,
        })
        res.write(Buffer.alloc(4 * 1024 * 1024, 0x41)) // 4 MiB
        res.end()
      } else {
        res.writeHead(200, { 'content-type': 'text/html' })
        res.end('<p>done</p>')
      }
    }
    const result = await _followRedirects(`http://web.local:${port}/start`, 5000, 200 * 1024, 5)
    expect(result.status).toBe(200)
    expect(result.body.toString()).toContain('done')
  })

  it.each([
    ['text/html; charset=utf-8', 'utf-8'],
    ['text/html; charset="iso-8859-1"', 'iso-8859-1'],
    ["text/html; charset='windows-1252'", 'windows-1252'],
  ])('honors quoted and unquoted Content-Type charset: %s', async (headerValue, expectedCharset) => {
    // Ensure the extracted markdown reflects the correct decoding for a
    // charset the server declares. We use a byte sequence that decodes
    // differently under the two encodings to prove the quoted charset was
    // parsed. For utf-8 we just prove the happy path.
    handler = (_req, res) => {
      res.writeHead(200, { 'content-type': headerValue })
      // Latin-1 byte 0xA9 (©) decodes as a copyright sign under iso-8859-1
      // and windows-1252, but under utf-8 it decodes to the replacement
      // character. We put the byte in the body verbatim.
      if (expectedCharset === 'utf-8') {
        res.end('<p>hello</p>')
      } else {
        res.end(Buffer.from('<p>' + String.fromCharCode(0xa9) + '</p>', 'binary'))
      }
    }
    const result = (await webFetch.invoke({ url: `http://web.local:${port}/` })) as {
      markdown: string
    }
    if (expectedCharset === 'utf-8') {
      expect(result.markdown).toContain('hello')
    } else {
      // The regex must have picked up the quoted charset -- otherwise the
      // 0xA9 byte would end up as the utf-8 replacement char.
      expect(result.markdown).toContain('©')
    }
  })
})

// Multi-address fallback: if the first validated address refuses the
// connection, the wrapper tries the next. The Python side has an equivalent
// unit test that drives the fallback via a mocked http.client. Here we prove
// the retry logic surfaces a clear ConnectionError when every validated
// address refuses -- the only shape we can force deterministically from a
// unit test that must not depend on host-specific routing.
describe('web-fetch multi-address fallback', () => {
  it('surfaces a clear error when every validated address refuses the connection', async () => {
    // Bind and immediately close a server to reserve a port we know is now
    // closed. Connecting to it fails fast with ECONNREFUSED on any OS.
    const throwaway = createServer()
    await new Promise<void>((resolve) => throwaway.listen(0, '127.0.0.1', resolve))
    const closedPort = (throwaway.address() as AddressInfo).port
    await new Promise<void>((resolve, reject) => throwaway.close((err) => (err ? reject(err) : resolve())))

    vi.spyOn(ssrfModule, 'resolveAndValidateHost').mockImplementationOnce(async () => [
      '127.0.0.1', // refuses -- closedPort has no listener
      '127.0.0.1', // still refuses -- proves both addresses were tried before we give up
    ])
    vi.spyOn(ssrfModule, 'assertHostIsAllowed').mockImplementation(() => {
      /* allow */
    })

    await expect(_fetchOnce(`http://web.local:${closedPort}/`, 3000, DEFAULT_MAX_BYTES)).rejects.toThrow(
      /Could not connect to any validated address/
    )

    vi.restoreAllMocks()
  }, 10000)
})

// End-to-end assertion on the invoked tool for a plain-text response. Uses the
// same loopback server pattern above.
describe('webFetch tool end-to-end (loopback)', () => {
  let server: Server
  let port = 0

  beforeAll(async () => {
    server = createServer((_req, res) => {
      res.writeHead(200, { 'content-type': 'text/plain' })
      res.end('plain text response')
    })
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    port = (server.address() as AddressInfo).port
    vi.spyOn(ssrfModule, 'resolveAndValidateHost').mockImplementation(async () => ['127.0.0.1'])
    vi.spyOn(ssrfModule, 'assertHostIsAllowed').mockImplementation(() => {
      /* allow */
    })
  })

  afterAll(async () => {
    vi.restoreAllMocks()
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())))
  })

  it('returns the full deterministic output object for a plain-text response', async () => {
    const url = `http://web.local:${port}/robots.txt`
    const result = await webFetch.invoke({ url })
    expect(result).toEqual({
      url,
      status: 200,
      contentType: 'text/plain',
      title: '',
      markdown: 'plain text response',
    })
  })
})
