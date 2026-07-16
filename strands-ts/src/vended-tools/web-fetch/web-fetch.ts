import { z } from 'zod'
import { Buffer } from 'buffer'
import { request as httpRequest } from 'http'
import { request as httpsRequest } from 'https'

import { tool } from '../../tools/tool-factory.js'
import { htmlToMarkdown } from './extract.js'
import { resolveAndValidateHost, stripIpv6Brackets, validateUrlScheme } from './ssrf.js'

const DEFAULT_TIMEOUT_SECONDS = 30
export const DEFAULT_MAX_BYTES = 5 * 1024 * 1024 // 5 MiB
export const DEFAULT_MAX_REDIRECTS = 5
const USER_AGENT = 'strands-agents-web-fetch/1.0'
// On a 3xx we only need the `Location` header; buffering the discarded body up
// to `maxBytes` is a waste of memory and an easy DoS vector. Cap at 4 KiB and
// tear the request down once we've seen the redirect headers.
const REDIRECT_BODY_DRAIN_CAP = 4096
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308])

export const webFetchInputSchema = z.object({
  url: z.string().describe('URL to fetch. Must be http:// or https://.'),
  timeout: z.number().positive().optional().describe('Total request timeout in seconds (default: 30).'),
})

interface RawResponse {
  status: number
  headers: Record<string, string>
  body: Buffer
  effectiveUrl: string
}

/**
 * Perform one HTTP(S) request with SSRF defenses, no redirect handling.
 *
 * Resolves the hostname, checks that every resolved address is public, then
 * connects to a pinned already-validated address so a DNS rebinder cannot
 * substitute a private address between validation and connect. Aborts as
 * soon as the response body exceeds `maxBytes`.
 *
 * Exported for testing against a loopback server; not part of the public API.
 */
export async function _fetchOnce(
  url: string,
  timeoutMs: number,
  maxBytes: number,
  signal?: AbortSignal
): Promise<RawResponse> {
  const parsed = validateUrlScheme(url)
  // parsed.hostname keeps [ ] around IPv6 literals; strip them for the
  // resolver and for the pinned-connect host below.
  const bareHost = stripIpv6Brackets(parsed.hostname)
  const addresses = await resolveAndValidateHost(bareHost)
  const isHttps = parsed.protocol === 'https:'
  const port = parsed.port ? Number(parsed.port) : isHttps ? 443 : 80

  // Try each validated address in turn. Every address has already passed the
  // SSRF check, so falling through does not weaken the security posture.
  // Callers on hosts where DNS returns an AAAA record first but the runtime
  // has no IPv6 route (or vice versa) still succeed on the second try.
  const retryable = new Set(['ECONNREFUSED', 'EHOSTUNREACH', 'ENETUNREACH', 'EADDRNOTAVAIL'])
  let lastError: (Error & { code?: string }) | undefined
  for (const pinnedIp of addresses) {
    try {
      return await fetchOncePinned(url, parsed, bareHost, pinnedIp, isHttps, port, timeoutMs, maxBytes, signal)
    } catch (err) {
      const e = err as Error & { code?: string }
      if (e && e.name !== 'AbortError' && e.code !== undefined && retryable.has(e.code)) {
        lastError = e
        continue
      }
      throw err
    }
  }
  throw new Error(
    `Could not connect to any validated address for host ${JSON.stringify(bareHost)} (tried ${addresses.length}): ${lastError?.message ?? 'unknown'}`,
    { cause: lastError }
  )
}

function fetchOncePinned(
  url: string,
  parsed: URL,
  bareHost: string,
  pinnedIp: string,
  isHttps: boolean,
  port: number,
  timeoutMs: number,
  maxBytes: number,
  signal?: AbortSignal
): Promise<RawResponse> {
  return new Promise<RawResponse>((resolve, reject) => {
    const abortReason = (): Error => {
      const reason = signal?.reason
      if (reason instanceof Error) return reason
      return new DOMException('Request aborted', 'AbortError')
    }
    // Reject immediately if the signal was already fired -- no request to make.
    if (signal?.aborted) {
      reject(abortReason())
      return
    }

    const requestFn = isHttps ? httpsRequest : httpRequest
    const req = requestFn({
      // Connect to the validated IP; the Host header + servername keep TLS
      // and vhost routing pointed at the original hostname.
      host: pinnedIp,
      port,
      method: 'GET',
      path: `${parsed.pathname || '/'}${parsed.search || ''}`,
      headers: {
        Host: parsed.host,
        'User-Agent': USER_AGENT,
        Accept: 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
        'Accept-Encoding': 'identity',
        Connection: 'close',
      },
      timeout: timeoutMs,
      // For HTTPS, override the certificate check hostname so verification
      // uses the public hostname (not the pinned IP literal). `servername`
      // wants the bare form -- no [ ] around IPv6 literals.
      ...(isHttps ? { servername: bareHost } : {}),
    })

    // Attach the abort listener AFTER req exists so the handler can never
    // reference a variable in its temporal dead zone. Re-check `aborted`
    // after attaching to close the window where the signal could have fired
    // between the initial check and addEventListener returning.
    const abortHandler = (): void => {
      req.destroy(abortReason())
    }
    if (signal) {
      signal.addEventListener('abort', abortHandler, { once: true })
      if (signal.aborted) {
        signal.removeEventListener('abort', abortHandler)
        req.destroy(abortReason())
        reject(abortReason())
        return
      }
    }

    const chunks: Buffer[] = []
    let total = 0

    const cleanupSignal = (): void => {
      if (signal) signal.removeEventListener('abort', abortHandler)
    }

    req.on('response', (res) => {
      // On a redirect, we only need the Location header -- resolve immediately
      // with an empty body and tear the connection down so a hostile server
      // can't force us to buffer megabytes of garbage on every 3xx hop.
      const status = res.statusCode ?? 0
      if (REDIRECT_STATUSES.has(status)) {
        cleanupSignal()
        const headers: Record<string, string> = {}
        for (const [k, v] of Object.entries(res.headers)) {
          if (Array.isArray(v)) headers[k.toLowerCase()] = v.join(', ')
          else if (typeof v === 'string') headers[k.toLowerCase()] = v
        }
        resolve({
          status,
          headers,
          body: Buffer.alloc(0),
          effectiveUrl: url,
        })
        // After we've resolved, still drain up to a small cap so the socket
        // can close cleanly, but discard the bytes without accumulating them.
        let drained = 0
        res.on('data', (chunk: Buffer) => {
          drained += chunk.length
          if (drained > REDIRECT_BODY_DRAIN_CAP) req.destroy()
        })
        res.on('error', () => {
          /* already resolved; ignore */
        })
        return
      }

      res.on('data', (chunk: Buffer) => {
        total += chunk.length
        if (total > maxBytes) {
          req.destroy(new Error(`Response body exceeded max_bytes=${maxBytes}. Refusing to buffer more.`))
          return
        }
        chunks.push(chunk)
      })
      res.on('end', () => {
        cleanupSignal()
        const headers: Record<string, string> = {}
        for (const [k, v] of Object.entries(res.headers)) {
          if (Array.isArray(v)) headers[k.toLowerCase()] = v.join(', ')
          else if (typeof v === 'string') headers[k.toLowerCase()] = v
        }
        resolve({
          status: res.statusCode ?? 0,
          headers,
          body: Buffer.concat(chunks),
          effectiveUrl: url,
        })
      })
      res.on('error', (err) => {
        cleanupSignal()
        reject(err)
      })
    })
    req.on('timeout', () => {
      req.destroy(new DOMException(`Request timed out after ${timeoutMs}ms`, 'AbortError'))
    })
    req.on('error', (err) => {
      cleanupSignal()
      reject(err)
    })
    req.end()
  })
}

// Exported for testing; not part of the public API.
export async function _followRedirects(
  url: string,
  timeoutMs: number,
  maxBytes: number,
  maxRedirects: number,
  signal?: AbortSignal
): Promise<RawResponse> {
  let current = url
  const seen = new Set<string>()
  for (let hop = 0; hop <= maxRedirects; hop += 1) {
    if (seen.has(current)) {
      throw new Error(`Redirect loop detected involving ${JSON.stringify(current)}`)
    }
    seen.add(current)

    const res = await _fetchOnce(current, timeoutMs, maxBytes, signal)
    if (REDIRECT_STATUSES.has(res.status)) {
      const location = res.headers['location']
      if (!location) {
        throw new Error(`Redirect ${res.status} from ${JSON.stringify(current)} without a Location header`)
      }
      const next = new URL(location, current).toString()
      // Re-validate the scheme immediately -- refuse javascript:/file:/etc.
      validateUrlScheme(next)
      current = next
      continue
    }
    return res
  }
  throw new Error(`Exceeded max_redirects=${maxRedirects} following ${JSON.stringify(url)}`)
}

function decodeBody(body: Buffer, contentType: string): string {
  // Accept quoted or unquoted charset values -- e.g. `charset=utf-8`,
  // `charset="iso-8859-1"`, or `charset='windows-1252'`. Trailing quote is
  // stripped in the same pass.
  const match = contentType.match(/charset=(?:"([^"]+)"|'([^']+)'|([^;\s]+))/i)
  const charset = (match?.[1] ?? match?.[2] ?? match?.[3] ?? 'utf-8').toLowerCase()
  try {
    return new TextDecoder(charset, { fatal: false }).decode(body)
  } catch {
    return new TextDecoder('utf-8', { fatal: false }).decode(body)
  }
}

const DEFAULT_DESCRIPTION =
  'Fetches an HTTP(S) URL and returns its readable content as markdown, suitable ' +
  'for a model to read. Scripts, styles, and other non-content noise are stripped. ' +
  'Only http:// and https:// URLs are allowed. Private, loopback, and link-local ' +
  'addresses are refused.'

export interface MakeWebFetchOptions {
  /** Tool name shown to the model. Defaults to `'web_fetch'`. */
  name?: string
  /** Tool description shown to the model. Defaults to a security-focused summary. */
  description?: string
  /**
   * Maximum response body size, in bytes. Larger responses are aborted before
   * the excess is buffered. Defaults to five mebibytes.
   */
  maxBytes?: number
  /**
   * Maximum number of HTTP redirects to follow. Each hop is revalidated
   * against the same SSRF rules as the initial URL. Defaults to five.
   */
  maxRedirects?: number
}

/**
 * Create a web fetch tool. Callers who want to tune the size cap, redirect
 * cap, or the tool's name/description use this factory; the exported
 * {@link webFetch} is a default instance with conservative limits.
 *
 * Only http and https URLs are accepted; every host is DNS-resolved and every
 * returned address must be publicly routable. Redirects are re-validated
 * against the same rules and the tool connects to an already-validated IP
 * address so a DNS rebinder cannot substitute a private address between
 * validation and connect. Response size is capped and scripts, styles, and
 * data URI images are stripped from the extracted markdown.
 */
export function makeWebFetch(options: MakeWebFetchOptions = {}): ReturnType<typeof tool> {
  const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES
  const maxRedirects = options.maxRedirects ?? DEFAULT_MAX_REDIRECTS
  if (!Number.isFinite(maxBytes) || maxBytes <= 0) {
    throw new Error(`maxBytes must be a positive number, got ${maxBytes}`)
  }
  if (!Number.isInteger(maxRedirects) || maxRedirects < 0) {
    throw new Error(`maxRedirects must be a non-negative integer, got ${maxRedirects}`)
  }

  return tool({
    name: options.name ?? 'web_fetch',
    description: options.description ?? DEFAULT_DESCRIPTION,
    inputSchema: webFetchInputSchema,
    callback: async (input, context) => {
      const { url, timeout = DEFAULT_TIMEOUT_SECONDS } = input
      const timeoutMs = timeout * 1000

      // Wire the agent's cancel signal in so a cancelled invocation kills the
      // request. The socket timeout catches unresponsive peers separately.
      const timeoutSignal = AbortSignal.timeout(timeoutMs)
      const signal = context ? AbortSignal.any([timeoutSignal, context.agent.cancelSignal]) : timeoutSignal

      const res = await _followRedirects(url, timeoutMs, maxBytes, maxRedirects, signal)
      const contentType = res.headers['content-type'] ?? ''
      const text = decodeBody(res.body, contentType)

      if (!/html|xml/i.test(contentType)) {
        return {
          url: res.effectiveUrl,
          status: res.status,
          contentType,
          title: '',
          markdown: text,
        }
      }

      const { title, markdown } = htmlToMarkdown(text)
      return {
        url: res.effectiveUrl,
        status: res.status,
        contentType,
        title,
        markdown,
      }
    },
  })
}

/**
 * Default web fetch tool with conservative size and redirect limits. See
 * {@link makeWebFetch} for a factory that lets callers tune those limits.
 *
 * @example
 * ```typescript
 * const agent = new Agent({ tools: [webFetch] })
 * ```
 */
export const webFetch = makeWebFetch()
