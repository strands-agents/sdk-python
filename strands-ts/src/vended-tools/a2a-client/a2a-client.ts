/**
 * A2A client vended tool.
 *
 * Thin shim over {@link A2AAgent}. The model supplies a URL and a message;
 * the tool resolves the remote agent's card and sends the message. URL
 * validation, size caps, and total timeout are enforced at the tool
 * boundary — {@link A2AAgent} has no SSRF or size guard of its own.
 */

import { Buffer } from 'buffer'
import { z } from 'zod'
import { A2AAgent } from '../../a2a/a2a-agent.js'
import { tool } from '../../tools/tool-factory.js'
import type { InvokableTool } from '../../tools/tool.js'
import { UrlNotAllowedError, validateUrl } from './url-guard.js'
import type { A2AClientOutput, A2AClientRemoteCard, MakeA2AClientOptions } from './types.js'

/**
 * Default tool description shown to the model.
 */
export const DEFAULT_A2A_CLIENT_DESCRIPTION =
  'Invokes a remote A2A (Agent-to-Agent) agent by URL. Resolves the agent card at ' +
  "the given URL and sends a single message. Returns the remote agent's response as text. " +
  'Only http(s) URLs to public hosts are permitted; private/loopback/link-local addresses ' +
  'are rejected.'

const DEFAULT_TIMEOUT_SECONDS = 60
const DEFAULT_MAX_CARD_BYTES = 256 * 1024
const DEFAULT_MAX_RESPONSE_BYTES = 256 * 1024
const MAX_MESSAGE_BYTES = 64 * 1024
const DEFAULT_MULTIAGENT_DEPTH_CAP = 3

const inputSchema = z.object({
  url: z
    .string()
    .describe(
      'HTTP(S) URL of the remote A2A agent (its base URL). Only public hosts are permitted. ' +
        'Private, loopback, link-local, and cloud metadata addresses are rejected.'
    ),
  message: z.string().describe('Message to send to the remote agent. Capped at 64 KiB.'),
})

/**
 * Create an `a2a_client` tool.
 *
 * All security-relevant configuration is bound at creation time by the developer.
 * The model can only supply `url` and `message` at call time. Auth material
 * carried on `agentConfig.clientFactory` is never accessible to the model.
 *
 * @param options - Developer-time options.
 * @returns An {@link InvokableTool} that invokes a remote A2A agent.
 */
export function makeA2AClient(
  options: MakeA2AClientOptions = {}
): InvokableTool<z.infer<typeof inputSchema>, A2AClientOutput> {
  const {
    name = 'a2a_client',
    description = DEFAULT_A2A_CLIENT_DESCRIPTION,
    allowedUrlPrefixes,
    timeoutSeconds = DEFAULT_TIMEOUT_SECONDS,
    maxCardBytes = DEFAULT_MAX_CARD_BYTES,
    maxResponseBytes = DEFAULT_MAX_RESPONSE_BYTES,
    agentConfig,
    multiagentDepthCap = DEFAULT_MULTIAGENT_DEPTH_CAP,
  } = options

  return tool({
    name,
    description,
    inputSchema,
    callback: async (input, context) => {
      const { url, message } = input

      // --- Multiagent depth cap ---
      // The a2a_client tool participates in the shared depth counter so an
      // adversarial chain (use_agent -> a2a_client -> ...) still respects the
      // cap. The counter is not propagated across the wire (documented in
      // `_multiagent-conventions.md`).
      const rawDepth = (context?.invocationState as { multiagentDepth?: unknown } | undefined)?.multiagentDepth
      const depth = typeof rawDepth === 'number' && Number.isFinite(rawDepth) ? rawDepth : 0
      if (depth >= multiagentDepthCap) {
        throw new Error(`multiagentDepth=${depth} exceeds cap ${multiagentDepthCap}`)
      }

      // --- Input validation ---
      if (typeof message !== 'string') {
        throw new Error(`message must be a string, got ${typeof message}`)
      }
      const messageBytes = Buffer.byteLength(message, 'utf-8')
      if (messageBytes > MAX_MESSAGE_BYTES) {
        throw new Error(`message is ${messageBytes} bytes; limit is ${MAX_MESSAGE_BYTES}`)
      }

      try {
        await validateUrl(url, allowedUrlPrefixes)
      } catch (err) {
        if (err instanceof UrlNotAllowedError) {
          // Rethrow as a plain Error so the tool wrapper formats it as an error status.
          throw new Error(err.message, { cause: err })
        }
        throw err
      }

      // --- Run under a total-timeout budget, respecting agent cancellation ---
      const startedAtMs = Date.now()
      const timeoutSignal = AbortSignal.timeout(timeoutSeconds * 1000)
      const parentSignal = context?.agent?.cancelSignal
      const abortSignal = parentSignal ? AbortSignal.any([timeoutSignal, parentSignal]) : timeoutSignal
      if (abortSignal.aborted) {
        throw (
          abortSignal.reason ?? new DOMException('a2a_client invocation was aborted before it started', 'AbortError')
        )
      }

      // Plumb a redirect-guarding fetch so both card discovery and every
      // subsequent transport hop are re-validated by `validateUrl`. The A2A
      // SDK defers to the developer's factory if one was supplied; otherwise
      // it uses the default factory constructed from our `fetchImpl`.
      const remote = new A2AAgent({
        ...(agentConfig ?? {}),
        url,
        // Only inject the guarded fetch if the developer didn't hand us a
        // pre-built factory. Their factory is expected to already carry
        // whatever fetch discipline it needs.
        ...(agentConfig?.clientFactory ? {} : { fetchImpl: makeGuardedFetch(allowedUrlPrefixes) }),
      })

      try {
        return await runToCompletion({
          remote,
          message,
          maxCardBytes,
          maxResponseBytes,
          abortSignal,
          requestUrl: url,
          allowedPrefixes: allowedUrlPrefixes,
          startedAtMs,
        })
      } catch (err) {
        if (timeoutSignal.aborted) {
          throw new DOMException(`a2a_client timed out after ${timeoutSeconds} seconds calling ${url}`, 'AbortError')
        }
        if (parentSignal?.aborted) {
          throw parentSignal.reason ?? new DOMException(`a2a_client cancelled while calling ${url}`, 'AbortError')
        }
        throw err
      }
    },
  })
}

/** Default `a2a_client` tool — no allowlist, no custom agent config. */
export const a2aClient = makeA2AClient()

/**
 * Run the underlying {@link A2AAgent} with total-timeout + cancellation.
 *
 * Fetches and validates the agent card *before* sending the message so a
 * malicious card `url` can't steer the send to a private host. Aborts the
 * async iteration when the abort signal fires so we don't accumulate bytes
 * past the deadline. Enforces card + response size caps.
 *
 * Note: the timeout is a soft cap. `A2AAgent#invoke` does not accept an
 * `AbortSignal`, so racing against `abortSignal` rejects the wrapper promise
 * on timeout but leaves the underlying HTTP request running until it
 * completes on its own. If you need a hard cap, plumb your own transport
 * `AbortSignal` through `agentConfig.clientFactory`.
 */
async function runToCompletion(args: {
  remote: A2AAgent
  message: string
  maxCardBytes: number
  maxResponseBytes: number
  abortSignal: AbortSignal
  requestUrl: string
  allowedPrefixes: readonly string[] | undefined
  startedAtMs: number
}): Promise<A2AClientOutput> {
  const { remote, message, maxCardBytes, maxResponseBytes, abortSignal, requestUrl, allowedPrefixes, startedAtMs } =
    args

  const abortPromise = new Promise<never>((_resolve, reject) => {
    const onAbort = (): void => {
      abortSignal.removeEventListener('abort', onAbort)
      reject(abortSignal.reason ?? new DOMException('a2a_client aborted', 'AbortError'))
    }
    if (abortSignal.aborted) {
      reject(abortSignal.reason ?? new DOMException('a2a_client aborted', 'AbortError'))
      return
    }
    abortSignal.addEventListener('abort', onAbort, { once: true })
  })

  // Resolve the card first, guard its `url`, then send. Otherwise a malicious
  // remote could return `card.url = "http://169.254.169.254"` and the message
  // would already be sent by the time we saw it.
  const card = await Promise.race([remote.getAgentCard(), abortPromise])
  assertCardWithinSizeLimit(card, maxCardBytes)
  if (card.url && card.url !== requestUrl) {
    // Re-apply the developer allowlist here: the card-advertised URL is the
    // actual send target, so a pinned allowlist bounds it just as it bounds
    // the model-supplied URL.
    try {
      await validateUrl(card.url, allowedPrefixes)
    } catch (err) {
      if (err instanceof UrlNotAllowedError) {
        throw new Error(`remote agent card points at disallowed url: ${err.message}`, { cause: err })
      }
      throw err
    }
  }

  if (abortSignal.aborted) {
    throw abortSignal.reason ?? new DOMException('a2a_client aborted', 'AbortError')
  }

  const result = await Promise.race([remote.invoke(message), abortPromise])

  const text = truncateUtf8(result.toString(), maxResponseBytes)
  const remoteCard: A2AClientRemoteCard = {
    name: card.name ?? remote.name ?? '',
    description: card.description ?? remote.description ?? '',
    url: card.url ?? requestUrl,
  }

  return {
    status: 'success',
    output: text,
    executionTimeMs: Date.now() - startedAtMs,
    remoteCard,
  }
}

/**
 * Wrap `fetch` so every hop is re-validated by {@link validateUrl}.
 *
 * The default `fetch` follows redirects transparently, which lets a
 * public-looking origin (`example.com`) 302 the client to a private target
 * like `http://169.254.169.254/…`. Manual redirect walking closes that hole:
 *
 * 1. `redirect: 'manual'` — the runtime returns the 3xx as a normal response.
 * 2. Read `Location`, resolve against the current request URL.
 * 3. Re-run the URL through {@link validateUrl} (same SSRF policy plus the
 *    developer allowlist, so a 3xx cannot escape the pinned prefix set).
 * 4. Repeat, up to a hard cap of 5 hops.
 * 5. Return the first non-3xx response.
 *
 * `Authorization` / `Cookie` / `Proxy-Authorization` headers are dropped on a
 * cross-origin redirect (compared on scheme + host + port, not host alone),
 * so a change to http on the same host or a port change also strips them.
 *
 * If the caller passes a `Request` object, its method, headers, body, and
 * other properties are preserved by constructing a fresh `Request` per hop.
 */
function makeGuardedFetch(allowedPrefixes?: readonly string[]): typeof globalThis.fetch {
  const MAX_REDIRECTS = 5
  const SENSITIVE_HEADERS = ['authorization', 'cookie', 'proxy-authorization']

  const guarded: typeof globalThis.fetch = async (input, init) => {
    // Materialise the caller-supplied input into a URL string plus a
    // Request-shaped init that carries method/headers/body across hops.
    // Preserving these matters when the caller passed a Request object —
    // extracting only `.url` and forwarding a plain string would drop the
    // rest of the request.
    let currentUrl: string
    let currentInit: RequestInit
    if (typeof input === 'string' || input instanceof URL) {
      currentUrl = input.toString()
      currentInit = init ? { ...init, redirect: 'manual' } : { redirect: 'manual' }
    } else {
      // input is a Request. Copy its state onto an init so we can rewrite
      // the URL per hop without losing method/headers/body. A caller-supplied
      // init overrides the Request's fields, but `redirect: 'manual'` is
      // pinned last so a caller can't override the manual-walk invariant
      // (which would silently bypass the per-hop `validateUrl`).
      const req = input
      currentUrl = req.url
      currentInit = {
        method: req.method,
        headers: new Headers(req.headers),
        body: req.body,
        mode: req.mode,
        credentials: req.credentials,
        cache: req.cache,
        referrer: req.referrer,
        referrerPolicy: req.referrerPolicy,
        integrity: req.integrity,
        keepalive: req.keepalive,
        signal: req.signal,
        ...(init ?? {}),
        redirect: 'manual',
      }
    }

    for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
      // Validate the URL before hitting it. Redirect targets get the same
      // treatment as the caller-supplied URL, *including* the developer
      // allowlist, so a 3xx from an allowlisted origin cannot steer the
      // client onto an off-list public host.
      await validateUrl(currentUrl, allowedPrefixes)

      const response = await globalThis.fetch(currentUrl, currentInit)
      const status = response.status
      if (status < 300 || status >= 400 || status === 304) {
        return response
      }
      // 3xx redirect — walk it manually.
      const location = response.headers.get('location')
      if (!location) {
        return response
      }
      if (hop === MAX_REDIRECTS) {
        throw new Error(`a2a_client: redirect cap (${MAX_REDIRECTS}) exceeded`)
      }
      let nextUrl: string
      try {
        nextUrl = new URL(location, currentUrl).toString()
      } catch {
        throw new Error(`a2a_client: invalid redirect Location "${location}"`)
      }
      // Compare on full origin (scheme + host + port), not just host, so a
      // scheme downgrade (https→http) or port change on the same host also
      // triggers sensitive-header stripping.
      const prevOrigin = new URL(currentUrl).origin
      const nextOrigin = new URL(nextUrl).origin
      if (prevOrigin !== nextOrigin && currentInit.headers) {
        const nextHeaders: Headers = new Headers(currentInit.headers)
        for (const h of SENSITIVE_HEADERS) {
          nextHeaders.delete(h)
        }
        currentInit = { ...currentInit, headers: nextHeaders }
      }
      currentUrl = nextUrl
    }
    // Loop cap is enforced above; unreachable, but satisfies TS.
    throw new Error('a2a_client: redirect walk terminated unexpectedly')
  }
  return guarded
}

/** Reject cards whose serialized JSON is larger than `limit` bytes. */
function assertCardWithinSizeLimit(card: object, limit: number): void {
  let serialized: string
  try {
    serialized = JSON.stringify(card)
  } catch {
    serialized = String(card)
  }
  const size = Buffer.byteLength(serialized, 'utf-8')
  if (size > limit) {
    throw new Error(`remote agent card is ${size} bytes; limit is ${limit}`)
  }
}

/**
 * Truncate `text` to at most `maxBytes` UTF-8 bytes without splitting a code
 * point. Appends `... [truncated]` when truncation occurs. The returned
 * string re-encoded as UTF-8 is guaranteed to be no larger than `maxBytes`.
 */
function truncateUtf8(text: string, maxBytes: number): string {
  const encoded = Buffer.from(text, 'utf-8')
  if (encoded.byteLength <= maxBytes) {
    return text
  }
  const suffix = '... [truncated]'
  const suffixBytes = Buffer.byteLength(suffix, 'utf-8')
  const budget = Math.max(0, maxBytes - suffixBytes)
  // `Buffer.slice(...).toString('utf-8')` on a boundary that falls inside a
  // multi-byte sequence emits U+FFFD (which re-encodes to three bytes) and
  // can exceed `maxBytes`. Walk back to the previous code-point boundary
  // instead: in UTF-8, continuation bytes have the top bits `10xxxxxx`,
  // and lead bytes are either ASCII (`0xxxxxxx`) or `11xxxxxx`.
  let end = budget
  while (end > 0 && (encoded[end]! & 0xc0) === 0x80) {
    end--
  }
  const truncated = encoded.subarray(0, end).toString('utf-8')
  return truncated + suffix
}
