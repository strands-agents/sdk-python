/**
 * URL validation for the a2a_client tool.
 *
 * Blocks the SSRF surface a model can steer via a URL: scheme allowlist,
 * private / loopback / link-local IPs, cloud-metadata IPs, and reserved DNS
 * suffixes such as `.internal`, `.local`, `.corp`. Validation is run both
 * on the initial URL and on the `url` field of the resolved agent card,
 * because a remote card can point at a different host than the request URL.
 */

import * as dns from 'node:dns/promises'
import * as net from 'node:net'

interface DnsLookupAddress {
  address: string
  family: number
}

/**
 * Hostnames whose DNS suffix we refuse outright. Cheaper than DNS resolution
 * and catches operators who put private services on `.internal` / `.local` /
 * `.corp` / `.home` even when those don't resolve to a private IP for us.
 */
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
]

/**
 * Bare hostnames — checked before DNS resolution. GCP's metadata server
 * answers on `http://metadata/` inside a VPC; blocking the label
 * short-circuits any resolver that would answer this locally.
 */
const BLOCKED_BARE_HOSTNAMES: readonly string[] = ['metadata']

/**
 * Well-known cloud metadata addresses. IPv4 metadata is in the link-local
 * range (169.254/16), which is caught by the private-IP check, but we spell
 * these out explicitly to catch anything a resolver returns via an unrelated
 * hostname.
 */
const BLOCKED_METADATA_ADDRESSES: readonly string[] = [
  '169.254.169.254',
  'fd00:ec2::254',
  '100.100.100.200',
  '192.0.0.192',
]

/** Error thrown when a URL fails validation before we make a network call. */
export class UrlNotAllowedError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'UrlNotAllowedError'
  }
}

/**
 * Validate a URL for use as an A2A endpoint.
 *
 * Enforces:
 * - http/https scheme only
 * - a non-empty hostname
 * - hostname not on the blocked-suffix list
 * - every DNS-resolved address is a global unicast address (not private,
 *   loopback, link-local, multicast, reserved, or unspecified)
 * - hostname not a known cloud-metadata IP
 * - URL starts with one of `allowedPrefixes` if supplied
 *
 * @param url - The URL to check.
 * @param allowedPrefixes - Optional developer-supplied allowlist.
 * @returns The parsed hostname (lowercased).
 * @throws UrlNotAllowedError if any check fails.
 */
export async function validateUrl(url: string, allowedPrefixes?: readonly string[]): Promise<string> {
  if (typeof url !== 'string' || url.length === 0) {
    throw new UrlNotAllowedError('url must be a non-empty string')
  }
  if (allowedPrefixes && !allowedPrefixes.some((prefix) => url.startsWith(prefix))) {
    throw new UrlNotAllowedError(
      `url ${JSON.stringify(url)} is not in the developer-configured allowlist ` +
        `(expected one of ${JSON.stringify(allowedPrefixes)})`
    )
  }

  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    throw new UrlNotAllowedError(`url is not a valid URL: ${url}`)
  }

  const scheme = parsed.protocol.replace(/:$/, '').toLowerCase()
  if (scheme !== 'http' && scheme !== 'https') {
    throw new UrlNotAllowedError(`url must use http or https, got scheme "${scheme}"`)
  }

  let host = stripBrackets(parsed.hostname).toLowerCase()
  // Strip a trailing dot so `foo.internal.` is caught by the suffix check.
  if (host.endsWith('.')) {
    host = host.slice(0, -1)
  }
  if (!host) {
    throw new UrlNotAllowedError('url must include a hostname')
  }

  if (BLOCKED_BARE_HOSTNAMES.includes(host)) {
    // GCP's metadata server answers on `http://metadata/` inside a VPC;
    // reject the label before we ever DNS-resolve.
    throw new UrlNotAllowedError(`hostname "${host}" is a cloud metadata label`)
  }

  for (const suffix of BLOCKED_HOST_SUFFIXES) {
    if (host === suffix.replace(/^\./, '') || host.endsWith(suffix)) {
      throw new UrlNotAllowedError(`hostname "${host}" matches blocked suffix "${suffix}"`)
    }
  }

  await assertHostResolvesToPublicIps(host)
  return host
}

/**
 * Resolve `host` and reject if any address is private, loopback, link-local,
 * multicast, reserved, unspecified, or a cloud-metadata address. If `host`
 * is a bare IP literal, skip DNS and check it directly.
 */
async function assertHostResolvesToPublicIps(host: string): Promise<void> {
  const ipVersion = net.isIP(host)
  if (ipVersion !== 0) {
    assertIpIsPublic(host)
    return
  }

  let addresses: DnsLookupAddress[]
  try {
    addresses = (await dns.lookup(host, { all: true, verbatim: true })) as DnsLookupAddress[]
  } catch (err) {
    throw new UrlNotAllowedError(`could not resolve hostname "${host}": ${(err as Error).message}`)
  }

  if (addresses.length === 0) {
    throw new UrlNotAllowedError(`hostname "${host}" resolved to no addresses`)
  }

  const seen = new Set<string>()
  for (const { address } of addresses) {
    if (seen.has(address)) continue
    seen.add(address)
    assertIpIsPublic(address)
  }
}

/**
 * Assert that an IP address is global-unicast public. Rejects loopback,
 * link-local, private, multicast, reserved, unspecified, and cloud-metadata IPs.
 */
function assertIpIsPublic(ip: string): void {
  if (BLOCKED_METADATA_ADDRESSES.includes(ip.toLowerCase())) {
    throw new UrlNotAllowedError(`ip ${ip} is a cloud metadata address`)
  }
  const family = net.isIP(ip)
  if (family === 0) {
    throw new UrlNotAllowedError(`invalid ip address: ${ip}`)
  }
  if (family === 4) {
    assertIpv4IsPublic(ip)
    return
  }
  assertIpv6IsPublic(ip)
}

function assertIpv4IsPublic(ip: string): void {
  const parts = ip.split('.').map(Number)
  if (parts.length !== 4 || parts.some((n) => !Number.isInteger(n) || n < 0 || n > 255)) {
    throw new UrlNotAllowedError(`invalid ipv4 address: ${ip}`)
  }
  const [a, b, c] = parts as [number, number, number, number]
  // 0.0.0.0/8 unspecified / this-network
  if (a === 0) throw new UrlNotAllowedError(`ip ${ip} is unspecified`)
  // 127.0.0.0/8 loopback
  if (a === 127) throw new UrlNotAllowedError(`ip ${ip} is loopback`)
  // 10.0.0.0/8
  if (a === 10) throw new UrlNotAllowedError(`ip ${ip} is private`)
  // 172.16.0.0/12
  if (a === 172 && b >= 16 && b <= 31) throw new UrlNotAllowedError(`ip ${ip} is private`)
  // 192.168.0.0/16
  if (a === 192 && b === 168) throw new UrlNotAllowedError(`ip ${ip} is private`)
  // 169.254.0.0/16 link-local
  if (a === 169 && b === 254) throw new UrlNotAllowedError(`ip ${ip} is link-local`)
  // 100.64.0.0/10 CGNAT (carrier-grade NAT)
  if (a === 100 && b >= 64 && b <= 127) throw new UrlNotAllowedError(`ip ${ip} is private (CGNAT)`)
  // 192.0.0.0/24 IETF protocol assignments
  if (a === 192 && b === 0 && c === 0) throw new UrlNotAllowedError(`ip ${ip} is reserved`)
  // 192.0.2.0/24 TEST-NET-1
  if (a === 192 && b === 0 && c === 2) throw new UrlNotAllowedError(`ip ${ip} is reserved (documentation)`)
  // 198.18.0.0/15 benchmarking
  if (a === 198 && (b === 18 || b === 19)) throw new UrlNotAllowedError(`ip ${ip} is reserved (benchmarking)`)
  // 198.51.100.0/24 TEST-NET-2
  if (a === 198 && b === 51 && c === 100) throw new UrlNotAllowedError(`ip ${ip} is reserved (documentation)`)
  // 203.0.113.0/24 TEST-NET-3
  if (a === 203 && b === 0 && c === 113) throw new UrlNotAllowedError(`ip ${ip} is reserved (documentation)`)
  // 224.0.0.0/4 multicast
  if (a >= 224 && a <= 239) throw new UrlNotAllowedError(`ip ${ip} is multicast`)
  // 240.0.0.0/4 reserved (255.255.255.255 falls in here)
  if (a >= 240) throw new UrlNotAllowedError(`ip ${ip} is reserved`)
}

function assertIpv6IsPublic(ip: string): void {
  const lower = ip.toLowerCase()

  // IPv4-mapped / -translated IPv6 -- unwrap and evaluate the embedded v4.
  // `URL.hostname` normalises `::ffff:127.0.0.1` into the hex form
  // `::ffff:7f00:1`, so both dotted and hex representations must be handled.
  // Without this, `http://[::ffff:127.0.0.1]` bypasses the IPv4 rules.
  const embedded = extractMappedIpv4(lower)
  if (embedded !== null) {
    assertIpv4IsPublic(embedded)
    return
  }

  // Categorical checks by operating on the full 128-bit expansion rather
  // than string-shape regexes. A prior regex-based version accepted
  // multicast/link-local addresses in some short-form representations because
  // it required a specific hextet length in the first group.
  const bytes = expandIpv6(lower)
  if (bytes === null) {
    throw new UrlNotAllowedError(`invalid ipv6 address: ${ip}`)
  }

  // ::1 loopback
  if (bytes.every((b, i) => (i < 15 ? b === 0 : b === 1))) {
    throw new UrlNotAllowedError(`ip ${ip} is loopback`)
  }
  // :: unspecified
  if (bytes.every((b) => b === 0)) {
    throw new UrlNotAllowedError(`ip ${ip} is unspecified`)
  }
  // fe80::/10 link-local — top ten bits are 1111_1110_10
  if (bytes[0] === 0xfe && (bytes[1]! & 0xc0) === 0x80) {
    throw new UrlNotAllowedError(`ip ${ip} is link-local`)
  }
  // fc00::/7 unique local (private) — top seven bits are 1111_110
  if ((bytes[0]! & 0xfe) === 0xfc) {
    throw new UrlNotAllowedError(`ip ${ip} is private (unique local)`)
  }
  // ff00::/8 multicast — top byte is 0xff
  if (bytes[0] === 0xff) {
    throw new UrlNotAllowedError(`ip ${ip} is multicast`)
  }
  // 2001:db8::/32 documentation
  if (bytes[0] === 0x20 && bytes[1] === 0x01 && bytes[2] === 0x0d && bytes[3] === 0xb8) {
    throw new UrlNotAllowedError(`ip ${ip} is reserved (documentation)`)
  }
  // fec0::/10 site-local (deprecated but still worth blocking) —
  // top ten bits are 1111_1110_11
  if (bytes[0] === 0xfe && (bytes[1]! & 0xc0) === 0xc0) {
    throw new UrlNotAllowedError(`ip ${ip} is reserved (site-local)`)
  }
}

/**
 * Expand an IPv6 literal to a 16-byte array of octets, or `null` if the input
 * is not a valid IPv6 address. Handles zero-compression (`::`), variable
 * hextet widths, and IPv4-in-IPv6 dotted-quad tails.
 */
function expandIpv6(ip: string): number[] | null {
  // Handle the trailing dotted-quad form (e.g. `::ffff:1.2.3.4`) by folding
  // the four decimal octets into two hex hextets before splitting.
  let normalized = ip
  const dottedTail = /^([0-9a-f:]*):(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(ip)
  if (dottedTail) {
    const prefix = dottedTail[1]!
    const parts = [dottedTail[2], dottedTail[3], dottedTail[4], dottedTail[5]].map((s) => Number(s))
    if (parts.some((n) => !Number.isInteger(n) || n < 0 || n > 255)) return null
    const hi = ((parts[0]! << 8) | parts[1]!).toString(16)
    const lo = ((parts[2]! << 8) | parts[3]!).toString(16)
    normalized = `${prefix}:${hi}:${lo}`
  }

  const doubleColonIndex = normalized.indexOf('::')
  let head: string[]
  let tail: string[]
  if (doubleColonIndex === -1) {
    head = normalized.split(':')
    tail = []
  } else {
    const before = normalized.slice(0, doubleColonIndex)
    const after = normalized.slice(doubleColonIndex + 2)
    head = before === '' ? [] : before.split(':')
    tail = after === '' ? [] : after.split(':')
  }
  const missing = 8 - head.length - tail.length
  if (missing < 0) return null
  const groups = [...head, ...Array<string>(missing).fill('0'), ...tail]
  const bytes: number[] = []
  for (const group of groups) {
    if (!/^[0-9a-f]{1,4}$/.test(group)) return null
    const value = parseInt(group, 16)
    bytes.push((value >> 8) & 0xff, value & 0xff)
  }
  return bytes.length === 16 ? bytes : null
}

function stripBrackets(host: string): string {
  if (host.startsWith('[') && host.endsWith(']')) {
    return host.slice(1, -1)
  }
  return host
}

/**
 * If `ip` is an IPv4-mapped IPv6 address in either dotted (`::ffff:a.b.c.d`)
 * or hex (`::ffff:7f00:1`) form, return the embedded IPv4 as a dotted-quad
 * string. Accepts the fully-expanded prefix (`0:0:0:0:0:ffff:...`) as well.
 * Returns `null` if `ip` is not an IPv4-mapped IPv6 address.
 */
function extractMappedIpv4(ip: string): string | null {
  const lower = ip.toLowerCase()

  // Dotted form: `::ffff:a.b.c.d` (and its fully-expanded 6-group variant).
  const dotted = lower.match(
    /^(?:0:0:0:0:0:)?::?ffff:(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$|^0:0:0:0:0:ffff:(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$/
  )
  if (dotted) {
    return dotted[1] ?? dotted[2] ?? null
  }

  // Hex form: `::ffff:XXXX:XXXX` or `0:0:0:0:0:ffff:XXXX:XXXX`.
  const hex = lower.match(
    /^(?:0:0:0:0:0:)?::?ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$|^0:0:0:0:0:ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/
  )
  if (hex) {
    const hi = hex[1] ?? hex[3]
    const lo = hex[2] ?? hex[4]
    if (hi !== undefined && lo !== undefined) {
      const hiN = parseInt(hi, 16)
      const loN = parseInt(lo, 16)
      const a = (hiN >> 8) & 0xff
      const b = hiN & 0xff
      const c = (loN >> 8) & 0xff
      const d = loN & 0xff
      return `${a}.${b}.${c}.${d}`
    }
  }
  return null
}
