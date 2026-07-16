/**
 * SSRF defenses for the web fetch tool.
 *
 * Only http and https schemes are allowed. Hostnames are resolved and every
 * returned address is checked against a deny list of private, loopback,
 * link-local, multicast, and unspecified ranges. Every hop of a redirect chain
 * is re-validated so DNS rebinding cannot swap a public address for a private
 * one between the check and the actual connect. The caller is expected to
 * connect using one of the already-validated addresses so check-time and
 * connect-time addresses agree.
 */

import { promises as dns, type LookupAddress } from 'dns'
import { isIP } from 'net'

const ALLOWED_SCHEMES = new Set(['http:', 'https:'])

// DNS suffix denylist. Common private-network TLDs and anonymized-routing
// namespaces that must never leak a lookup to the public resolver. Compared
// case-folded, with any trailing "." stripped. Cloud metadata bare label
// "metadata" is checked separately in `assertHostIsAllowed`. Kept in sync with
// the Python implementation in `strands-py/.../_ssrf.py`.
const DENIED_DNS_SUFFIXES = [
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

// Explicit metadata endpoints. Belt-and-suspenders: every one of these
// resolves to an address the range checks below would already refuse, but
// naming them here means a future refactor of those predicates cannot silently
// expose them without also removing them from this list. Covers AWS/GCP IMDSv1
// (169.254.169.254), IPv6 EC2 (fd00:ec2::254), Alibaba
// (100.100.100.200 / 192.0.0.192), and GCP's DNS name.
const DENIED_METADATA_HOSTS = new Set([
  '169.254.169.254',
  'fd00:ec2::254',
  '100.100.100.200',
  '192.0.0.192',
  'metadata.google.internal',
  // bare-label metadata is refused before DNS in assertHostIsAllowed
])

/**
 * Throw if `url` is not an http:// or https:// URL. Returns the parsed URL.
 */
export function validateUrlScheme(url: string): URL {
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    throw new Error(`Invalid URL: ${url}`)
  }
  if (!ALLOWED_SCHEMES.has(parsed.protocol)) {
    throw new Error(
      `Only http:// and https:// URLs are allowed. Got scheme=${JSON.stringify(parsed.protocol)} for URL ${JSON.stringify(url)}.`
    )
  }
  if (!parsed.hostname) {
    throw new Error(`URL has no host: ${JSON.stringify(url)}`)
  }
  return parsed
}

/**
 * Normalize a URL hostname for comparison against the DNS denylists:
 * strip surrounding brackets from IPv6 literals, strip a trailing dot, and
 * fold to lower case.
 */
function normalizeHostForDenylist(host: string): string {
  let bare = stripIpv6Brackets(host).toLowerCase()
  if (bare.endsWith('.')) bare = bare.slice(0, -1)
  return bare
}

/**
 * Refuse the DNS suffix denylist and the explicit metadata hostnames before
 * DNS is even queried. Returns silently on allow; throws on deny.
 */
export function assertHostIsAllowed(host: string): void {
  const normalized = normalizeHostForDenylist(host)
  if (!normalized) return
  // GCP bare-label metadata endpoint.
  if (normalized === 'metadata') {
    throw new Error(`Refusing to fetch host ${JSON.stringify(host)}: bare-label metadata endpoint is refused.`)
  }
  if (DENIED_METADATA_HOSTS.has(normalized)) {
    throw new Error(
      `Refusing to fetch host ${JSON.stringify(host)}: cloud metadata endpoint ${JSON.stringify(normalized)} is refused.`
    )
  }
  for (const suffix of DENIED_DNS_SUFFIXES) {
    if (normalized === suffix.slice(1) || normalized.endsWith(suffix)) {
      throw new Error(
        `Refusing to fetch host ${JSON.stringify(host)}: DNS suffix ${JSON.stringify(suffix)} is on the denylist.`
      )
    }
  }
}

/**
 * Return true only for globally routable unicast addresses.
 *
 * Rejects private, loopback, link-local, multicast, and unspecified ranges.
 * IPv4-mapped IPv6 addresses (::ffff:a.b.c.d, ::ffff:hex:hex, and the fully
 * expanded 0:0:0:0:0:ffff:a.b.c.d form) are unwrapped before the check.
 */
export function addressIsPublic(ip: string): boolean {
  const family = isIP(ip)
  if (family === 0) return false

  if (family === 6) {
    // Unwrap IPv4-mapped IPv6 (::ffff:a.b.c.d) so we evaluate the real IPv4
    // category, not the wrapping. Anything else in IPv6 falls through to the
    // IPv6 checks below.
    //
    // We expand to the full 8-hextet form and check whether the first six
    // hextets are 0000:0000:0000:0000:0000:ffff. This catches both the
    // dotted-quad form (`::ffff:127.0.0.1`) and the pure-hex form
    // (`::ffff:7f00:1`) that Node's URL parser normalizes to.
    const lower = ip.toLowerCase()
    const expanded = expandIpv6(lower)
    if (expanded !== null && expanded.startsWith('0000:0000:0000:0000:0000:ffff:')) {
      const parts = expanded.split(':')
      const high = parseInt(parts[6] ?? '0', 16)
      const low = parseInt(parts[7] ?? '0', 16)
      const a = (high >> 8) & 0xff
      const b = high & 0xff
      const c = (low >> 8) & 0xff
      const d = low & 0xff
      return addressIsPublic(`${a}.${b}.${c}.${d}`)
    }
    return ipv6IsPublic(lower)
  }

  return ipv4IsPublic(ip)
}

function ipv4IsPublic(ip: string): boolean {
  const parts = ip.split('.').map((p) => Number(p))
  if (parts.length !== 4 || parts.some((n) => !Number.isInteger(n) || n < 0 || n > 255)) {
    return false
  }
  const [a, b, c] = parts as [number, number, number, number]
  // 0.0.0.0/8 -- unspecified/current network
  if (a === 0) return false
  // 10.0.0.0/8 -- private
  if (a === 10) return false
  // 100.64.0.0/10 -- CGNAT
  if (a === 100 && b >= 64 && b <= 127) return false
  // 127.0.0.0/8 -- loopback
  if (a === 127) return false
  // 169.254.0.0/16 -- link-local (incl. cloud metadata endpoints)
  if (a === 169 && b === 254) return false
  // 172.16.0.0/12 -- private
  if (a === 172 && b >= 16 && b <= 31) return false
  // 192.0.0.0/24 -- IETF protocol assignments
  if (a === 192 && b === 0 && c === 0) return false
  // 192.0.2.0/24 -- TEST-NET-1
  if (a === 192 && b === 0 && c === 2) return false
  // 192.168.0.0/16 -- private
  if (a === 192 && b === 168) return false
  // 198.18.0.0/15 -- benchmarking
  if (a === 198 && (b === 18 || b === 19)) return false
  // 198.51.100.0/24 -- TEST-NET-2
  if (a === 198 && b === 51 && c === 100) return false
  // 203.0.113.0/24 -- TEST-NET-3
  if (a === 203 && b === 0 && c === 113) return false
  // 224.0.0.0/4 -- multicast
  if (a >= 224 && a <= 239) return false
  // 240.0.0.0/4 -- reserved / broadcast
  if (a >= 240) return false
  return true
}

function ipv6IsPublic(ip: string): boolean {
  const lower = ip.toLowerCase()
  // Unspecified :: and loopback ::1
  if (lower === '::' || lower === '::1') return false
  const expanded = expandIpv6(lower)
  if (expanded === null) return false
  // Link-local fe80::/10
  if (
    expanded.startsWith('fe8') ||
    expanded.startsWith('fe9') ||
    expanded.startsWith('fea') ||
    expanded.startsWith('feb')
  ) {
    return false
  }
  // Site-local fec0::/10 (deprecated but still filtered).
  if (
    expanded.startsWith('fec') ||
    expanded.startsWith('fed') ||
    expanded.startsWith('fee') ||
    expanded.startsWith('fef')
  ) {
    return false
  }
  // Unique-local fc00::/7 -> fc.. or fd..
  if (expanded.startsWith('fc') || expanded.startsWith('fd')) return false
  // Multicast ff00::/8
  if (expanded.startsWith('ff')) return false
  // Documentation 2001:db8::/32
  if (expanded.startsWith('2001:0db8:')) return false
  // 100::/64 -- discard-only address block (first 64 bits are zero).
  if (expanded.startsWith('0100:0000:0000:0000:')) return false
  return true
}

/**
 * Expand an IPv6 address to its fully-written 8-group form with each group
 * zero-padded to 4 hex digits. Returns null if the input is not a valid IPv6
 * address. Used for prefix matching against reserved ranges and for
 * IPv4-mapped-IPv6 unwrapping.
 */
function expandIpv6(ip: string): string | null {
  // Reject anything that isn't valid IPv6 up front.
  if (isIP(ip) !== 6) return null
  const lower = ip.toLowerCase()

  // Node accepts a trailing dotted-quad form (`::ffff:127.0.0.1`). Convert it
  // to the equivalent pure-hex form before regular expansion so we can produce
  // 8 hextets uniformly.
  const dottedMatch = lower.match(/^(.*:)(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/)
  let prefix = lower
  if (dottedMatch) {
    const nums = [dottedMatch[2], dottedMatch[3], dottedMatch[4], dottedMatch[5]].map((s) => Number(s))
    if (nums.some((n) => !Number.isInteger(n) || n < 0 || n > 255)) return null
    const hi = ((nums[0] ?? 0) << 8) | (nums[1] ?? 0)
    const lo = ((nums[2] ?? 0) << 8) | (nums[3] ?? 0)
    prefix = `${dottedMatch[1] ?? ''}${hi.toString(16)}:${lo.toString(16)}`
  }

  const parts = prefix.split('::')
  if (parts.length > 2) return null
  const head = parts[0] === '' ? [] : (parts[0] ?? '').split(':')
  const tail = parts.length === 2 ? (parts[1] === '' ? [] : (parts[1] ?? '').split(':')) : []
  const missing = 8 - head.length - tail.length
  if (missing < 0) return null
  const groups = [...head, ...Array<string>(missing).fill('0'), ...tail]
  if (groups.length !== 8) return null
  return groups.map((g) => g.padStart(4, '0')).join(':')
}

/**
 * Strip the surrounding `[...]` from an IPv6 literal host. `new URL().hostname`
 * keeps the brackets, but `isIP()` and `dns.lookup()` want the bare address.
 */
export function stripIpv6Brackets(host: string): string {
  if (host.length >= 2 && host.startsWith('[') && host.endsWith(']')) {
    return host.slice(1, -1)
  }
  return host
}

/**
 * Resolve `host` to IP addresses and require every one to be public.
 *
 * Returns the resolved addresses on success. If the host is an IP literal it is
 * validated directly and returned as-is. If any resolved address is not
 * public, throws. The DNS-suffix denylist and named metadata endpoints are
 * refused *before* DNS is queried.
 */
export async function resolveAndValidateHost(host: string): Promise<string[]> {
  // Refuse .internal / .onion / metadata.google.internal etc. before DNS.
  assertHostIsAllowed(host)

  // Strip [ ] around IPv6 URL literals so isIP / DNS see the bare address.
  const bare = stripIpv6Brackets(host)
  const family = isIP(bare)
  if (family !== 0) {
    // Host is an IP literal -- no DNS query, just validate.
    if (!addressIsPublic(bare)) {
      throw new Error(
        `Refusing to fetch host ${JSON.stringify(host)}: address ${JSON.stringify(bare)} is not public (private, loopback, link-local, site-local, multicast, CGNAT, or reserved).`
      )
    }
    return [bare]
  }

  let records: LookupAddress[]
  try {
    records = await dns.lookup(bare, { all: true, verbatim: true })
  } catch (err) {
    throw new Error(`Could not resolve host ${JSON.stringify(host)}: ${(err as Error).message}`, {
      cause: err,
    })
  }
  if (records.length === 0) {
    throw new Error(`Could not resolve host ${JSON.stringify(host)}: no addresses returned`)
  }
  const addresses: string[] = []
  for (const rec of records) {
    if (!addressIsPublic(rec.address)) {
      throw new Error(
        `Refusing to fetch host ${JSON.stringify(host)}: resolved address ${JSON.stringify(rec.address)} is not public (private, loopback, link-local, site-local, multicast, CGNAT, or reserved).`
      )
    }
    addresses.push(rec.address)
  }
  return addresses
}
