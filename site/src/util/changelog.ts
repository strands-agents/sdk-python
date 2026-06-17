import { getCollection, type CollectionEntry } from 'astro:content'
import type { ChangelogEntry } from '../content.config'

export type ChangelogRelease = CollectionEntry<'changelog'>

const FEATURE_TYPES = new Set(['feat', 'breaking', 'perf'])
const FIX_TYPES = new Set(['fix'])

/**
 * Compare two version strings (e.g. "1.0.0", "1.0.0-rc.1") newest-first.
 * Numeric release components outrank a prerelease of the same number
 * (1.0.0 > 1.0.0-rc.1 > 1.0.0-rc.0); prerelease identifiers compare
 * numerically where both are numbers, else lexically. Returns >0 if `b` is
 * newer than `a` (so it sorts after, consistent with date-desc usage).
 */
export function compareVersionDesc(a: string, b: string): number {
  const parse = (v: string) => {
    const [core = '', pre] = v.replace(/^v/, '').split('-')
    return { core: core.split('.').map((n) => parseInt(n, 10) || 0), pre: pre ? pre.split('.') : null }
  }
  const pa = parse(a)
  const pb = parse(b)
  for (let i = 0; i < Math.max(pa.core.length, pb.core.length); i++) {
    const d = (pb.core[i] ?? 0) - (pa.core[i] ?? 0)
    if (d !== 0) return d
  }
  // Same core: a release (no prerelease) is newer than any prerelease of it.
  if (!pa.pre && pb.pre) return -1
  if (pa.pre && !pb.pre) return 1
  if (pa.pre && pb.pre) {
    for (let i = 0; i < Math.max(pa.pre.length, pb.pre.length); i++) {
      const x = pa.pre[i] ?? ''
      const y = pb.pre[i] ?? ''
      const nx = Number(x)
      const ny = Number(y)
      const d = Number.isNaN(nx) || Number.isNaN(ny) ? y.localeCompare(x) : ny - nx
      if (d !== 0) return d
    }
  }
  return 0
}

/**
 * All releases sorted newest first. Filtering by SDK/language happens
 * client-side on the page. Ties on date are broken by version (newest first)
 * then id, so same-day releases (e.g. typescript rc.0 and rc.1) get a stable,
 * loader-order-independent ordering — which the prev/next links depend on.
 */
export async function getReleases(): Promise<ChangelogRelease[]> {
  const releases = await getCollection('changelog')
  return releases.sort((a, b) => {
    const byDate = b.data.date.getTime() - a.data.date.getTime()
    if (byDate !== 0) return byDate
    const byVersion = compareVersionDesc(a.data.version, b.data.version)
    if (byVersion !== 0) return byVersion
    return a.id.localeCompare(b.id)
  })
}

/**
 * URL slug for a release, e.g. `harness/python-v1.43.0`, `evals/v0.2.1`.
 * Derived from frontmatter (NOT collection `id`): the glob loader slugifies ids
 * with github-slugger, which strips the dots from version numbers and would
 * make `/changelog/harness/python-v1430/` (ugly and ambiguous). This keeps the
 * dotted version the team chose. Used for both the route param and links so
 * they always match.
 */
export function releaseSlug(r: ChangelogRelease): string {
  const file = r.data.language ? `${r.data.language}-v${r.data.version}` : `v${r.data.version}`
  return `${r.data.sdk}/${file}`
}

/**
 * Build the getStaticPaths array for the per-release routes, asserting slug
 * uniqueness. Two files mapping to the same slug (e.g. a duplicated sdk+lang+
 * version) would otherwise collide into one route silently; fail the build fast
 * with a clear message instead.
 */
export async function getReleasePaths(): Promise<Array<{ params: { release: string }; props: { release: ChangelogRelease } }>> {
  const releases = await getReleases()
  const seen = new Map<string, string>()
  return releases.map((release) => {
    const slug = releaseSlug(release)
    if (seen.has(slug)) {
      throw new Error(`changelog: duplicate release slug "${slug}" from ${release.id} and ${seen.get(slug)}`)
    }
    seen.set(slug, release.id)
    return { params: { release: slug }, props: { release } }
  })
}

/** A release belongs to a stream identified by sdk + language (evals has none). */
function streamKey(r: ChangelogRelease): string {
  return `${r.data.sdk}:${r.data.language ?? ''}`
}

/**
 * Newer/older neighbours of `release` within its own stream (same sdk+language),
 * for prev/next links on the detail page. `newer`/`older` are relative to date;
 * either may be null at the ends of the stream.
 */
export function getStreamNeighbors(
  release: ChangelogRelease,
  all: ChangelogRelease[]
): { newer: ChangelogRelease | null; older: ChangelogRelease | null } {
  const key = streamKey(release)
  const stream = all.filter((r) => streamKey(r) === key) // `all` is newest-first
  const i = stream.findIndex((r) => r.id === release.id)
  return {
    newer: i > 0 ? stream[i - 1] ?? null : null,
    older: i >= 0 && i < stream.length - 1 ? stream[i + 1] ?? null : null,
  }
}

interface GroupedEntries {
  features: ChangelogEntry[]
  fixes: ChangelogEntry[]
  other: ChangelogEntry[]
}

/** Group a version's entries into Features / Fixes / Other, breaking changes first within features. */
export function groupEntries(entries: ChangelogEntry[]): GroupedEntries {
  const features = entries.filter((e) => FEATURE_TYPES.has(e.type))
  features.sort((a, b) => Number(b.breaking || b.type === 'breaking') - Number(a.breaking || a.type === 'breaking'))
  return {
    features,
    fixes: entries.filter((e) => FIX_TYPES.has(e.type)),
    other: entries.filter((e) => !FEATURE_TYPES.has(e.type) && !FIX_TYPES.has(e.type)),
  }
}

export interface AreaCount {
  area: string
  count: number
}

// Areas suppressed from the facet sidebar and entry tags. `community` is a
// contribution-origin label, not a product area — surfacing it implied
// community work was a separate track from the rest of the changelog.
export const HIDDEN_AREAS = new Set(['community'])

/**
 * Count entries per area across the given entries, sorted by count desc then
 * name. Only the curated `areas` field counts — raw conventional-commit scopes
 * are deliberately NOT folded in (they're an unbounded vocabulary: `tests`,
 * `readme`, `gemini`, … which polluted the filter sidebar). Area values come
 * from `area-*` labels or the backfill classifier, both on the canonical
 * taxonomy. Must mirror the client-side `entryAreas` in the page script.
 */
export function getAreaCounts(entries: ChangelogEntry[]): AreaCount[] {
  const map = new Map<string, number>()
  for (const e of entries) {
    for (const area of e.areas) {
      if (HIDDEN_AREAS.has(area)) continue
      map.set(area, (map.get(area) ?? 0) + 1)
    }
  }
  return [...map.entries()]
    .map(([area, count]) => ({ area, count }))
    .sort((a, b) => b.count - a.count || a.area.localeCompare(b.area))
}

export function formatChangelogDate(date: Date): string {
  return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' })
}
