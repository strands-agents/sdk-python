import { getCollection, type CollectionEntry } from 'astro:content'
import type { ChangelogEntry } from '../content.config'
import type { Sdk, Language } from '../config/changelog'

export type ChangelogRelease = CollectionEntry<'changelog'>

const FEATURE_TYPES = new Set(['feat', 'breaking', 'perf'])
const FIX_TYPES = new Set(['fix'])

/** All releases sorted newest first, optionally filtered by sdk/language. */
export async function getReleases(opts: { sdk?: Sdk; language?: Language } = {}): Promise<ChangelogRelease[]> {
  const releases = await getCollection('changelog')
  return releases
    .filter((r) => (opts.sdk ? r.data.sdk === opts.sdk : true))
    .filter((r) => (opts.language ? r.data.language === opts.language : true))
    .sort((a, b) => b.data.date.getTime() - a.data.date.getTime())
}

export interface GroupedEntries {
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

/** Count entries per area across the given entries, sorted by count desc then name. */
export function getAreaCounts(entries: ChangelogEntry[]): AreaCount[] {
  const map = new Map<string, number>()
  for (const e of entries) {
    for (const area of e.areas) {
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
