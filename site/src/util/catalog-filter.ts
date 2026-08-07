/**
 * Pure filter-matching and URL-state logic for the catalog page. Kept free of
 * DOM access so the page script can import it and vitest can test it directly.
 */

import { CATALOG_TYPES } from '../components/catalog/types'

export interface CatalogFilterState {
  search: string
  types: Set<string>
  languages: Set<string>
  badges: Set<string>
  sdk: string
}

export interface CardFilterData {
  search: string
  type: string
  languages: string[]
  badges: string[]
  sdk: string
}

const KNOWN_TYPES = new Set<string>(CATALOG_TYPES.map((t) => t.value))
const KNOWN_LANGUAGES = new Set(['python', 'typescript'])
const KNOWN_BADGES = new Set(['verified', 'featured', 'new'])
const KNOWN_SDKS = new Set(['agents', 'evals'])
const DEFAULT_SDK = 'agents'

/** Facets AND together; selections within a facet OR. Empty facet = no constraint. */
export function matchesFilters(card: CardFilterData, state: CatalogFilterState): boolean {
  if (card.sdk !== state.sdk) return false
  const q = state.search.trim().toLowerCase()
  if (q && !card.search.includes(q)) return false
  if (state.types.size > 0 && !state.types.has(card.type)) return false
  if (state.languages.size > 0 && !card.languages.some((l) => state.languages.has(l))) return false
  if (state.badges.size > 0 && !card.badges.some((b) => state.badges.has(b))) return false
  return true
}

/** Serialize non-default state to a query string ('' when everything is default). */
export function stateToQuery(state: CatalogFilterState): string {
  const params = new URLSearchParams()
  if (state.search.trim()) params.set('q', state.search.trim())
  if (state.types.size > 0) params.set('type', [...state.types].sort().join(','))
  if (state.languages.size > 0) params.set('lang', [...state.languages].sort().join(','))
  if (state.badges.size > 0) params.set('badge', [...state.badges].sort().join(','))
  if (state.sdk !== DEFAULT_SDK) params.set('sdk', state.sdk)
  return params.toString()
}

/** Parse a query string into filter state, silently dropping unknown values. */
export function queryToState(query: string): CatalogFilterState {
  const params = new URLSearchParams(query)
  const pick = (key: string, known: Set<string>) =>
    new Set((params.get(key) || '').split(',').filter((v) => known.has(v)))
  const sdkParam = params.get('sdk') || DEFAULT_SDK
  return {
    search: params.get('q') || '',
    types: pick('type', KNOWN_TYPES),
    languages: pick('lang', KNOWN_LANGUAGES),
    badges: pick('badge', KNOWN_BADGES),
    sdk: KNOWN_SDKS.has(sdkParam) ? sdkParam : DEFAULT_SDK,
  }
}
