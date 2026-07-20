/**
 * Build-time processing for catalog entries: derives card view-models from
 * collection data, joins the bot-maintained stats file, and orders the grid.
 */

import type { CatalogEntryData } from '../content.config'

export interface CatalogStats {
  stars?: number
  downloads?: { python?: number; typescript?: number }
  lastRelease?: string
}

/** Shape of src/data/catalog-stats.json — keyed by entry id (filename without .yaml). */
export type CatalogStatsFile = Record<string, CatalogStats>

export interface CatalogCardModel {
  id: string
  name: string
  description: string
  integrationType: CatalogEntryData['integrationType']
  sdk: 'agents' | 'evals'
  languages: ('python' | 'typescript')[]
  href: string
  external: boolean
  github: string
  registryLinks: { label: 'PyPI' | 'npm'; href: string }[]
  maintainer: string
  featured: boolean
  badges: string[]
  stars?: number
  downloads?: number
}

/** Window (in days) during which an entry carries the derived "new" badge. */
export const NEW_BADGE_DAYS = 30

export function toCardModel(
  id: string,
  data: CatalogEntryData,
  stats: CatalogStats | undefined,
  buildDate: Date
): CatalogCardModel {
  const languages: ('python' | 'typescript')[] = []
  const registryLinks: CatalogCardModel['registryLinks'] = []
  if (data.languages.python) {
    languages.push('python')
    registryLinks.push({ label: 'PyPI', href: data.languages.python.registry })
  }
  if (data.languages.typescript) {
    languages.push('typescript')
    registryLinks.push({ label: 'npm', href: data.languages.typescript.registry })
  }

  const badges: string[] = [...data.badges]
  const ageDays = (buildDate.getTime() - data.addedDate.getTime()) / 86_400_000
  if (ageDays >= 0 && ageDays < NEW_BADGE_DAYS) badges.push('new')

  // Primary link priority: on-site docs page, then the integration's own
  // Strands instructions page, then the bare GitHub repo.
  const docsHref = data.docsPage ? `/${data.docsPage}/` : undefined

  const pythonDownloads = stats?.downloads?.python ?? 0
  const typescriptDownloads = stats?.downloads?.typescript ?? 0
  const totalDownloads = pythonDownloads + typescriptDownloads

  return {
    id,
    name: data.name,
    description: data.description,
    integrationType: data.integrationType,
    sdk: data.sdk,
    languages,
    href: docsHref ?? data.docsUrl ?? data.github,
    external: !docsHref,
    github: data.github,
    registryLinks,
    maintainer: data.maintainer,
    featured: data.featured,
    badges,
    ...(stats?.stars !== undefined && { stars: stats.stars }),
    ...(totalDownloads > 0 && { downloads: totalDownloads }),
  }
}

/** Featured entries first, then alphabetical by name. */
export function sortEntries(cards: CatalogCardModel[]): CatalogCardModel[] {
  return [...cards].sort((a, b) => {
    if (a.featured !== b.featured) return a.featured ? -1 : 1
    return a.name.localeCompare(b.name)
  })
}

/** The SDK facet renders only when the catalog actually has evals entries. */
export function hasEvalsEntries(cards: CatalogCardModel[]): boolean {
  return cards.some((c) => c.sdk === 'evals')
}
