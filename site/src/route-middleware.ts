import { defineRouteMiddleware, type StarlightRouteData } from '@astrojs/starlight/route-data'
import { getCollection } from 'astro:content'
import { buildPythonApiSidebar, buildTypeScriptApiSidebar, getPrevNextLinks, type DocInfo } from './dynamic-sidebar'
import { pathWithBase } from './util/links'
import { navLinks, type NavLink } from './config/navbar'

type SidebarEntry = StarlightRouteData['sidebar'][number]
type SidebarGroup = Extract<SidebarEntry, { type: 'group' }>

function isSidebarGroup(entry: SidebarEntry): entry is SidebarGroup {
  return entry.type === 'group'
}

/**
 * Find which nav section the current page belongs to based on URL path.
 * Matches the most specific basePath (longest match wins).
 * Also checks additionalBasePaths for nav items that span multiple sections.
 */
export function findCurrentNavSection(currentPath: string, links: NavLink[]): NavLink | undefined {
  let bestMatch: NavLink | undefined
  let bestMatchLength = 0

  for (const link of links) {
    if (link.external) continue
    const basePaths = link.basePath
      ? (Array.isArray(link.basePath) ? link.basePath : [link.basePath])
      : [link.href]
    for (const bp of basePaths) {
      if (currentPath.startsWith(bp) && bp.length > bestMatchLength) {
        bestMatch = link
        bestMatchLength = bp.length
      }
    }
  }

  return bestMatch
}

/**
 * Filter sidebar entries to only include items matching one or more base paths.
 * If the result is a single top-level group, unwrap it to return just its entries.
 */
export function filterSidebarByBasePath(entries: SidebarEntry[], basePath: string | string[]): SidebarEntry[] {
  const basePaths = Array.isArray(basePath) ? basePath : [basePath]

  const matchesAnyBase = (href: string) => basePaths.some((bp) => href.startsWith(bp))

  const filtered = entries
    .map((entry) => {
      if (entry.type === 'link') {
        return matchesAnyBase(entry.href) ? entry : null
      }
      if (entry.type === 'group') {
        const filteredEntries = filterSidebarByBasePath(entry.entries, basePaths)
        return filteredEntries.length > 0 ? { ...entry, entries: filteredEntries } : null
      }
      return null
    })
    .filter((entry): entry is SidebarEntry => entry !== null)

  // If we have a single top-level group, unwrap it to show its entries directly
  const firstEntry = filtered[0]
  if (filtered.length === 1 && firstEntry && isSidebarGroup(firstEntry)) {
    return firstEntry.entries
  }

  return filtered
}

/**
 * Apply collapse behavior to all sidebar groups.
 * Starlight normalizes all unset collapsed values to false before the middleware
 * runs, making collapsed: false indistinguishable from "not set". Only an explicit
 * collapsed: true in navigation.yml can override the depth-based default.
 */
export function applyCollapse(items: SidebarEntry[], depth: number = 0): SidebarEntry[] {
  return items.map((item) => {
    if (item.type === 'group') {
      const collapsed = item.collapsed === true ? true : depth >= 1
      return { ...item, collapsed, entries: applyCollapse(item.entries, depth + 1) }
    }
    return item
  })
}

/**
 * Route middleware that filters the sidebar to only show items
 * matching the current nav section based on URL path.
 *
 * Uses the navbar config's basePath to determine which section
 * the current page belongs to, then filters sidebar to only show
 * items whose href starts with that basePath.
 */
/**
 * Build a map of href -> page title from the docs collection.
 * Used to override sidebar labels with actual page titles in prev/next navigation.
 */
async function buildTitlesByHref(): Promise<Map<string, string>> {
  const docs = await getCollection('docs')
  const map = new Map<string, string>()
  for (const doc of docs) {
    if (doc.data.title) {
      map.set(pathWithBase(`/${doc.id}/`), doc.data.title as string)
    }
  }
  return map
}

export const onRequest = defineRouteMiddleware(async (context) => {
  const { starlightRoute } = context.locals
  const { sidebar } = starlightRoute
  // Sidebar hrefs include base path, so use full URL pathname for comparison
  const currentPath = context.url.pathname
  const currentSlug = starlightRoute.id

  // Check if we're on an API page (Python or TypeScript)
  if (currentSlug.startsWith('docs/api/python') || currentSlug.startsWith('docs/api/typescript')) {
    const docs = await getCollection('docs')
    const docInfos: DocInfo[] = docs.map((doc: { id: string; data: { title: unknown; category?: unknown } }) => ({
      id: doc.id,
      title: doc.data.title as string,
      category: doc.data.category as string | undefined,
    }))

    const isPython = currentSlug.startsWith('docs/api/python')
    const apiSidebar = isPython
      ? buildPythonApiSidebar(docInfos, currentSlug)
      : buildTypeScriptApiSidebar(docInfos, currentSlug)

    // Add index link at the top
    const overviewHref = isPython ? '/docs/api/python/' : '/docs/api/typescript/'
    const overviewSlug = isPython ? 'docs/api/python' : 'docs/api/typescript'
    apiSidebar.unshift({
      type: 'link',
      label: 'Overview',
      href: pathWithBase(overviewHref),
      isCurrent: currentSlug === overviewSlug,
      badge: undefined,
      attrs: {},
    })

    const titlesByHref = await buildTitlesByHref()
    starlightRoute.sidebar = apiSidebar
    starlightRoute.pagination = getPrevNextLinks(apiSidebar, titlesByHref)
    return
  }

  // Find which nav section the current page belongs to
  const currentNav = findCurrentNavSection(currentPath, navLinks)

  // If no matching nav section, show empty sidebar
  if (!currentNav || currentNav.label == "Home") {
    starlightRoute.sidebar = []
    return
  }

  // Collect all base paths for this nav section
  const bp = currentNav.basePath || currentNav.href
  const allBasePaths = Array.isArray(bp) ? bp : [bp]

  // Otherwise filter it down to the major section that we're in
  const filteredSidebar = filterSidebarByBasePath(sidebar, allBasePaths)
  starlightRoute.sidebar = applyCollapse(filteredSidebar)

  // Starlight pre-computes pagination from the full sidebar before our middleware runs.
  // Prune any prev/next links that fall outside the current nav section, then override
  // labels with actual page titles instead of sidebar nav labels.
  const matchesAnyBase = (href: string) => allBasePaths.some((bp) => href.startsWith(bp))
  const titlesByHref = await buildTitlesByHref()
  const { prev, next } = starlightRoute.pagination
  starlightRoute.pagination = {
    prev: prev && matchesAnyBase(prev.href) ? { ...prev, label: titlesByHref.get(prev.href) ?? prev.label } : undefined,
    next: next && matchesAnyBase(next.href) ? { ...next, label: titlesByHref.get(next.href) ?? next.label } : undefined,
  }
})
