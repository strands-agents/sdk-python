/**
 * Build-time static redirects for legacy MkDocs URLs.
 *
 * The site is deployed to GitHub Pages, so there are no server-side redirects.
 * The client-side fallback in Redirect404.astro works for humans, but crawlers
 * don't execute it — old URLs return HTTP 404 and backlink equity is lost.
 *
 * buildStaticRedirects() enumerates every *known* legacy URL at config time so
 * astro.config.mjs can feed them to Astro's `redirects` option, which emits a
 * static HTML stub per URL (meta refresh + canonical link) that crawlers do
 * follow. Sources come from:
 *
 *   1. STATIC_SLUG_REDIRECTS in redirect.ts (exact-match slug renames)
 *   2. `redirectFrom` frontmatter entries in src/content docs
 *
 * Dynamic rules — version-prefix stripping (/latest/, /1.x/) and
 * /documentation/docs/ normalization — can't be enumerated and remain handled
 * only by the client-side 404 fallback.
 *
 * This module runs at config time, before the content layer exists, so it
 * reads frontmatter from disk instead of using astro:content (unlike
 * redirect.build.ts, which serves the same map to the 404 page at build time).
 */

import fs from 'node:fs'
import path from 'node:path'
import fg from 'fast-glob'
import yaml from 'js-yaml'

import { STATIC_SLUG_REDIRECTS } from './redirect'
import { pathToDocsSlug } from './links'

/**
 * Extract frontmatter from a markdown/MDX file. Returns an empty object when
 * the file has no frontmatter block.
 *
 * This must stay compatible with the frontmatter parsing Astro's content
 * loader does (gray-matter): both treat the first `---`-fenced block as YAML.
 * A divergence would produce a wrong target slug here, which the build-time
 * "has no content file" validation below then catches.
 */
function readFrontmatter(filePath: string): Record<string, unknown> {
  const source = fs.readFileSync(filePath, 'utf-8')
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  if (!match) return {}
  const parsed = yaml.load(match[1] ?? '')
  return typeof parsed === 'object' && parsed !== null ? (parsed as Record<string, unknown>) : {}
}

/**
 * Scan the docs content directory once, producing the set of every real page
 * slug (for validation) and the old-slug → new-slug pairs declared in
 * `redirectFrom` frontmatter (same source of truth as redirect.build.ts, read
 * from disk because astro:content isn't available at config time).
 *
 * Slugs come from pathToDocsSlug — including frontmatter `slug` overrides —
 * so validation agrees with the ids the content collection actually uses,
 * rather than guessing filenames from slugs.
 */
function scanContent(contentDir: string): { slugs: Set<string>; redirectFromEntries: Record<string, string> } {
  const slugs = new Set<string>()
  const redirectFromEntries: Record<string, string> = {}

  const files = fg.sync('docs/**/*.{md,mdx}', {
    cwd: contentDir,
    followSymbolicLinks: false,
  })

  for (const relativePath of files) {
    const frontmatter = readFrontmatter(path.join(contentDir, relativePath))
    const target = pathToDocsSlug(relativePath, frontmatter.slug)
    slugs.add(target)

    const redirectFrom = frontmatter.redirectFrom
    if (!Array.isArray(redirectFrom)) continue

    for (const source of redirectFrom) {
      if (typeof source !== 'string') continue
      const existing = redirectFromEntries[source]
      if (existing !== undefined && existing !== target) {
        throw new Error(
          `[redirect.static] duplicate redirectFrom slug "${source}" points to both "${existing}" and "${target}"`
        )
      }
      redirectFromEntries[source] = target
    }
  }

  return { slugs, redirectFromEntries }
}

/** Format a slug as a root-relative URL path in the site's directory format. */
function toUrlPath(slug: string, base: string): string {
  const prefix = base.replace(/\/+$/, '')
  return `${prefix}/${slug.replace(/^\/+|\/+$/g, '')}/`
}

/**
 * Build the value for Astro's `redirects` config option: a map of
 * old URL path → new URL path (or external URL).
 *
 * @param contentDir - Absolute path to src/content, used to validate that no
 *   redirect source shadows a real page and every internal target exists
 * @param base - The site's base path (Astro `base` config), prepended to
 *   internal destinations so meta-refresh URLs work under a subpath deploy
 * @param staticRedirects - Exact-match rename rules; defaults to the production
 *   STATIC_SLUG_REDIRECTS. Injectable so unit tests aren't coupled to the
 *   production entries (adding a rename must not break the test suite).
 */
export function buildStaticRedirects(
  contentDir: string,
  base = '/',
  staticRedirects: Record<string, string> = STATIC_SLUG_REDIRECTS
): Record<string, string> {
  const { slugs, redirectFromEntries } = scanContent(contentDir)

  // A redirectFrom entry that collides with a rename rule but points elsewhere
  // is an authoring mistake — the rename rule would silently win. Same
  // fail-fast treatment as duplicate redirectFrom entries.
  for (const [source, target] of Object.entries(redirectFromEntries)) {
    const staticTarget = staticRedirects[source]
    if (staticTarget !== undefined && staticTarget !== target) {
      throw new Error(
        `[redirect.static] redirectFrom slug "${source}" conflicts with a static rename rule ` +
          `("${target}" vs "${staticTarget}")`
      )
    }
  }

  const slugMap: Record<string, string> = {
    ...redirectFromEntries,
    // Exact-match renames take priority, matching resolveRedirect's rule order
    ...staticRedirects,
  }

  const redirects: Record<string, string> = {}

  for (const [source, target] of Object.entries(slugMap)) {
    // A source that resolves to a real page would make Astro emit a redirect
    // stub on top of it (or fail the build) — always a configuration mistake.
    if (slugs.has(source)) {
      throw new Error(`[redirect.static] redirect source "${source}" collides with an existing content file`)
    }

    if (/^https?:\/\//.test(target)) {
      redirects[`/${source}`] = target
      continue
    }

    if (!slugs.has(target)) {
      throw new Error(`[redirect.static] redirect target "${target}" (from "${source}") has no content file`)
    }

    redirects[`/${source}`] = toUrlPath(target, base)
  }

  return redirects
}
