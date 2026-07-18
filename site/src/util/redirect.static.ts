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
import { slug as githubSlug } from 'github-slugger'

import { STATIC_SLUG_REDIRECTS } from './redirect'
import { normalizePathToSlug } from './links'

/**
 * Extract frontmatter from a markdown/MDX file. Returns an empty object when
 * the file has no frontmatter block.
 */
function readFrontmatter(filePath: string): Record<string, unknown> {
  const source = fs.readFileSync(filePath, 'utf-8')
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  if (!match) return {}
  const parsed = yaml.load(match[1] ?? '')
  return typeof parsed === 'object' && parsed !== null ? (parsed as Record<string, unknown>) : {}
}

/**
 * Derive the content-collection ID (slug) for a doc file, mirroring
 * generateDocsId in content.config.ts: an explicit frontmatter `slug` wins,
 * otherwise the path is normalised and each segment slugified.
 */
function docIdFor(relativePath: string, frontmatter: Record<string, unknown>): string {
  if (typeof frontmatter.slug === 'string') return frontmatter.slug

  const normalized = normalizePathToSlug(relativePath)
  if (!normalized) return 'index'

  return normalized
    .split('/')
    .map((segment) => githubSlug(segment))
    .join('/')
}

/**
 * Check whether a slug resolves to a real content file — used to reject
 * redirect sources that would shadow an existing page. Mirrors the candidate
 * list used by sidebar.ts, plus README variants (see normalizePathToSlug).
 */
function contentExists(slug: string, contentDir: string): boolean {
  const candidates = [
    path.join(contentDir, `${slug}.md`),
    path.join(contentDir, `${slug}.mdx`),
    path.join(contentDir, slug, 'index.md'),
    path.join(contentDir, slug, 'index.mdx'),
    path.join(contentDir, slug, 'README.md'),
    path.join(contentDir, slug, 'README.mdx'),
  ]
  return candidates.some((p) => fs.existsSync(p))
}

/**
 * Collect old-slug → new-slug pairs from `redirectFrom` frontmatter across the
 * docs content directory (same source of truth as redirect.build.ts, read from
 * disk because astro:content isn't available at config time).
 */
function collectRedirectFromEntries(contentDir: string): Record<string, string> {
  const entries: Record<string, string> = {}

  const files = fg.sync('docs/**/*.{md,mdx}', {
    cwd: contentDir,
    followSymbolicLinks: false,
  })

  for (const relativePath of files) {
    const frontmatter = readFrontmatter(path.join(contentDir, relativePath))
    const redirectFrom = frontmatter.redirectFrom
    if (!Array.isArray(redirectFrom)) continue

    const target = docIdFor(relativePath, frontmatter)
    for (const source of redirectFrom) {
      if (typeof source !== 'string') continue
      const existing = entries[source]
      if (existing !== undefined && existing !== target) {
        throw new Error(
          `[redirect.static] duplicate redirectFrom slug "${source}" points to both "${existing}" and "${target}"`
        )
      }
      entries[source] = target
    }
  }

  return entries
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
 */
export function buildStaticRedirects(contentDir: string, base = '/'): Record<string, string> {
  const slugMap: Record<string, string> = {
    ...collectRedirectFromEntries(contentDir),
    // Exact-match renames take priority, matching resolveRedirect's rule order
    ...STATIC_SLUG_REDIRECTS,
  }

  const redirects: Record<string, string> = {}

  for (const [source, target] of Object.entries(slugMap)) {
    // A source that resolves to a real page would make Astro emit a redirect
    // stub on top of it (or fail the build) — always a configuration mistake.
    if (contentExists(source, contentDir)) {
      throw new Error(`[redirect.static] redirect source "${source}" collides with an existing content file`)
    }

    if (/^https?:\/\//.test(target)) {
      redirects[`/${source}`] = target
      continue
    }

    if (!contentExists(target, contentDir)) {
      throw new Error(`[redirect.static] redirect target "${target}" (from "${source}") has no content file`)
    }

    redirects[`/${source}`] = toUrlPath(target, base)
  }

  return redirects
}
