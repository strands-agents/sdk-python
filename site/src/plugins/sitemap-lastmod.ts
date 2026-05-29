/**
 * Sitemap lastmod plugin
 *
 * Adds <lastmod> dates to the XML sitemap by looking up each page's
 * last git commit date. Runs a single `git log` command at config time
 * to build a map of file paths to ISO dates, then injects them via
 * the @astrojs/sitemap `serialize` callback.
 *
 * When Starlight sees @astrojs/sitemap already in the integrations
 * array, it skips adding its own — so this replaces the default
 * sitemap with one that includes lastmod.
 */
import { execFileSync } from 'node:child_process'
import path from 'node:path'
import sitemap from '@astrojs/sitemap'
import type { SitemapItem } from '@astrojs/sitemap'

/**
 * Build a map of content file path → last git commit date (ISO 8601).
 *
 * Uses a single `git log` call with --name-only to get all commits
 * touching the content directory, then picks the most recent date
 * for each file. This is much faster than running git log per file.
 */
function buildLastModMap(contentDir: string): Map<string, string> {
  const map = new Map<string, string>()

  try {
    // Single git command: output all commits with dates and affected files
    // Format: each commit outputs its ISO date, then the list of files
    const output = execFileSync(
      'git',
      ['log', '--format=%cI', '--name-only', '--diff-filter=ACMR', '--', contentDir],
      { encoding: 'utf-8', maxBuffer: 50 * 1024 * 1024 }
    )

    // Parse: lines alternate between date lines and file path lines,
    // separated by blank lines. First occurrence of each file = most recent.
    let currentDate = ''
    for (const line of output.split('\n')) {
      const trimmed = line.trim()
      if (!trimmed) continue

      // ISO date lines start with a digit (e.g., "2026-03-07T00:22:25-08:00")
      if (/^\d{4}-/.test(trimmed)) {
        currentDate = trimmed
      } else if (currentDate && !map.has(trimmed)) {
        // File path — only store if we haven't seen it yet (first = most recent)
        map.set(trimmed, currentDate)
      }
    }
  } catch (error) {
    // If git isn't available (e.g., Docker build without .git),
    // fall back gracefully — sitemap just won't have lastmod
    console.warn('[sitemap-lastmod] Failed to build lastmod map:', error instanceof Error ? error.message : error)
  }

  return map
}

/**
 * Convert a URL path to likely content file paths.
 * e.g., /docs/user-guide/quickstart/python/ → src/content/docs/user-guide/quickstart/python.mdx
 */
function urlToContentPaths(urlPath: string, contentDir: string): string[] {
  const slug = urlPath.replace(/^\//, '').replace(/\/$/, '')

  return [
    path.join(contentDir, `${slug}.mdx`),
    path.join(contentDir, `${slug}.md`),
    path.join(contentDir, slug, 'index.mdx'),
    path.join(contentDir, slug, 'index.md'),
  ]
}

export function sitemapWithLastmod(contentDir: string = 'src/content') {
  const lastModMap = buildLastModMap(contentDir)
  console.log(`[sitemap-lastmod] Loaded ${lastModMap.size} git dates`)

  return sitemap({
    serialize(item: SitemapItem) {
      const url = new URL(item.url)

      // Skip API reference pages — generated at build time, not git-tracked
      if (url.pathname.includes('/api/')) return item

      const candidates = urlToContentPaths(url.pathname, contentDir)

      for (const candidate of candidates) {
        const date = lastModMap.get(candidate)
        if (date) {
          item.lastmod = date
          break
        }
      }

      return item
    },
  })
}
