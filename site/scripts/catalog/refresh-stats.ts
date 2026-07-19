/**
 * Refreshes site/src/data/catalog-stats.json with GitHub stars, release dates,
 * and registry download counts for every catalog entry. Run by the
 * catalog-stats.yml scheduled workflow; per-source failures are logged and the
 * previous committed value is kept, so one broken upstream can neither block
 * the whole refresh nor regress the stats it can't fetch.
 *
 * Usage: npm run catalog:stats   (requires GITHUB_TOKEN for the GitHub API)
 */

import { readFileSync, readdirSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import yaml from 'js-yaml'

export interface StatsEntry {
  id: string
  github?: string
  python?: string
  typescript?: string
}

export interface StatsFetchers {
  githubRepo(repoUrl: string): Promise<{ stars: number; lastRelease?: string }>
  pypiDownloads(pkg: string): Promise<number>
  npmDownloads(pkg: string): Promise<number>
}

export interface EntryStats {
  stars?: number
  lastRelease?: string
  downloads: { python?: number; typescript?: number }
}

export async function buildStats(
  entries: StatsEntry[],
  fetchers: StatsFetchers,
  previous: Record<string, EntryStats> = {}
): Promise<Record<string, EntryStats>> {
  const result: Record<string, EntryStats> = {}
  for (const entry of entries) {
    const stats: EntryStats = { downloads: {} }
    const prev = previous[entry.id]
    if (entry.github) {
      try {
        const repo = await fetchers.githubRepo(entry.github)
        stats.stars = repo.stars
        if (repo.lastRelease) stats.lastRelease = repo.lastRelease
      } catch (err) {
        // Keep the previous values so a transient outage doesn't regress stats.
        console.warn(`entry=${entry.id}, source=github | fetch failed, keeping previous value`, err)
        if (prev?.stars !== undefined) stats.stars = prev.stars
        if (prev?.lastRelease !== undefined) stats.lastRelease = prev.lastRelease
      }
    }
    if (entry.python) {
      try {
        stats.downloads.python = await fetchers.pypiDownloads(entry.python)
      } catch (err) {
        console.warn(`entry=${entry.id}, source=pypi | fetch failed, keeping previous value`, err)
        if (prev?.downloads?.python !== undefined) stats.downloads.python = prev.downloads.python
      }
    }
    if (entry.typescript) {
      try {
        stats.downloads.typescript = await fetchers.npmDownloads(entry.typescript)
      } catch (err) {
        console.warn(`entry=${entry.id}, source=npm | fetch failed, keeping previous value`, err)
        if (prev?.downloads?.typescript !== undefined) stats.downloads.typescript = prev.downloads.typescript
      }
    }
    result[entry.id] = stats
  }
  return result
}

// ── Live fetchers ─────────────────────────────────────────────────────────────

function githubApiHeaders(): Record<string, string> {
  const headers: Record<string, string> = { accept: 'application/vnd.github+json' }
  if (process.env.GITHUB_TOKEN) headers.authorization = `Bearer ${process.env.GITHUB_TOKEN}`
  return headers
}

async function fetchJson(url: string, headers: Record<string, string> = {}): Promise<unknown> {
  const res = await fetch(url, { headers })
  if (!res.ok) throw new Error(`status=${res.status} url=${url}`)
  return res.json()
}

export const liveFetchers: StatsFetchers = {
  async githubRepo(repoUrl) {
    // Keep only the org/repo segments so tree/blob URLs resolve to the repo.
    const slug = new URL(repoUrl).pathname
      .replace(/^\/|\/$/g, '')
      .split('/')
      .slice(0, 2)
      .join('/')
    const repo = (await fetchJson(`https://api.github.com/repos/${slug}`, githubApiHeaders())) as {
      stargazers_count: number
    }
    let lastRelease: string | undefined
    try {
      const release = (await fetchJson(`https://api.github.com/repos/${slug}/releases/latest`, githubApiHeaders())) as {
        published_at?: string
      }
      lastRelease = release.published_at?.slice(0, 10)
    } catch {
      // Repos without releases 404 here; stars alone are still useful.
    }
    return { stars: repo.stargazers_count, lastRelease }
  },
  async pypiDownloads(pkg) {
    // pypistats.org: last-month downloads for the package.
    const data = (await fetchJson(`https://pypistats.org/api/packages/${pkg}/recent`)) as {
      data: { last_month: number }
    }
    return data.data.last_month
  },
  async npmDownloads(pkg) {
    const data = (await fetchJson(`https://api.npmjs.org/downloads/point/last-month/${encodeURIComponent(pkg)}`)) as {
      downloads: number
    }
    return data.downloads
  },
}

// ── CLI entry point ───────────────────────────────────────────────────────────

export function loadEntries(catalogDir: string): StatsEntry[] {
  const entries: StatsEntry[] = []
  for (const f of readdirSync(catalogDir)) {
    if (!f.endsWith('.yaml')) continue
    try {
      const data = yaml.load(readFileSync(path.join(catalogDir, f), 'utf-8')) as {
        github?: string
        languages?: { python?: { package: string }; typescript?: { package: string } }
      }
      if (!data?.github || typeof data.github !== 'string') {
        const id = f.replace(/\.yaml$/, '')
        console.warn(`entry=${id} | malformed yaml, skipping`)
        continue
      }
      const id = f.replace(/\.yaml$/, '')
      const languages = data?.languages ?? {}
      const entry: StatsEntry = { id }
      // Entries anchored to the SDK itself (no dedicated package or repo) must
      // not display the SDK's own downloads/stars as their popularity.
      const repoSlug = new URL(data.github).pathname.replace(/^\//, '')
      if (repoSlug.startsWith('strands-agents/')) {
        console.warn(`entry=${id}, source=github | repo is the sdk itself, skipping stats`)
      } else {
        entry.github = data.github
      }
      if (languages.python?.package === 'strands-agents') {
        console.warn(`entry=${id}, source=pypi | package is the sdk itself, skipping stats`)
      } else {
        entry.python = languages.python?.package
      }
      if (languages.typescript?.package === '@strands-agents/sdk') {
        console.warn(`entry=${id}, source=npm | package is the sdk itself, skipping stats`)
      } else {
        entry.typescript = languages.typescript?.package
      }
      entries.push(entry)
    } catch (err) {
      const id = f.replace(/\.yaml$/, '')
      console.warn(`entry=${id} | malformed yaml, skipping`, err)
    }
  }
  return entries
}

const isDirectRun = process.argv[1]?.endsWith('refresh-stats.ts')
if (isDirectRun) {
  const catalogDir = path.resolve('src/content/catalog')
  const outPath = path.resolve('src/data/catalog-stats.json')
  const entries = loadEntries(catalogDir)
  let previous: Record<string, EntryStats> = {}
  try {
    previous = JSON.parse(readFileSync(outPath, 'utf-8')) as Record<string, EntryStats>
  } catch {
    console.warn(`out=${outPath} | no previous stats file, starting fresh`)
  }
  console.log(`entries=${entries.length} | refreshing catalog stats`)
  const stats = await buildStats(entries, liveFetchers, previous)
  writeFileSync(outPath, JSON.stringify(stats, null, 2) + '\n')
  console.log(`out=${outPath} | catalog stats written`)
}
