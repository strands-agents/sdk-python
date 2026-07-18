/**
 * Refreshes site/src/data/catalog-stats.json with GitHub stars, release dates,
 * and registry download counts for every catalog entry. Run by the
 * catalog-stats.yml scheduled workflow; per-package failures are logged and
 * skipped so one broken upstream can't block the whole refresh.
 *
 * Usage: npm run catalog:stats   (requires GITHUB_TOKEN for the GitHub API)
 */

import { readFileSync, readdirSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import yaml from 'js-yaml'

export interface StatsEntry {
  id: string
  github: string
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
  fetchers: StatsFetchers
): Promise<Record<string, EntryStats>> {
  const result: Record<string, EntryStats> = {}
  for (const entry of entries) {
    const stats: EntryStats = { downloads: {} }
    try {
      const repo = await fetchers.githubRepo(entry.github)
      stats.stars = repo.stars
      if (repo.lastRelease) stats.lastRelease = repo.lastRelease
    } catch (err) {
      console.warn(`entry=${entry.id}, source=github | fetch failed, skipping`, err)
    }
    if (entry.python) {
      try {
        stats.downloads.python = await fetchers.pypiDownloads(entry.python)
      } catch (err) {
        console.warn(`entry=${entry.id}, source=pypi | fetch failed, skipping`, err)
      }
    }
    if (entry.typescript) {
      try {
        stats.downloads.typescript = await fetchers.npmDownloads(entry.typescript)
      } catch (err) {
        console.warn(`entry=${entry.id}, source=npm | fetch failed, skipping`, err)
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
    const slug = new URL(repoUrl).pathname.replace(/^\/|\/$/g, '')
    const repo = (await fetchJson(`https://api.github.com/repos/${slug}`, githubApiHeaders())) as {
      stargazers_count: number
    }
    let lastRelease: string | undefined
    try {
      const release = (await fetchJson(
        `https://api.github.com/repos/${slug}/releases/latest`,
        githubApiHeaders()
      )) as { published_at?: string }
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
    const data = (await fetchJson(
      `https://api.npmjs.org/downloads/point/last-month/${encodeURIComponent(pkg)}`
    )) as { downloads: number }
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
      const languages = data?.languages ?? {}
      entries.push({
        id: f.replace(/\.yaml$/, ''),
        github: data.github,
        python: languages.python?.package,
        typescript: languages.typescript?.package,
      })
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
  console.log(`entries=${entries.length} | refreshing catalog stats`)
  const stats = await buildStats(entries, liveFetchers)
  writeFileSync(outPath, JSON.stringify(stats, null, 2) + '\n')
  console.log(`out=${outPath} | catalog stats written`)
}
