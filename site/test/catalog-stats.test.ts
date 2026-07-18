import { describe, it, expect } from 'vitest'
import { buildStats, type StatsFetchers, type StatsEntry } from '../scripts/catalog/refresh-stats'

const entries: StatsEntry[] = [
  {
    id: 'strands-example',
    github: 'https://github.com/example/strands-example',
    python: 'strands-example',
    typescript: '@example/strands',
  },
  { id: 'strands-broken', github: 'https://github.com/example/strands-broken', python: 'strands-broken' },
]

function fetchers(overrides: Partial<StatsFetchers> = {}): StatsFetchers {
  return {
    githubRepo: async () => ({ stars: 42, lastRelease: '2026-07-01' }),
    pypiDownloads: async () => 100,
    npmDownloads: async () => 50,
    ...overrides,
  }
}

describe('buildStats', () => {
  it('aggregates stats per entry', async () => {
    const stats = await buildStats(entries, fetchers())
    expect(stats['strands-example']).toEqual({
      stars: 42,
      lastRelease: '2026-07-01',
      downloads: { python: 100, typescript: 50 },
    })
    expect(stats['strands-broken']).toEqual({ stars: 42, lastRelease: '2026-07-01', downloads: { python: 100 } })
  })

  it('skips failing sources without failing the run', async () => {
    const stats = await buildStats(
      entries,
      fetchers({
        githubRepo: async (repo) => {
          if (repo.includes('broken')) throw new Error('boom')
          return { stars: 42, lastRelease: '2026-07-01' }
        },
        pypiDownloads: async (pkg) => {
          if (pkg === 'strands-broken') throw new Error('boom')
          return 100
        },
      })
    )
    expect(stats['strands-example']?.stars).toBe(42)
    expect(stats['strands-broken']).toEqual({ downloads: {} })
  })
})
