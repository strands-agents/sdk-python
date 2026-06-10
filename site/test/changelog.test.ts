import { describe, it, expect } from 'vitest'
import { changelogEntrySchema, changelogFrontmatterSchema } from '../src/content.config'
import { getPackageUrl, SDK_META, LANGUAGE_META } from '../src/config/changelog'

describe('changelogEntrySchema', () => {
  it('parses a full entry', () => {
    const entry = changelogEntrySchema.parse({
      type: 'feat',
      breaking: false,
      scope: 'model',
      areas: ['model'],
      title: 'plumb Gemini cache tokens',
      pr: 2287,
      prUrl: 'https://github.com/strands-agents/harness-sdk/pull/2287',
      commit: 'a1b2c3d',
      commitUrl: 'https://github.com/strands-agents/harness-sdk/commit/a1b2c3d',
      author: 'yatszhash',
    })
    expect(entry.type).toBe('feat')
    expect(entry.areas).toEqual(['model'])
  })

  it('applies defaults for sparse entries', () => {
    const entry = changelogEntrySchema.parse({ type: 'fix', title: 'handle null' })
    expect(entry.breaking).toBe(false)
    expect(entry.areas).toEqual([])
    expect(entry.pr).toBeNull()
  })

  it('rejects unknown type', () => {
    expect(() => changelogEntrySchema.parse({ type: 'wat', title: 'x' })).toThrow()
  })
})

describe('changelogFrontmatterSchema', () => {
  it('parses a harness python release', () => {
    const fm = changelogFrontmatterSchema.parse({
      sdk: 'harness',
      language: 'python',
      version: '1.42.0',
      tag: 'python/v1.42.0',
      date: '2026-06-01',
      releaseUrl: 'https://github.com/strands-agents/harness-sdk/releases/tag/python%2Fv1.42.0',
      packageUrl: 'https://pypi.org/project/strands-agents/1.42.0/',
      entries: [{ type: 'feat', title: 'add Limits' }],
    })
    expect(fm.sdk).toBe('harness')
    expect(fm.date).toBeInstanceOf(Date)
  })

  it('allows evals release without language', () => {
    const fm = changelogFrontmatterSchema.parse({
      sdk: 'evals',
      version: '0.2.1',
      tag: 'v0.2.1',
      date: '2026-05-29',
      releaseUrl: 'https://github.com/strands-agents/evals/releases/tag/v0.2.1',
      packageUrl: 'https://pypi.org/project/strands-agents-evals/0.2.1/',
      entries: [],
    })
    expect(fm.language).toBeUndefined()
  })
})

describe('getPackageUrl', () => {
  it('builds a PyPI url for harness python', () => {
    expect(getPackageUrl('harness', 'python', '1.42.0')).toBe(
      'https://pypi.org/project/strands-agents/1.42.0/'
    )
  })
  it('builds an npm url for harness typescript', () => {
    expect(getPackageUrl('harness', 'typescript', '1.4.0')).toBe(
      'https://www.npmjs.com/package/@strands-agents/sdk/v/1.4.0'
    )
  })
  it('builds a PyPI url for evals', () => {
    expect(getPackageUrl('evals', undefined, '0.2.1')).toBe(
      'https://pypi.org/project/strands-agents-evals/0.2.1/'
    )
  })
})

describe('SDK_META', () => {
  it('lists languages per sdk', () => {
    expect(SDK_META.harness.languages).toEqual(['python', 'typescript'])
    expect(SDK_META.evals.languages).toEqual(['python'])
  })
})

import { groupEntries, getAreaCounts, formatChangelogDate } from '../src/util/changelog'
import type { ChangelogEntry } from '../src/content.config'

const mk = (over: Partial<ChangelogEntry>): ChangelogEntry => ({
  type: 'feat', breaking: false, scope: null, areas: [], title: 't',
  pr: null, prUrl: null, commit: null, commitUrl: null, author: null, ...over,
})

describe('groupEntries', () => {
  it('splits into features, fixes, other and keeps breaking in features', () => {
    const g = groupEntries([
      mk({ type: 'feat', title: 'a' }),
      mk({ type: 'breaking', title: 'b' }),
      mk({ type: 'fix', title: 'c' }),
      mk({ type: 'chore', title: 'd' }),
      mk({ type: 'docs', title: 'e' }),
    ])
    expect(g.features.map((e) => e.title)).toEqual(['b', 'a']) // breaking first
    expect(g.fixes.map((e) => e.title)).toEqual(['c'])
    expect(g.other.map((e) => e.title)).toEqual(['d', 'e'])
  })
})

describe('getAreaCounts', () => {
  it('counts entries per area, sorted desc', () => {
    const counts = getAreaCounts([
      mk({ areas: ['model'] }),
      mk({ areas: ['model', 'mcp'] }),
      mk({ areas: [] }),
    ])
    expect(counts).toEqual([
      { area: 'model', count: 2 },
      { area: 'mcp', count: 1 },
    ])
  })
})

describe('formatChangelogDate', () => {
  it('formats as a short date', () => {
    expect(formatChangelogDate(new Date('2026-06-01T00:00:00Z'))).toMatch(/Jun 1, 2026/)
  })
})
