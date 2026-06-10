import { describe, it, expect } from 'vitest'
import { changelogEntrySchema, changelogFrontmatterSchema } from '../src/content.config'

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
