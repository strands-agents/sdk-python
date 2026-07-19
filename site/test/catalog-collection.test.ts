import { describe, it, expect } from 'vitest'
import { getCollection } from 'astro:content'
import { catalogEntrySchema } from '../src/content.config'

describe('catalog content collection', () => {
  it('loads catalog entries with validated data', async () => {
    const entries = await getCollection('catalog')
    expect(entries.length).toBeGreaterThan(0)

    const deepgram = entries.find((e) => e.id === 'strands-deepgram')
    expect(deepgram).toBeDefined()
    expect(deepgram!.data.name).toBe('strands-deepgram')
    expect(deepgram!.data.integrationType).toBe('tool')
    expect(deepgram!.data.sdk).toBe('agents')
    expect(deepgram!.data.languages.python?.package).toBe('strands-deepgram')
    expect(deepgram!.data.languages.typescript).toBeUndefined()
    expect(deepgram!.data.featured).toBe(false)
    expect(deepgram!.data.badges).toEqual([])
  })

  it('rejects an entry with no language blocks', () => {
    const result = catalogEntrySchema.safeParse({
      name: 'bad-entry',
      description: 'missing languages',
      integrationType: 'tool',
      languages: {},
      github: 'https://github.com/example/bad-entry',
      maintainer: 'example',
      addedDate: '2026-07-17',
    })
    expect(result.success).toBe(false)
  })

  it('rejects urls outside the expected hosts', () => {
    const base = {
      name: 'bad-urls',
      description: 'url smuggling',
      integrationType: 'tool',
      languages: { python: { package: 'x', registry: 'https://pypi.org/project/x/' } },
      github: 'https://github.com/example/x',
      maintainer: 'example',
      addedDate: '2026-07-17',
    }
    // javascript: scheme in github
    expect(catalogEntrySchema.safeParse({ ...base, github: 'javascript:alert(1)' }).success).toBe(false)
    // https but wrong host
    expect(catalogEntrySchema.safeParse({ ...base, github: 'https://evil.example/x' }).success).toBe(false)
    // registry on the wrong host for the language
    expect(
      catalogEntrySchema.safeParse({
        ...base,
        languages: { python: { package: 'x', registry: 'https://www.npmjs.com/package/x' } },
      }).success
    ).toBe(false)
    expect(
      catalogEntrySchema.safeParse({
        ...base,
        languages: { typescript: { package: 'x', registry: 'https://pypi.org/project/x/' } },
      }).success
    ).toBe(false)
  })

  it('rejects an unknown integrationType', () => {
    const result = catalogEntrySchema.safeParse({
      name: 'bad-type',
      description: 'bad type',
      integrationType: 'widget',
      languages: { python: { package: 'x', registry: 'https://pypi.org/project/x/' } },
      github: 'https://github.com/example/x',
      maintainer: 'example',
      addedDate: '2026-07-17',
    })
    expect(result.success).toBe(false)
  })

  it('every docsPage points at a real docs collection entry', async () => {
    const [entries, docs] = await Promise.all([getCollection('catalog'), getCollection('docs')])
    const docIds = new Set(docs.map((d) => d.id))
    for (const e of entries) {
      if (e.data.docsPage) {
        expect(docIds.has(e.data.docsPage), `catalog entry ${e.id} docsPage ${e.data.docsPage} not found in docs`).toBe(true)
      }
    }
  })

  it('every community docs page with an integrationType has a catalog entry', async () => {
    const [entries, docs] = await Promise.all([getCollection('catalog'), getCollection('docs')])
    const cataloged = new Set(entries.map((e) => e.data.docsPage).filter(Boolean))
    const communityPages = docs.filter(
      (d) => d.id.startsWith('docs/community/') && d.data.community === true && d.data.integrationType
    )
    for (const page of communityPages) {
      expect(cataloged.has(page.id), `community page ${page.id} has no catalog entry`).toBe(true)
    }
  })
})
