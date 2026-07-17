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
})
