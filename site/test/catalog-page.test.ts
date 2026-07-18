import { describe, it, expect } from 'vitest'
import { getCollection } from 'astro:content'
import { toCardModel, sortEntries } from '../src/util/catalog'

describe('catalog page data', () => {
  it('produces a card model for every collection entry', async () => {
    const entries = await getCollection('catalog')
    const cards = sortEntries(entries.map((e) => toCardModel(e.id, e.data, undefined, new Date())))
    expect(cards.length).toBe(entries.length)
    for (const card of cards) {
      expect(card.name.length).toBeGreaterThan(0)
      expect(card.href.length).toBeGreaterThan(0)
      expect(card.languages.length).toBeGreaterThan(0)
    }
  })
})
