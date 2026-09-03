import { describe, it, expect, vi, beforeEach } from 'vitest'
import { InMemoryVectorSearchStrategy } from '../in-memory-vector.js'

describe('InMemoryVectorSearchStrategy', () => {
  const fakeEmbedder = vi.fn()
  const mockStorage = { write: vi.fn(), read: vi.fn(), delete: vi.fn(), list: vi.fn() }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('index', () => {
    it('embeds content and stores the vector', async () => {
      fakeEmbedder.mockResolvedValue([0.1, 0.2, 0.3])
      const strategy = new InMemoryVectorSearchStrategy({ embedder: fakeEmbedder })

      await strategy.index(mockStorage, 'notes/meeting.md', new TextEncoder().encode('discuss roadmap'))

      expect(fakeEmbedder).toHaveBeenCalledWith('discuss roadmap')
    })
  })

  describe('search', () => {
    it('returns ranked results by cosine similarity', async () => {
      const strategy = new InMemoryVectorSearchStrategy({ embedder: fakeEmbedder })

      fakeEmbedder.mockResolvedValueOnce([1, 0, 0])
      await strategy.index(mockStorage, 'a.md', new TextEncoder().encode('a'))

      fakeEmbedder.mockResolvedValueOnce([0.9, 0.1, 0])
      await strategy.index(mockStorage, 'b.md', new TextEncoder().encode('b'))

      fakeEmbedder.mockResolvedValueOnce([0, 1, 0])
      await strategy.index(mockStorage, 'c.md', new TextEncoder().encode('c'))

      fakeEmbedder.mockResolvedValueOnce([1, 0, 0])
      const results = await strategy.search(mockStorage, 'query')

      expect(results[0]!.key).toBe('a.md')
      expect(results[0]!.score).toBeCloseTo(1.0)
      expect(results[1]!.key).toBe('b.md')
      expect(results[1]!.score).toBeGreaterThan(0)
      expect(results).toHaveLength(2)
    })

    it('returns empty array when no vectors stored', async () => {
      fakeEmbedder.mockResolvedValue([0.1, 0.2])
      const strategy = new InMemoryVectorSearchStrategy({ embedder: fakeEmbedder })

      const results = await strategy.search(mockStorage, 'anything')

      expect(results).toEqual([])
    })

    it('respects maxResults', async () => {
      const strategy = new InMemoryVectorSearchStrategy({ embedder: fakeEmbedder, maxResults: 1 })

      fakeEmbedder.mockResolvedValueOnce([1, 0])
      await strategy.index(mockStorage, 'a.md', new TextEncoder().encode('a'))

      fakeEmbedder.mockResolvedValueOnce([0.9, 0.1])
      await strategy.index(mockStorage, 'b.md', new TextEncoder().encode('b'))

      fakeEmbedder.mockResolvedValueOnce([1, 0])
      const results = await strategy.search(mockStorage, 'query')

      expect(results).toHaveLength(1)
      expect(results[0]!.key).toBe('a.md')
    })
  })
})
