import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { rm, readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { randomUUID } from 'node:crypto'
import { LocalVectorSearchStrategy } from '../local-vector.js'

describe('LocalVectorSearchStrategy', () => {
  const fakeEmbedder = vi.fn()
  const mockStorage = { write: vi.fn(), read: vi.fn(), delete: vi.fn(), list: vi.fn() }
  let baseDir: string

  beforeEach(() => {
    vi.clearAllMocks()
    baseDir = join(tmpdir(), `strands-vector-test-${randomUUID()}`)
  })

  afterEach(async () => {
    await rm(baseDir, { recursive: true, force: true })
  })

  describe('index', () => {
    it('embeds content and writes a vector file', async () => {
      fakeEmbedder.mockResolvedValue([0.1, 0.2, 0.3])
      const strategy = new LocalVectorSearchStrategy({ embedder: fakeEmbedder }, baseDir)

      await strategy.index(mockStorage, 'notes/meeting.md', new TextEncoder().encode('discuss roadmap'))

      expect(fakeEmbedder).toHaveBeenCalledWith('discuss roadmap')
      const content = await readFile(join(baseDir, '.vectors/notes/meeting.md.vec.json'), 'utf-8')
      expect(JSON.parse(content)).toEqual([0.1, 0.2, 0.3])
    })
  })

  describe('search', () => {
    it('returns ranked results by cosine similarity', async () => {
      const strategy = new LocalVectorSearchStrategy({ embedder: fakeEmbedder }, baseDir)

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

    it('returns empty array when vector dir does not exist', async () => {
      fakeEmbedder.mockResolvedValue([0.1, 0.2])
      const strategy = new LocalVectorSearchStrategy({ embedder: fakeEmbedder }, baseDir)

      const results = await strategy.search(mockStorage, 'anything')

      expect(results).toEqual([])
    })

    it('respects maxResults', async () => {
      const strategy = new LocalVectorSearchStrategy({ embedder: fakeEmbedder, maxResults: 1 }, baseDir)

      fakeEmbedder.mockResolvedValueOnce([1, 0])
      await strategy.index(mockStorage, 'a.md', new TextEncoder().encode('a'))

      fakeEmbedder.mockResolvedValueOnce([0.9, 0.1])
      await strategy.index(mockStorage, 'b.md', new TextEncoder().encode('b'))

      fakeEmbedder.mockResolvedValueOnce([1, 0])
      const results = await strategy.search(mockStorage, 'query')

      expect(results).toHaveLength(1)
      expect(results[0]!.key).toBe('a.md')
    })

    it('uses custom vectorDir', async () => {
      fakeEmbedder.mockResolvedValue([0.5, 0.5])
      const customDir = join(baseDir, 'custom-vectors')
      const strategy = new LocalVectorSearchStrategy({ embedder: fakeEmbedder, vectorDir: customDir }, baseDir)

      await strategy.index(mockStorage, 'doc.md', new TextEncoder().encode('hello'))

      const content = await readFile(join(customDir, 'doc.md.vec.json'), 'utf-8')
      expect(JSON.parse(content)).toEqual([0.5, 0.5])
    })
  })
})
