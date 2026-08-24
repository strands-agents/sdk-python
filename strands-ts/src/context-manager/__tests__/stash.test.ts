import { describe, it, expect } from 'vitest'
import { Stash } from '../stash.js'
import { InMemoryStorage } from '../../storage/in-memory-storage.js'

describe('Stash', () => {
  describe('store and retrieve', () => {
    it('round-trips text content', async () => {
      const stash = new Stash(new InMemoryStorage())
      const content = new TextEncoder().encode('hello world')
      const ref = await stash.store('tool-123', 0, content, 'text/plain')

      expect(ref).toContain('tool-123')

      const result = await stash.retrieve(ref)
      expect(result).not.toBeNull()
      expect(result!.contentType).toBe('text/plain')
      expect(new TextDecoder().decode(result!.content)).toBe('hello world')
    })

    it('round-trips JSON content', async () => {
      const stash = new Stash(new InMemoryStorage())
      const json = JSON.stringify({ key: 'value', count: 42 })
      const content = new TextEncoder().encode(json)
      const ref = await stash.store('tool-456', 1, content, 'application/json')

      const result = await stash.retrieve(ref)
      expect(result).not.toBeNull()
      expect(result!.contentType).toBe('application/json')
      expect(new TextDecoder().decode(result!.content)).toBe(json)
    })

    it('returns null for unknown references', async () => {
      const stash = new Stash(new InMemoryStorage())
      const result = await stash.retrieve('nonexistent')
      expect(result).toBeNull()
    })

    it('generates unique references for multiple stores', async () => {
      const stash = new Stash(new InMemoryStorage())
      const content = new TextEncoder().encode('data')
      const ref1 = await stash.store('tool-1', 0, content, 'text/plain')
      const ref2 = await stash.store('tool-1', 0, content, 'text/plain')

      expect(ref1).not.toBe(ref2)
    })
  })

  describe('list', () => {
    it('lists all stored references', async () => {
      const stash = new Stash(new InMemoryStorage())
      const content = new TextEncoder().encode('data')
      const ref1 = await stash.store('tool-a', 0, content, 'text/plain')
      const ref2 = await stash.store('tool-b', 0, content, 'text/plain')

      const keys = await stash.list()
      expect(keys).toContain(ref1)
      expect(keys).toContain(ref2)
    })
  })

  describe('delete', () => {
    it('removes a stashed entry', async () => {
      const stash = new Stash(new InMemoryStorage())
      const content = new TextEncoder().encode('data')
      const ref = await stash.store('tool-x', 0, content, 'text/plain')

      await stash.delete(ref)
      const result = await stash.retrieve(ref)
      expect(result).toBeNull()
    })
  })

  describe('namespacing', () => {
    it('does not conflict with other storage users', async () => {
      const storage = new InMemoryStorage()
      const stash = new Stash(storage)

      const content = new TextEncoder().encode('stash data')
      await stash.store('tool-1', 0, content, 'text/plain')

      await storage.write('other-key', new TextEncoder().encode('other'))

      const topKeys = await storage.list('')
      expect(topKeys.some((key) => key.startsWith('context-stash/'))).toBe(true)
      expect(topKeys).toContain('other-key')
    })
  })
})
