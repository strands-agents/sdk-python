import { describe, it, expect, beforeEach } from 'vitest'
import { InMemoryStorage } from '../in-memory-storage.js'
import { StorageError } from '../../errors.js'

describe('InMemoryStorage', () => {
  let storage: InMemoryStorage

  beforeEach(() => {
    storage = new InMemoryStorage({ maxEntries: null })
  })

  describe('put', () => {
    it('stores data under the given key', async () => {
      const data = new TextEncoder().encode('hello')
      await storage.put('test/key', data)
      const result = await storage.get('test/key')
      expect(result).toEqual(data)
    })

    it('overwrites existing data', async () => {
      await storage.put('key', new TextEncoder().encode('first'))
      await storage.put('key', new TextEncoder().encode('second'))
      const result = await storage.get('key')
      expect(new TextDecoder().decode(result!)).toBe('second')
    })

    it('copies bytes on put to prevent aliasing', async () => {
      const data = new Uint8Array([1, 2, 3])
      await storage.put('key', data)
      data[0] = 99
      const result = await storage.get('key')
      expect(result![0]).toBe(1)
    })
  })

  describe('get', () => {
    it('returns null for missing keys', async () => {
      const result = await storage.get('nonexistent')
      expect(result).toBeNull()
    })

    it('copies bytes on get to prevent aliasing', async () => {
      await storage.put('key', new Uint8Array([1, 2, 3]))
      const first = await storage.get('key')
      first![0] = 99
      const second = await storage.get('key')
      expect(second![0]).toBe(1)
    })
  })

  describe('delete', () => {
    it('removes an existing key', async () => {
      await storage.put('key', new Uint8Array([1]))
      await storage.delete('key')
      const result = await storage.get('key')
      expect(result).toBeNull()
    })

    it('is a no-op for missing keys', async () => {
      await expect(storage.delete('nonexistent')).resolves.toBeUndefined()
    })
  })

  describe('list', () => {
    it('returns keys matching a prefix', async () => {
      await storage.put('sessions/a/data', new Uint8Array([1]))
      await storage.put('sessions/b/data', new Uint8Array([2]))
      await storage.put('memory/notes', new Uint8Array([3]))

      const keys = await storage.list('sessions/')
      expect(keys).toEqual(['sessions/a/data', 'sessions/b/data'])
    })

    it('returns all keys when prefix is empty', async () => {
      await storage.put('a', new Uint8Array([1]))
      await storage.put('b', new Uint8Array([2]))

      const keys = await storage.list('')
      expect(keys).toEqual(['a', 'b'])
    })

    it('returns keys sorted lexicographically', async () => {
      await storage.put('c', new Uint8Array([3]))
      await storage.put('a', new Uint8Array([1]))
      await storage.put('b', new Uint8Array([2]))

      const keys = await storage.list('')
      expect(keys).toEqual(['a', 'b', 'c'])
    })

    it('returns empty array when no keys match', async () => {
      await storage.put('other/key', new Uint8Array([1]))
      const keys = await storage.list('sessions/')
      expect(keys).toEqual([])
    })
  })

  describe('clear', () => {
    it('removes all entries', async () => {
      await storage.put('a', new Uint8Array([1]))
      await storage.put('b', new Uint8Array([2]))
      storage.clear()
      const keys = await storage.list('')
      expect(keys).toEqual([])
    })
  })

  describe('LRU eviction', () => {
    it('evicts the least-recently-used entry when maxEntries is exceeded', async () => {
      const bounded = new InMemoryStorage({ maxEntries: 2 })
      await bounded.put('a', new Uint8Array([1]))
      await bounded.put('b', new Uint8Array([2]))
      await bounded.put('c', new Uint8Array([3]))

      expect(await bounded.get('a')).toBeNull()
      expect(await bounded.get('b')).not.toBeNull()
      expect(await bounded.get('c')).not.toBeNull()
    })

    it('get promotes an entry so it is not evicted next', async () => {
      const bounded = new InMemoryStorage({ maxEntries: 2 })
      await bounded.put('a', new Uint8Array([1]))
      await bounded.put('b', new Uint8Array([2]))
      await bounded.get('a')
      await bounded.put('c', new Uint8Array([3]))

      expect(await bounded.get('a')).not.toBeNull()
      expect(await bounded.get('b')).toBeNull()
      expect(await bounded.get('c')).not.toBeNull()
    })

    it('works with maxEntries: 1', async () => {
      const bounded = new InMemoryStorage({ maxEntries: 1 })
      await bounded.put('a', new Uint8Array([1]))
      await bounded.put('b', new Uint8Array([2]))

      expect(await bounded.get('a')).toBeNull()
      expect(await bounded.get('b')).not.toBeNull()
    })

    it('overwriting an existing key does not trigger eviction', async () => {
      const bounded = new InMemoryStorage({ maxEntries: 2 })
      await bounded.put('a', new Uint8Array([1]))
      await bounded.put('b', new Uint8Array([2]))
      await bounded.put('a', new Uint8Array([99]))

      expect(await bounded.get('a')).toEqual(new Uint8Array([99]))
      expect(await bounded.get('b')).not.toBeNull()
    })

    it('does not evict when maxEntries is null', async () => {
      const unbounded = new InMemoryStorage({ maxEntries: null })
      for (let i = 0; i < 1000; i++) {
        await unbounded.put(`key-${i}`, new Uint8Array([i]))
      }
      const keys = await unbounded.list('')
      expect(keys).toHaveLength(1000)
    })

    it('rejects maxEntries less than 1', () => {
      expect(() => new InMemoryStorage({ maxEntries: 0 })).toThrow()
      expect(() => new InMemoryStorage({ maxEntries: -1 })).toThrow()
    })

    it('rejects non-integer maxEntries', () => {
      expect(() => new InMemoryStorage({ maxEntries: 1.5 })).toThrow()
      expect(() => new InMemoryStorage({ maxEntries: 0.9 })).toThrow()
    })
  })

  describe('key normalization', () => {
    it('normalizes slashes so equivalent keys resolve to the same entry', async () => {
      await storage.put('/a//b/', new Uint8Array([1]))
      const result = await storage.get('a/b')
      expect(result).toEqual(new Uint8Array([1]))
    })

    it('rejects empty keys', async () => {
      await expect(storage.put('', new Uint8Array([1]))).rejects.toThrow(StorageError)
    })

    it('rejects keys with .. segments', async () => {
      await expect(storage.put('a/../b', new Uint8Array([1]))).rejects.toThrow(StorageError)
    })

    it('rejects prefixes with .. segments', async () => {
      await expect(storage.list('../')).rejects.toThrow(StorageError)
    })
  })
})
