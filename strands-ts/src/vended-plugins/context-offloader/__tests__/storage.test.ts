import { describe, it, expect } from 'vitest'
import { InMemoryStorage } from '../storage.js'

describe('InMemoryStorage', () => {
  it('stores and retrieves text content', async () => {
    const storage = new InMemoryStorage()
    const content = new TextEncoder().encode('hello world')
    const ref = await storage.store('key1', content, 'text/plain')

    const result = await storage.retrieve(ref)
    expect(new TextDecoder().decode(result.content)).toBe('hello world')
    expect(result.contentType).toBe('text/plain')
  })

  it('stores and retrieves binary content', async () => {
    const storage = new InMemoryStorage()
    const content = new Uint8Array([1, 2, 3, 4, 5])
    const ref = await storage.store('key1', content, 'image/png')

    const result = await storage.retrieve(ref)
    expect(result.content).toEqual(content)
    expect(result.contentType).toBe('image/png')
  })

  it('generates unique references', async () => {
    const storage = new InMemoryStorage()
    const content = new TextEncoder().encode('test')
    const ref1 = await storage.store('key1', content)
    const ref2 = await storage.store('key2', content)
    expect(ref1).not.toBe(ref2)
  })

  it('uses mem_ prefix in references', async () => {
    const storage = new InMemoryStorage()
    const ref = await storage.store('mykey', new TextEncoder().encode('test'))
    expect(ref).toMatch(/^mem_\d+_mykey$/)
  })

  it('throws on missing reference', async () => {
    const storage = new InMemoryStorage()
    await expect(storage.retrieve('nonexistent')).rejects.toThrow('Reference not found: nonexistent')
  })

  it('clears all stored content', async () => {
    const storage = new InMemoryStorage()
    const ref = await storage.store('key1', new TextEncoder().encode('test'))
    storage.clear()
    await expect(storage.retrieve(ref)).rejects.toThrow('Reference not found')
  })

  it('defaults content type to text/plain', async () => {
    const storage = new InMemoryStorage()
    const ref = await storage.store('key1', new TextEncoder().encode('test'))
    const result = await storage.retrieve(ref)
    expect(result.contentType).toBe('text/plain')
  })
})

describe('InMemoryStorage eviction', () => {
  it('eviction enabled by default (20 cycles)', async () => {
    const storage = new InMemoryStorage()
    const ref = await storage.store('key1', new TextEncoder().encode('test'))
    storage._evict(21)
    await expect(storage.retrieve(ref)).rejects.toThrow('Reference not found')
  })

  it('eviction disabled with null', async () => {
    const storage = new InMemoryStorage(null)
    const ref = await storage.store('key1', new TextEncoder().encode('test'))
    storage._evict(100)
    const result = await storage.retrieve(ref)
    expect(new TextDecoder().decode(result.content)).toBe('test')
  })

  it('throws on invalid evictAfterTurns', () => {
    expect(() => new InMemoryStorage(0)).toThrow('evictAfterTurns must be a positive integer')
    expect(() => new InMemoryStorage(-1)).toThrow('evictAfterTurns must be a positive integer')
  })

  it('entry survives within TTL', async () => {
    const storage = new InMemoryStorage(3)
    const ref = await storage.store('key1', new TextEncoder().encode('test'))
    // stored at cycle 0, evict at cycle 3: threshold = 3 - 3 = 0, 0 < 0 is false
    storage._evict(3)
    const result = await storage.retrieve(ref)
    expect(new TextDecoder().decode(result.content)).toBe('test')
  })

  it('entry evicted past TTL', async () => {
    const storage = new InMemoryStorage(2)
    const ref = await storage.store('key1', new TextEncoder().encode('test'))
    // stored at cycle 0, evict at cycle 3: threshold = 3 - 2 = 1, 0 < 1 → evicted
    storage._evict(3)
    await expect(storage.retrieve(ref)).rejects.toThrow('Reference not found')
  })

  it('retrieve refreshes last accessed cycle', async () => {
    const storage = new InMemoryStorage(2)
    const ref = await storage.store('key1', new TextEncoder().encode('test'))
    storage._evict(1)
    await storage.retrieve(ref) // refreshes to cycle 1
    // threshold at cycle 3 = 3 - 2 = 1, last_accessed = 1, 1 < 1 is false
    storage._evict(3)
    const result = await storage.retrieve(ref)
    expect(new TextDecoder().decode(result.content)).toBe('test')
  })

  it('multiple entries evicted independently', async () => {
    const storage = new InMemoryStorage(2)
    const ref1 = await storage.store('key1', new TextEncoder().encode('first'))
    storage._evict(1)
    const ref2 = await storage.store('key2', new TextEncoder().encode('second'))
    // threshold at cycle 3 = 3 - 2 = 1. ref1 at 0 (evicted), ref2 at 1 (survives)
    storage._evict(3)
    await expect(storage.retrieve(ref1)).rejects.toThrow('Reference not found')
    const result = await storage.retrieve(ref2)
    expect(new TextDecoder().decode(result.content)).toBe('second')
  })

  it('rejects shared storage across agents', () => {
    const storage = new InMemoryStorage()
    const agentA = {}
    const agentB = {}
    storage._bind(agentA)
    expect(() => storage._bind(agentB)).toThrow('cannot be shared')
  })

  it('allows same agent repeated bind', () => {
    const storage = new InMemoryStorage()
    const agent = {}
    storage._bind(agent)
    storage._bind(agent)
  })
})
