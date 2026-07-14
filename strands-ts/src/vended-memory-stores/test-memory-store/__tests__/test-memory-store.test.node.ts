import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { TestMemoryStore } from '../store.js'
import { InMemoryStorage } from '../../../storage/in-memory-storage.js'
import { LocalFileStorage } from '../../../storage/local-file-storage.js'
import { StorageError } from '../../../errors.js'
import { MemoryManager } from '../../../memory/index.js'
import { InvocationTrigger } from '../../../memory/extraction/triggers.js'
import type { Extractor } from '../../../memory/extraction/types.js'
import { Message, TextBlock } from '../../../types/messages.js'
import { AfterInvocationEvent, MessageAddedEvent } from '../../../hooks/events.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'

// Persistence tests point a LocalFileStorage at a unique temp dir, never the real ./.strands or
// ~/.strands, so tests never touch a developer's actual memory.
let dir: string
let counter = 0

beforeEach(async () => {
  dir = join(tmpdir(), `strands-test-memory-${process.pid}-${counter++}`)
  await fs.mkdir(dir, { recursive: true })
})

afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

/** The on-disk location a `LocalFileStorage(dir)` writes for a store's `memory/<name>.json` key. */
function storageFilePath(name = 'notes'): string {
  return join(dir, 'memory', `${name}.json`)
}

describe('TestMemoryStore', () => {
  describe('constructor', () => {
    it('is writable by default', () => {
      expect(new TestMemoryStore({ name: 'notes' }).writable).toBe(true)
    })

    it('honors an explicit writable: false', () => {
      expect(new TestMemoryStore({ name: 'notes', writable: false }).writable).toBe(false)
    })

    it('exposes name, description, and maxSearchResults', () => {
      const store = new TestMemoryStore({ name: 'notes', description: 'my notes', maxSearchResults: 7 })
      expect(store.name).toBe('notes')
      expect(store.description).toBe('my notes')
      expect(store.maxSearchResults).toBe(7)
    })

    it('throws when maxSearchResults is less than 1', () => {
      expect(() => new TestMemoryStore({ name: 'notes', maxSearchResults: 0 })).toThrow(
        'maxSearchResults must be at least 1'
      )
    })

    it('throws when name is empty', () => {
      expect(() => new TestMemoryStore({ name: '   ' })).toThrow('name must not be empty')
    })

    it('does no filesystem I/O on construction', async () => {
      new TestMemoryStore({ name: 'notes', storage: new LocalFileStorage(dir) })
      await expect(fs.access(storageFilePath())).rejects.toThrow()
    })
  })

  describe('add', () => {
    it('throws when the store is not writable', async () => {
      const store = new TestMemoryStore({ name: 'notes', writable: false, storage: new InMemoryStorage() })
      await expect(store.add('fact')).rejects.toThrow('store is not writable')
    })

    it('throws on empty or whitespace content', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new InMemoryStorage() })
      await expect(store.add('   ')).rejects.toThrow('content must not be empty')
    })

    it('returns an id for stored content', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new InMemoryStorage() })
      const { id } = await store.add('user prefers dark mode')
      expect(id).toBeTruthy()
    })

    it('deduplicates identical content, returning the existing id', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new InMemoryStorage() })
      const first = await store.add('user prefers dark mode')
      const second = await store.add('user prefers dark mode')
      expect(second.id).toBe(first.id)
      const results = await store.search('dark mode preferences')
      expect(results).toHaveLength(1)
    })

    it('persists a human-readable JSON array through storage', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new LocalFileStorage(dir) })
      await store.add('user prefers dark mode', { source: 'user' })
      const raw = await fs.readFile(storageFilePath(), 'utf8')
      expect(raw).toContain('\n  ') // pretty-printed
      const parsed = JSON.parse(raw)
      expect(parsed).toHaveLength(1)
      expect(parsed[0]).toMatchObject({ content: 'user prefers dark mode', metadata: { source: 'user' } })
      expect(parsed[0].id).toBeTruthy()
      expect(parsed[0].createdAt).toBeTruthy()
    })
  })

  describe('search', () => {
    it('throws when maxSearchResults is less than 1', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new InMemoryStorage() })
      await expect(store.search('q', { maxSearchResults: 0 })).rejects.toThrow('maxSearchResults must be at least 1')
    })

    it('returns no results for an empty or token-less query', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new InMemoryStorage() })
      await store.add('user prefers dark mode')
      expect(await store.search('')).toEqual([])
      expect(await store.search('   ...  ')).toEqual([])
    })

    it('ranks higher token overlap first and exposes _relevanceScore', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new InMemoryStorage() })
      await store.add('the cat sat on the mat')
      await store.add('the cat chased the dog in the park')

      const results = await store.search('cat dog park')
      expect(results).toEqual([
        { content: 'the cat chased the dog in the park', metadata: { _relevanceScore: 3 } },
        { content: 'the cat sat on the mat', metadata: { _relevanceScore: 1 } },
      ])
    })

    it('excludes records with no token overlap', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new InMemoryStorage() })
      await store.add('the cat sat on the mat')
      await store.add('a completely unrelated note')

      const results = await store.search('cat')
      expect(results).toHaveLength(1)
      expect(results[0]?.content).toBe('the cat sat on the mat')
    })

    it('breaks ties by recency (most recent first)', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new InMemoryStorage() })
      // Stamp deterministic, increasing timestamps so the tie-break is testable.
      const isoSpy = vi
        .spyOn(Date.prototype, 'toISOString')
        .mockReturnValueOnce('2026-01-01T00:00:00.000Z')
        .mockReturnValueOnce('2026-01-02T00:00:00.000Z')
      await store.add('coffee is great')
      await store.add('coffee is bitter')
      isoSpy.mockRestore()

      const results = await store.search('coffee')
      expect(results[0]?.content).toBe('coffee is bitter')
      expect(results[1]?.content).toBe('coffee is great')
    })

    it('caps results to maxSearchResults', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new InMemoryStorage() })
      await store.add('alpha match')
      await store.add('beta match')
      await store.add('gamma match')
      const results = await store.search('match', { maxSearchResults: 2 })
      expect(results).toHaveLength(2)
    })

    it('tokenizes non-ASCII content as whole words (Unicode-aware, matching Python)', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new InMemoryStorage() })
      await store.add('the café in 日本 is naïve')
      expect(await store.search('café')).toHaveLength(1)
      expect(await store.search('日本')).toHaveLength(1)
    })
  })

  describe('persistence', () => {
    it('recalls entries from a fresh instance sharing the same storage (survives restart)', async () => {
      const storage = new LocalFileStorage(dir)
      const first = new TestMemoryStore({ name: 'notes', storage })
      await first.add('user lives in Berlin')

      const second = new TestMemoryStore({ name: 'notes', storage: new LocalFileStorage(dir) })
      const results = await second.search('where does the user live')
      expect(results).toHaveLength(1)
      expect(results[0]?.content).toBe('user lives in Berlin')
    })

    it('defaults to a per-instance ephemeral InMemoryStorage: a fresh instance forgets', async () => {
      const first = new TestMemoryStore({ name: 'notes' })
      await first.add('only in first')

      const second = new TestMemoryStore({ name: 'notes' })
      expect(await second.search('only in first')).toEqual([])
      expect(await first.search('only in first')).toHaveLength(1)
    })

    it('starts empty when the backing store is missing', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new LocalFileStorage(dir) })
      expect(await store.search('anything')).toEqual([])
    })

    it('sanitizes an unsafe name into a single safe storage key', async () => {
      const store = new TestMemoryStore({ name: '../weird/name', storage: new LocalFileStorage(dir) })
      await store.add('a fact worth keeping')
      await expect(fs.access(join(dir, 'memory', '__weird_name.json'))).resolves.toBeUndefined()
    })

    it('throws a clear error on a corrupt backing store', async () => {
      const storage = new LocalFileStorage(dir)
      await storage.write('memory/notes.json', new TextEncoder().encode('not json{'))
      const store = new TestMemoryStore({ name: 'notes', storage })
      await expect(store.search('anything')).rejects.toThrow('invalid JSON')
    })

    it('throws a clear error on a valid-but-wrong-shape backing store', async () => {
      const storage = new LocalFileStorage(dir)
      await storage.write('memory/notes.json', new TextEncoder().encode('{}'))
      const store = new TestMemoryStore({ name: 'notes', storage })
      await expect(store.search('anything')).rejects.toThrow('expected a JSON array')
    })

    it('throws a clear error on a malformed record', async () => {
      const storage = new LocalFileStorage(dir)
      await storage.write('memory/notes.json', new TextEncoder().encode(JSON.stringify([{ foo: 'bar' }])))
      const store = new TestMemoryStore({ name: 'notes', storage })
      await expect(store.search('anything')).rejects.toThrow('each record must have string')
    })

    it('surfaces the backend StorageError when the backing store is unreachable', async () => {
      // Point the backend's base dir at an existing FILE, so writes under it hit ENOTDIR — the
      // backend raises a StorageError naming the key rather than a bare OS error.
      const blocker = join(dir, 'blocker')
      await fs.writeFile(blocker, 'not a directory', 'utf8')
      const store = new TestMemoryStore({ name: 'notes', storage: new LocalFileStorage(blocker) })
      await expect(store.add('user prefers dark mode')).rejects.toThrow(StorageError)
      await expect(store.add('user prefers dark mode')).rejects.toThrow('Failed to write')
    })

    it('keeps all entries when many writes happen concurrently', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new LocalFileStorage(dir) })
      await Promise.all(Array.from({ length: 10 }, (_, index) => store.add(`fact number ${index}`)))
      const raw = await fs.readFile(storageFilePath(), 'utf8')
      expect(JSON.parse(raw)).toHaveLength(10)
    })

    it('reflects a just-added record when a search races the first add (no stale cache)', async () => {
      // search() and add() both read fresh from storage; with no in-memory cache, a search that
      // races the first add can never leave a subsequent search reading a pre-write snapshot.
      const store = new TestMemoryStore({ name: 'notes', storage: new LocalFileStorage(dir) })
      await Promise.all([store.add('zebra giraffe'), store.search('apple')])
      const results = await store.search('zebra')
      expect(results).toHaveLength(1)
      expect(results[0]?.content).toBe('zebra giraffe')
    })
  })

  describe('cross-SDK interop', () => {
    it('loads a record written in the shared camelCase format', async () => {
      // A record shaped exactly as the Python SDK writes it: camelCase keys and a millisecond,
      // Z-suffixed timestamp. The TS store must read it without translation.
      const pyWritten = [
        {
          id: '019ed65a-fd27-746c-aa3a-693a4a5434df',
          content: 'the user prefers dark mode',
          metadata: { source: 'py' },
          createdAt: '2026-01-02T00:00:00.000Z',
        },
      ]
      const storage = new LocalFileStorage(dir)
      await storage.write('memory/notes.json', new TextEncoder().encode(JSON.stringify(pyWritten)))

      const store = new TestMemoryStore({ name: 'notes', storage })
      const results = await store.search('dark mode preference')
      expect(results).toHaveLength(1)
      expect(results[0]?.content).toBe('the user prefers dark mode')
      expect(results[0]?.metadata?.source).toBe('py')
      expect(results[0]?.metadata?._relevanceScore).toBe(2)
    })

    it('writes the shared camelCase format with a Z-suffixed millisecond timestamp', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new LocalFileStorage(dir) })
      await store.add('hello world')
      const record = JSON.parse(await fs.readFile(storageFilePath(), 'utf8'))[0]
      expect(Object.keys(record).sort()).toEqual(['content', 'createdAt', 'id'])
      expect(record.createdAt).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/)
    })
  })

  describe('MemoryManager integration', () => {
    it('stamps storeName on results returned through the manager', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new InMemoryStorage() })
      await store.add('user prefers dark mode')

      const mm = new MemoryManager({ stores: [store] })
      const results = await mm.search('dark mode')
      expect(results).toHaveLength(1)
      expect(results[0]?.storeName).toBe('notes')
    })

    it('writes through the manager add API', async () => {
      const store = new TestMemoryStore({ name: 'notes', storage: new LocalFileStorage(dir) })
      const mm = new MemoryManager({ stores: [store], addToolConfig: true })
      await mm.add('user likes coffee')
      expect(JSON.parse(await fs.readFile(storageFilePath(), 'utf8'))).toHaveLength(1)
    })

    it('ingests extracted facts through add when extraction fires (client-side extractor)', async () => {
      const extractor: Extractor = {
        extract: vi.fn().mockResolvedValue([{ content: 'user prefers dark mode' }]),
      }
      const store = new TestMemoryStore({
        name: 'notes',
        storage: new LocalFileStorage(dir),
        extraction: { trigger: new InvocationTrigger(), extractor },
      })

      const mm = new MemoryManager({ stores: [store] })
      const agent = createMockAgent()
      await mm.initAgent(agent)

      const message = new Message({ role: 'user', content: [new TextBlock('I like dark mode')] })
      const added = agent.trackedHooks.filter((hook) => hook.eventType === MessageAddedEvent)
      for (const hook of added) await hook.callback(new MessageAddedEvent({ agent, message, invocationState: {} }))
      const after = agent.trackedHooks.filter((hook) => hook.eventType === AfterInvocationEvent)
      for (const hook of after) await hook.callback(new AfterInvocationEvent({ agent, invocationState: {} }))
      await mm.flush()

      expect(extractor.extract).toHaveBeenCalledTimes(1)
      const parsed = JSON.parse(await fs.readFile(storageFilePath(), 'utf8'))
      expect(parsed).toHaveLength(1)
      expect(parsed[0].content).toBe('user prefers dark mode')
    })
  })
})
