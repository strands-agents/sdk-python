import { describe, it, expect, beforeEach } from 'vitest'
import { FileMemoryStore } from '../file-memory-store.js'
import { InMemoryStorage } from '../../../storage/in-memory-storage.js'
import type { Storage } from '../../../storage/storage.js'

const encoder = new TextEncoder()
const decoder = new TextDecoder()

describe('FileMemoryStore', () => {
  let storage: InMemoryStorage
  let store: FileMemoryStore

  beforeEach(() => {
    storage = new InMemoryStorage()
    store = new FileMemoryStore({
      name: 'test-store',
      description: 'A test file memory store',
      storage,
    })
  })

  describe('constructor', () => {
    it('sets name and description from config', () => {
      expect(store.name).toBe('test-store')
      expect(store.description).toBe('A test file memory store')
    })

    it('defaults to writable true', () => {
      expect(store.writable).toBe(true)
    })

    it('respects writable false from config', () => {
      const readOnly = new FileMemoryStore({ name: 'readonly', storage, writable: false })
      expect(readOnly.writable).toBe(false)
    })

    it('uses default maxSearchResults when not configured', async () => {
      for (let index = 0; index < 15; index++) {
        await store.add(`Fact number ${index}`, { title: `fact-${index}`, description: `Fact ${index}` })
      }
      const results = await store.search('fact')
      expect(results).toHaveLength(10)
    })

    it('respects maxSearchResults from config', async () => {
      const customStore = new FileMemoryStore({ name: 'custom', storage, maxSearchResults: 2 })
      for (let index = 0; index < 5; index++) {
        await customStore.add(`Fact number ${index}`, { title: `fact-${index}`, description: `Fact ${index}` })
      }
      const results = await customStore.search('fact')
      expect(results).toHaveLength(2)
    })

    it('stores extraction config', () => {
      const storeWithExtraction = new FileMemoryStore({ name: 'ext', storage, extraction: true })
      expect(storeWithExtraction.extraction).toBe(true)
    })

    it('defaults to no description when omitted', () => {
      const minimal = new FileMemoryStore({ name: 'minimal', storage })
      expect(minimal.description).toBeUndefined()
    })
  })

  describe('add', () => {
    it('writes a markdown file to knowledge/facts/ with frontmatter', async () => {
      await store.add('User prefers dark mode', { title: 'dark-mode', description: 'Theme preference' })
      const bytes = await storage.read('knowledge/facts/dark-mode.md')
      expect(bytes).not.toBeNull()
      const content = decoder.decode(bytes!)
      expect(content).toBe('---\ndescription: "Theme preference"\n---\n\nUser prefers dark mode\n')
    })

    it('derives filename from content when no title provided', async () => {
      await store.add('The user likes vim keybindings')
      const keys = await storage.list('knowledge/facts/')
      expect(keys).toHaveLength(1)
      expect(keys[0]).toBe('knowledge/facts/the-user-likes-vim-keybindings.md')
    })

    it('derives description from first sentence when not provided', async () => {
      await store.add('Always use strict mode. It prevents bugs.')
      const keys = await storage.list('knowledge/facts/')
      const bytes = await storage.read(keys[0]!)
      const content = decoder.decode(bytes!)
      expect(content).toContain('description: "Always use strict mode"')
    })

    it('splits on newline for description derivation', async () => {
      await store.add('First line\nSecond line\nThird line')
      const keys = await storage.list('knowledge/facts/')
      const bytes = await storage.read(keys[0]!)
      const content = decoder.decode(bytes!)
      expect(content).toContain('description: "First line"')
    })

    it('truncates derived description at 120 characters', async () => {
      const longSentence = 'a'.repeat(200)
      await store.add(longSentence)
      const keys = await storage.list('knowledge/facts/')
      const bytes = await storage.read(keys[0]!)
      const content = decoder.decode(bytes!)
      const descMatch = content.match(/description: "(.+?)"/)
      expect(descMatch![1]!.length).toBeLessThanOrEqual(120)
    })

    it('truncates derived slug at 50 characters', async () => {
      const longContent =
        'this is a very long sentence that should be truncated when used as a filename slug for storage'
      await store.add(longContent)
      const keys = await storage.list('knowledge/facts/')
      const slug = keys[0]!.replace('knowledge/facts/', '').replace('.md', '')
      expect(slug.length).toBeLessThanOrEqual(50)
    })

    it('escapes double quotes in description frontmatter', async () => {
      await store.add('Use "strict" mode always', { title: 'quotes', description: 'Prefers "strict" mode' })
      const bytes = await storage.read('knowledge/facts/quotes.md')
      const content = decoder.decode(bytes!)
      expect(content).toContain('description: "Prefers \\"strict\\" mode"')
    })

    it('round-trips description containing double quotes', async () => {
      await store.add('Use strict mode', { title: 'quotes-rt', description: 'Prefers "strict" mode' })
      const results = await store.search('strict mode')
      const entry = results.find((r) => (r.metadata?.['path'] as string).includes('quotes-rt'))
      expect(entry!.metadata?.['description']).toBe('Prefers "strict" mode')
    })

    it('round-trips description containing newlines', async () => {
      await store.add('multi-line desc content', { title: 'newline-rt', description: 'line one\nline two' })
      const results = await store.search('multi-line desc')
      const entry = results.find((r) => (r.metadata?.['path'] as string).includes('newline-rt'))
      expect(entry!.metadata?.['description']).toBe('line one\nline two')
    })

    it('does not allow YAML frontmatter injection via description', async () => {
      await store.add('real body', { title: 'inject-rt', description: 'intro\n---\nsecret: injected' })
      const results = await store.search('real body')
      const entry = results.find((r) => (r.metadata?.['path'] as string).includes('inject-rt'))
      expect(entry!.content).toBe('real body')
      expect(entry!.metadata?.['description']).toBe('intro\n---\nsecret: injected')
    })

    it('writes to custom path when metadata.path is provided', async () => {
      await store.add('Check CloudWatch logs first', { path: 'operations/debugging', description: 'Debugging runbook' })
      const bytes = await storage.read('knowledge/operations/debugging.md')
      expect(bytes).not.toBeNull()
      expect(decoder.decode(bytes!)).toContain('Check CloudWatch logs first')
    })

    it('accepts custom path with knowledge/ prefix already included', async () => {
      await store.add('Code review patterns', {
        path: 'knowledge/operations/code-review.md',
        description: 'CR patterns',
      })
      const bytes = await storage.read('knowledge/operations/code-review.md')
      expect(bytes).not.toBeNull()
    })

    it('appends .md extension to custom path if missing', async () => {
      await store.add('Deploy steps', { path: 'operations/deploy', description: 'Deploy process' })
      const bytes = await storage.read('knowledge/operations/deploy.md')
      expect(bytes).not.toBeNull()
    })

    it('does not double-append .md if already present', async () => {
      await store.add('Steps', { path: 'operations/deploy.md', description: 'Deploy' })
      const keys = await storage.list('knowledge/')
      expect(keys).not.toContain('knowledge/operations/deploy.md.md')
      expect(keys).toContain('knowledge/operations/deploy.md')
    })

    it('does not overwrite an existing entry when slugs collide', async () => {
      await store.add('Python is great')
      await store.add('Python is great. But has a GIL.')
      const keys = await storage.list('knowledge/facts/')
      expect(keys).toHaveLength(2)
      expect(keys).toContain('knowledge/facts/python-is-great.md')
      expect(keys).toContain('knowledge/facts/python-is-great-1.md')
      const first = decoder.decode((await storage.read('knowledge/facts/python-is-great.md'))!)
      expect(first).toContain('Python is great')
      expect(first).not.toContain('GIL')
    })

    it('avoids collisions on a case-insensitive backend', async () => {
      // A case-insensitive filesystem treats Topic.md and topic.md as one file. The probe
      // must delegate to the backend rather than compare exact key spellings in memory.
      const files = new Map<string, Uint8Array>()
      const caseInsensitiveStorage: Storage = {
        async write(key: string, data: Uint8Array): Promise<void> {
          const existing = [...files.keys()].find((k) => k.toLowerCase() === key.toLowerCase())
          files.set(existing ?? key, data)
        },
        async read(key: string): Promise<Uint8Array | null> {
          const existing = [...files.keys()].find((k) => k.toLowerCase() === key.toLowerCase())
          return existing ? files.get(existing)! : null
        },
        async delete(key: string): Promise<void> {
          const existing = [...files.keys()].find((k) => k.toLowerCase() === key.toLowerCase())
          if (existing) files.delete(existing)
        },
        async list(prefix: string): Promise<string[]> {
          return [...files.keys()].filter((k) => k.startsWith(prefix)).sort()
        },
      }
      files.set('knowledge/facts/Topic.md', encoder.encode('---\ndescription: "x"\n---\n\npreexisting fact'))
      const caseStore = new FileMemoryStore({ name: 'case-store', storage: caseInsensitiveStorage })

      await caseStore.add('New fact', { title: 'topic' })

      expect(files.size).toBe(2)
      expect(decoder.decode(files.get('knowledge/facts/Topic.md')!)).toContain('preexisting fact')
      expect(decoder.decode(files.get('knowledge/facts/topic-1.md')!)).toContain('New fact')
    })

    it('uses Date.now fallback when content produces empty slug', async () => {
      await store.add('!!!???')
      const keys = await storage.list('knowledge/facts/')
      expect(keys).toHaveLength(1)
      expect(keys[0]).toMatch(/knowledge\/facts\/entry-\d+\.md$/)
    })

    it('slugifies special characters out of the filename', async () => {
      await store.add("User's #1 testing rule!", { title: "User's #1 testing rule!" })
      const keys = await storage.list('knowledge/facts/')
      expect(keys[0]).toBe('knowledge/facts/users-1-testing-rule.md')
    })

    it('returns the key for a default facts/ path', async () => {
      const key = await store.add('User prefers dark mode', { title: 'dark-mode' })
      expect(key).toBe('knowledge/facts/dark-mode.md')
    })

    it('returns the key for a custom path', async () => {
      const key = await store.add('Deploy steps', { path: 'operations/deploy', description: 'Deploy process' })
      expect(key).toBe('knowledge/operations/deploy.md')
    })

    it('returns the collision-suffixed key when slugs collide', async () => {
      await store.add('Python is great')
      const key = await store.add('Python is great. But has a GIL.')
      expect(key).toBe('knowledge/facts/python-is-great-1.md')
    })

    it('returns the canonical key search and list report, not the pre-normalized path', async () => {
      const key = await store.add('Rollback runbook', { path: 'operations//deploy' })
      expect(key).toBe('knowledge/operations/deploy.md')
      // The returned receipt must match what the backend actually stored under.
      const keys = await storage.list('knowledge/')
      expect(keys).toContain(key)
      expect(await storage.read(key)).not.toBeNull()
    })
  })

  describe('search', () => {
    beforeEach(async () => {
      await store.add('User prefers dark mode for all editors', {
        title: 'dark-mode',
        description: 'Theme preference: dark mode',
      })
      await store.add('Testing philosophy: integration first, mock at boundaries', {
        title: 'testing',
        description: 'Integration-first testing approach',
      })
      await store.add('Deploy process uses blue-green strategy', {
        title: 'deploy',
        description: 'Deployment pipeline details',
      })
    })

    it('returns matching entries by keyword in content', async () => {
      const results = await store.search('dark mode')
      expect(results[0]).toEqual({
        content: 'User prefers dark mode for all editors',
        metadata: {
          path: 'knowledge/facts/dark-mode.md',
          description: 'Theme preference: dark mode',
          _relevanceScore: expect.any(Number),
        },
      })
    })

    it('matches against filenames', async () => {
      const results = await store.search('deploy')
      expect(results[0]).toEqual({
        content: 'Deploy process uses blue-green strategy',
        metadata: {
          path: 'knowledge/facts/deploy.md',
          description: 'Deployment pipeline details',
          _relevanceScore: expect.any(Number),
        },
      })
    })

    it('matches against description frontmatter', async () => {
      const results = await store.search('integration-first')
      expect(results[0]).toEqual({
        content: 'Testing philosophy: integration first, mock at boundaries',
        metadata: {
          path: 'knowledge/facts/testing.md',
          description: 'Integration-first testing approach',
          _relevanceScore: expect.any(Number),
        },
      })
    })

    it('is case-insensitive', async () => {
      const results = await store.search('DARK MODE')
      expect(results[0]).toEqual({
        content: 'User prefers dark mode for all editors',
        metadata: {
          path: 'knowledge/facts/dark-mode.md',
          description: 'Theme preference: dark mode',
          _relevanceScore: expect.any(Number),
        },
      })
    })

    it('returns empty array for no matches', async () => {
      const results = await store.search('quantum computing')
      expect(results).toEqual([])
    })

    it('returns empty array for empty query', async () => {
      const results = await store.search('')
      expect(results).toEqual([])
    })

    it('returns empty array for whitespace-only query', async () => {
      const results = await store.search('   ')
      expect(results).toEqual([])
    })

    it('respects maxSearchResults option', async () => {
      const results = await store.search('process', { maxSearchResults: 1 })
      expect(results).toHaveLength(1)
    })

    it('ranks results by number of distinct matching tokens', async () => {
      await store.add('covers deploy, testing, and integration boundaries', {
        title: 'broad-match',
        description: 'Broad topic coverage',
      })
      const results = await store.search('deploy testing integration')
      expect(results[0]).toEqual({
        content: 'covers deploy, testing, and integration boundaries',
        metadata: {
          path: 'knowledge/facts/broad-match.md',
          description: 'Broad topic coverage',
          _relevanceScore: expect.any(Number),
        },
      })
    })

    it('includes knowledge/system/ files in results', async () => {
      await storage.write(
        'knowledge/system/prefs.md',
        encoder.encode('---\ndescription: "User prefs"\n---\n\ndark mode everywhere')
      )
      const results = await store.search('dark mode')
      const paths = results.map((r) => r.metadata?.['path'] as string)
      expect(paths.some((p) => p.startsWith('knowledge/system/'))).toBe(true)
    })

    it('returns entries with path and description in metadata', async () => {
      const results = await store.search('deploy')
      expect(results[0]).toEqual({
        content: 'Deploy process uses blue-green strategy',
        metadata: {
          path: 'knowledge/facts/deploy.md',
          description: 'Deployment pipeline details',
          _relevanceScore: expect.any(Number),
        },
      })
    })

    it('returns body content without frontmatter', async () => {
      const results = await store.search('blue-green')
      expect(results[0]).toEqual({
        content: 'Deploy process uses blue-green strategy',
        metadata: {
          path: 'knowledge/facts/deploy.md',
          description: 'Deployment pipeline details',
          _relevanceScore: expect.any(Number),
        },
      })
    })

    it('handles multi-word queries by scoring each term independently', async () => {
      const results = await store.search('integration boundaries')
      expect(results[0]).toEqual({
        content: 'Testing philosophy: integration first, mock at boundaries',
        metadata: {
          path: 'knowledge/facts/testing.md',
          description: 'Integration-first testing approach',
          _relevanceScore: expect.any(Number),
        },
      })
    })

    it('searches across subdirectories', async () => {
      await store.add('Check CloudWatch logs first', { path: 'operations/debugging', description: 'Debugging runbook' })
      const results = await store.search('CloudWatch')
      expect(results[0]).toEqual({
        content: 'Check CloudWatch logs first',
        metadata: {
          path: 'knowledge/operations/debugging.md',
          description: 'Debugging runbook',
          _relevanceScore: expect.any(Number),
        },
      })
    })

    it('searches files without frontmatter', async () => {
      await storage.write('knowledge/facts/plain.md', encoder.encode('Retry with exponential backoff'))
      const results = await store.search('exponential backoff')
      expect(results[0]).toEqual({
        content: 'Retry with exponential backoff',
        metadata: { path: 'knowledge/facts/plain.md', description: '', _relevanceScore: expect.any(Number) },
      })
    })

    it('skips keys where storage returns null', async () => {
      const nullStorage: Storage = {
        async write(): Promise<void> {},
        async read(): Promise<Uint8Array | null> {
          return null
        },
        async delete(): Promise<void> {},
        async list(): Promise<string[]> {
          return ['knowledge/facts/ghost.md']
        },
      }
      const nullStore = new FileMemoryStore({ name: 'null-store', storage: nullStorage })
      const results = await nullStore.search('anything')
      expect(results).toEqual([])
    })

    it('skips keys where storage.read throws and still returns other matches', async () => {
      const throwingStorage: Storage = {
        async write(): Promise<void> {},
        async read(key: string): Promise<Uint8Array | null> {
          if (key.includes('broken')) throw new Error('EACCES: permission denied')
          return encoder.encode('---\ndescription: "A good entry"\n---\n\nvalid content about deploy')
        },
        async delete(): Promise<void> {},
        async list(): Promise<string[]> {
          return ['knowledge/facts/broken.md', 'knowledge/facts/good.md']
        },
      }
      const throwingStore = new FileMemoryStore({ name: 'throwing-store', storage: throwingStorage })
      const results = await throwingStore.search('deploy')
      expect(results).toEqual([
        {
          content: 'valid content about deploy',
          metadata: {
            path: 'knowledge/facts/good.md',
            description: 'A good entry',
            _relevanceScore: expect.any(Number),
          },
        },
      ])
    })

    it('bounds concurrent reads so a capacity-limited backend returns every match', async () => {
      // A backend that throws once more than MAX_ACTIVE reads overlap. An unbounded fan-out
      // (one read per key) would trip this on a large corpus and silently drop those matches.
      // MAX_ACTIVE matches the store's internal SEARCH_READ_CONCURRENCY cap.
      const MAX_ACTIVE = 8
      const keys = Array.from({ length: 30 }, (_, index) => `knowledge/facts/fact-${index}.md`)
      let active = 0
      let peak = 0
      const boundedStorage: Storage = {
        async write(): Promise<void> {},
        async read(): Promise<Uint8Array | null> {
          active++
          peak = Math.max(peak, active)
          if (active > MAX_ACTIVE) {
            active--
            throw new Error('TooManyConcurrentReads')
          }
          try {
            await new Promise((resolve) => setTimeout(resolve, 1))
            return encoder.encode('---\ndescription: "A fact"\n---\n\nfact about deploy')
          } finally {
            active--
          }
        },
        async delete(): Promise<void> {},
        async list(): Promise<string[]> {
          return keys
        },
      }
      const boundedStore = new FileMemoryStore({
        name: 'bounded-store',
        storage: boundedStorage,
        maxSearchResults: keys.length,
      })
      const results = await boundedStore.search('deploy')
      expect(results).toHaveLength(keys.length)
      expect(peak).toBeLessThanOrEqual(MAX_ACTIVE)
    })
  })
})
