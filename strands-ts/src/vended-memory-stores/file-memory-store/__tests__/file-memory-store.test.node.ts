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
      for (let index = 0; index < 10; index++) {
        await store.add(`Fact number ${index}`, { title: `fact-${index}`, description: `Fact ${index}` })
      }
      const results = await store.search('fact')
      expect(results).toHaveLength(5)
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
      expect(results.length).toBeGreaterThanOrEqual(1)
      expect(results[0]!.content).toContain('dark mode')
    })

    it('matches against filenames', async () => {
      const results = await store.search('deploy')
      expect(results.length).toBeGreaterThanOrEqual(1)
      expect(results[0]!.metadata?.['path']).toContain('deploy')
    })

    it('matches against description frontmatter', async () => {
      const results = await store.search('integration-first')
      expect(results.length).toBeGreaterThanOrEqual(1)
      expect(results[0]!.metadata?.['description']).toContain('Integration-first')
    })

    it('is case-insensitive', async () => {
      const results = await store.search('DARK MODE')
      expect(results.length).toBeGreaterThanOrEqual(1)
      expect(results[0]!.content).toContain('dark mode')
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

    it('ranks results by term frequency', async () => {
      await store.add('dark mode dark mode dark mode repeated many times', {
        title: 'dark-repeated',
        description: 'Repeated dark mode mentions',
      })
      const results = await store.search('dark mode')
      expect(results[0]!.metadata?.['path']).toContain('dark-repeated')
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
      expect(results[0]!.metadata?.['path']).toBe('knowledge/facts/deploy.md')
      expect(results[0]!.metadata?.['description']).toBe('Deployment pipeline details')
    })

    it('returns body content without frontmatter', async () => {
      const results = await store.search('blue-green')
      expect(results[0]!.content).not.toContain('---')
      expect(results[0]!.content).not.toContain('description:')
      expect(results[0]!.content).toContain('blue-green strategy')
    })

    it('handles multi-word queries by scoring each term independently', async () => {
      const results = await store.search('integration boundaries')
      expect(results.length).toBeGreaterThanOrEqual(1)
      expect(results[0]!.metadata?.['path']).toContain('testing')
    })

    it('searches across subdirectories', async () => {
      await store.add('Check CloudWatch logs first', { path: 'operations/debugging', description: 'Debugging runbook' })
      const results = await store.search('CloudWatch')
      expect(results.length).toBeGreaterThanOrEqual(1)
      expect(results[0]!.metadata?.['path']).toBe('knowledge/operations/debugging.md')
    })

    it('searches files without frontmatter', async () => {
      await storage.write('knowledge/facts/plain.md', encoder.encode('Retry with exponential backoff'))
      const results = await store.search('exponential backoff')
      expect(results.length).toBeGreaterThanOrEqual(1)
      expect(results[0]!.content).toContain('exponential backoff')
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
  })
})
