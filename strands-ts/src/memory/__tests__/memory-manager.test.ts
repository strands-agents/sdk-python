import { describe, it, expect, vi } from 'vitest'
import { Agent } from '../../agent/agent.js'
import { MemoryManager } from '../memory-manager.js'
import type { MemoryStore, MemoryEntry } from '../types.js'
import type { InvokableTool } from '../../tools/tool.js'

function createMockStore(
  name: string,
  options?: { entries?: MemoryEntry[]; writable?: boolean; description?: string; limit?: number }
): MemoryStore {
  const store: MemoryStore = {
    name,
    ...(options?.description && { description: options.description }),
    ...(options?.limit != null && { limit: options.limit }),
    search: vi.fn().mockResolvedValue(options?.entries ?? []),
  }
  if (options?.writable) {
    store.add = vi.fn().mockResolvedValue(undefined)
  }
  return store
}

describe('MemoryManager', () => {
  describe('constructor', () => {
    it('throws when stores array is empty', () => {
      expect(() => new MemoryManager({ stores: [] })).toThrow('at least one store is required')
    })

    it('creates instance with valid config', () => {
      const mm = new MemoryManager({ stores: [createMockStore('test')] })
      expect(mm.name).toBe('strands:memory-manager')
    })

    it('throws when two stores share a name', () => {
      expect(() => new MemoryManager({ stores: [createMockStore('dup'), createMockStore('dup')] })).toThrow(
        "duplicate store name 'dup'"
      )
    })

    it('throws when storeToolConfig references non-existent store', () => {
      expect(
        () =>
          new MemoryManager({
            stores: [createMockStore('a')],
            storeToolConfig: { stores: ['nonexistent'] },
          })
      ).toThrow("store 'nonexistent' not found")
    })

    it('throws when storeToolConfig targets no writable stores', () => {
      expect(
        () =>
          new MemoryManager({
            stores: [createMockStore('a')],
            storeToolConfig: true,
          })
      ).toThrow('storeToolConfig targets no writable stores')
    })

    it('throws when storeToolConfig is true with multiple writable stores and no explicit stores', () => {
      expect(
        () =>
          new MemoryManager({
            stores: [createMockStore('a', { writable: true }), createMockStore('b', { writable: true })],
            storeToolConfig: true,
          })
      ).toThrow('must specify `stores` when multiple writable stores are configured')
    })

    it('allows storeToolConfig true with single writable store', () => {
      const mm = new MemoryManager({
        stores: [createMockStore('a', { writable: true })],
        storeToolConfig: true,
      })
      expect(mm.getTools().map((t) => t.name)).toContain('store_memory')
    })
  })

  describe('getTools', () => {
    it('registers search tool by default', () => {
      const mm = new MemoryManager({ stores: [createMockStore('test')] })
      const tools = mm.getTools()
      expect(tools).toHaveLength(1)
      expect(tools[0]!.name).toBe('search_memory')
    })

    it('registers store tool when storeToolConfig is enabled', () => {
      const mm = new MemoryManager({
        stores: [createMockStore('test', { writable: true })],
        storeToolConfig: true,
      })
      const tools = mm.getTools()
      expect(tools.map((t) => t.name)).toStrictEqual(['search_memory', 'store_memory'])
    })

    it('does not register store tool by default', () => {
      const mm = new MemoryManager({ stores: [createMockStore('test', { writable: true })] })
      const tools = mm.getTools()
      expect(tools.map((t) => t.name)).toStrictEqual(['search_memory'])
    })

    it('returns empty array when searchToolConfig is false and storeToolConfig is false', () => {
      const mm = new MemoryManager({
        stores: [createMockStore('test', { writable: true })],
        searchToolConfig: false,
        storeToolConfig: false,
      })
      expect(mm.getTools()).toStrictEqual([])
    })

    it('uses custom tool names from MemoryToolConfig', () => {
      const mm = new MemoryManager({
        stores: [createMockStore('test', { writable: true })],
        searchToolConfig: { name: 'recall' },
        storeToolConfig: { name: 'remember', stores: ['test'] },
      })
      const tools = mm.getTools()
      expect(tools.map((t) => t.name)).toStrictEqual(['recall', 'remember'])
    })

    it('includes store descriptions in search tool description', () => {
      const store = createMockStore('personal', { description: 'User preferences' })
      const mm = new MemoryManager({ stores: [store] })
      const tools = mm.getTools()
      expect(tools[0]!.description).toContain('personal: User preferences')
      expect(tools[0]!.description).toContain('target one or more memory stores by name')
    })

    it('includes store descriptions in store tool description', () => {
      const store = createMockStore('notes', { writable: true, description: 'Personal notes' })
      const mm = new MemoryManager({ stores: [store], storeToolConfig: true })
      const tools = mm.getTools()
      const storeTool = tools.find((t) => t.name === 'store_memory')!
      expect(storeTool.description).toContain('notes: Personal notes')
      expect(storeTool.description).toContain('target a specific store by name')
    })
  })

  describe('search', () => {
    it('queries all stores and concatenates results', async () => {
      const store1 = createMockStore('a', { entries: [{ content: 'fact one' }] })
      const store2 = createMockStore('b', { entries: [{ content: 'fact two' }] })
      const mm = new MemoryManager({ stores: [store1, store2] })

      const results = await mm.search('query')
      expect(results).toStrictEqual([{ content: 'fact one' }, { content: 'fact two' }])
    })

    it('passes limit to each store', async () => {
      const store = createMockStore('a', { limit: 5 })
      const mm = new MemoryManager({ stores: [store] })

      await mm.search('query')
      expect(store.search).toHaveBeenCalledWith('query', { limit: 5 })
    })

    it('overrides per-store limit with options.limit', async () => {
      const store = createMockStore('a', { limit: 5 })
      const mm = new MemoryManager({ stores: [store] })

      await mm.search('query', { limit: 2 })
      expect(store.search).toHaveBeenCalledWith('query', { limit: 2 })
    })

    it('defaults to limit of 3 when no limit configured', async () => {
      const store = createMockStore('a')
      const mm = new MemoryManager({ stores: [store] })

      await mm.search('query')
      expect(store.search).toHaveBeenCalledWith('query', { limit: 3 })
    })

    it('filters to named stores when options.stores is provided', async () => {
      const store1 = createMockStore('personal', { entries: [{ content: 'personal fact' }] })
      const store2 = createMockStore('team', { entries: [{ content: 'team fact' }] })
      const mm = new MemoryManager({ stores: [store1, store2] })

      const results = await mm.search('query', { stores: ['personal'] })
      expect(results).toStrictEqual([{ content: 'personal fact' }])
      expect(store2.search).not.toHaveBeenCalled()
    })

    it('gracefully handles store failures', async () => {
      const store1: MemoryStore = { name: 'failing', search: vi.fn().mockRejectedValue(new Error('network error')) }
      const store2 = createMockStore('ok', { entries: [{ content: 'fact' }] })
      const mm = new MemoryManager({ stores: [store1, store2] })

      const results = await mm.search('query')
      expect(results).toStrictEqual([{ content: 'fact' }])
    })

    it('searches all stores when stores option is omitted', async () => {
      const store1 = createMockStore('a', { entries: [{ content: 'fact one' }] })
      const store2 = createMockStore('b', { entries: [{ content: 'fact two' }] })
      const mm = new MemoryManager({ stores: [store1, store2] })

      const results = await mm.search('query')
      expect(results).toStrictEqual([{ content: 'fact one' }, { content: 'fact two' }])
    })

    it('searches no stores when stores option is an empty array', async () => {
      const store1 = createMockStore('a', { entries: [{ content: 'fact one' }] })
      const store2 = createMockStore('b', { entries: [{ content: 'fact two' }] })
      const mm = new MemoryManager({ stores: [store1, store2] })

      const results = await mm.search('query', { stores: [] })
      expect(results).toStrictEqual([])
      expect(store1.search).not.toHaveBeenCalled()
      expect(store2.search).not.toHaveBeenCalled()
    })
  })

  describe('store', () => {
    it('writes to all writable stores', async () => {
      const store1 = createMockStore('a', { writable: true })
      const store2 = createMockStore('b', { writable: true })
      const mm = new MemoryManager({ stores: [store1, store2] })

      await mm.store('user likes coffee')
      expect(store1.add).toHaveBeenCalledWith('user likes coffee', undefined)
      expect(store2.add).toHaveBeenCalledWith('user likes coffee', undefined)
    })

    it('passes metadata to stores', async () => {
      const store = createMockStore('a', { writable: true })
      const mm = new MemoryManager({ stores: [store] })

      await mm.store('fact', { metadata: { source: 'user' } })
      expect(store.add).toHaveBeenCalledWith('fact', { source: 'user' })
    })

    it('filters to named stores when options.stores is provided', async () => {
      const store1 = createMockStore('personal', { writable: true })
      const store2 = createMockStore('team', { writable: true })
      const mm = new MemoryManager({ stores: [store1, store2] })

      await mm.store('my preference', { stores: ['personal'] })
      expect(store1.add).toHaveBeenCalledWith('my preference', undefined)
      expect(store2.add).not.toHaveBeenCalled()
    })

    it('throws when no writable stores match', async () => {
      const mm = new MemoryManager({ stores: [createMockStore('a')] })
      await expect(mm.store('fact')).rejects.toThrow('no writable store matched')
    })

    it('throws a not-found error when a named store does not exist', async () => {
      const mm = new MemoryManager({ stores: [createMockStore('a', { writable: true })] })
      await expect(mm.store('fact', { stores: ['nonexistent'] })).rejects.toThrow("store 'nonexistent' not found")
    })

    it('throws a read-only error when a named store cannot be written', async () => {
      const mm = new MemoryManager({ stores: [createMockStore('readonly')] })
      await expect(mm.store('fact', { stores: ['readonly'] })).rejects.toThrow("store 'readonly' is read-only")
    })

    it('succeeds with partial write failures (some stores fail, some succeed)', async () => {
      const store1: MemoryStore = {
        name: 'failing',
        search: vi.fn().mockResolvedValue([]),
        add: vi.fn().mockRejectedValue(new Error('write error')),
      }
      const store2 = createMockStore('ok', { writable: true })
      const mm = new MemoryManager({ stores: [store1, store2] })

      await mm.store('fact')
      expect(store2.add).toHaveBeenCalledWith('fact', undefined)
    })

    it('throws AggregateError naming the failed stores when all writes fail', async () => {
      const store: MemoryStore = {
        name: 'failing',
        search: vi.fn().mockResolvedValue([]),
        add: vi.fn().mockRejectedValue(new Error('write error')),
      }
      const mm = new MemoryManager({ stores: [store] })

      await expect(mm.store('fact')).rejects.toThrow('all store writes failed: failing')
    })
  })

  describe('tool store scoping', () => {
    function searchTool(
      mm: MemoryManager
    ): InvokableTool<{ query: string; limit?: number; stores?: string[] }, unknown> {
      return mm.getTools().find((t) => t.name === 'search_memory') as never
    }

    function storeTool(mm: MemoryManager): InvokableTool<{ entries: string[]; stores?: string[] }, unknown> {
      return mm.getTools().find((t) => t.name === 'store_memory') as never
    }

    it('search tool only queries scoped stores when model omits stores', async () => {
      const personal = createMockStore('personal', { entries: [{ content: 'personal fact' }] })
      const team = createMockStore('team', { entries: [{ content: 'team fact' }] })
      const mm = new MemoryManager({ stores: [personal, team], searchToolConfig: { stores: ['personal'] } })

      await searchTool(mm).invoke({ query: 'q' })
      expect(personal.search).toHaveBeenCalled()
      expect(team.search).not.toHaveBeenCalled()
    })

    it('search tool drops out-of-scope store names supplied by the model', async () => {
      const personal = createMockStore('personal', { entries: [{ content: 'personal fact' }] })
      const team = createMockStore('team', { entries: [{ content: 'team fact' }] })
      const mm = new MemoryManager({ stores: [personal, team], searchToolConfig: { stores: ['personal'] } })

      await searchTool(mm).invoke({ query: 'q', stores: ['team'] })
      expect(personal.search).not.toHaveBeenCalled()
      expect(team.search).not.toHaveBeenCalled()
    })

    it('store tool only writes to scoped stores when model omits stores', async () => {
      const personal = createMockStore('personal', { writable: true })
      const team = createMockStore('team', { writable: true })
      const mm = new MemoryManager({ stores: [personal, team], storeToolConfig: { stores: ['personal'] } })

      await storeTool(mm).invoke({ entries: ['fact'] })
      expect(personal.add).toHaveBeenCalledWith('fact', undefined)
      expect(team.add).not.toHaveBeenCalled()
    })

    it('store tool drops out-of-scope store names supplied by the model', async () => {
      const personal = createMockStore('personal', { writable: true })
      const team = createMockStore('team', { writable: true })
      const mm = new MemoryManager({ stores: [personal, team], storeToolConfig: { stores: ['personal'] } })

      const result = await storeTool(mm).invoke({ entries: ['fact'], stores: ['team'] })
      expect(personal.add).not.toHaveBeenCalled()
      expect(team.add).not.toHaveBeenCalled()
      expect(result).toStrictEqual({ stored: 0, failed: 1 })
    })
  })

  describe('initAgent', () => {
    it('does not throw', () => {
      const mm = new MemoryManager({ stores: [createMockStore('test')] })
      expect(() => mm.initAgent({} as any)).not.toThrow()
    })
  })

  describe('AgentConfig integration', () => {
    it('auto-wraps MemoryManagerConfig into MemoryManager instance', () => {
      const store = createMockStore('test')
      const agent = new Agent({ memoryManager: { stores: [store] } })
      expect(agent.memoryManager).toBeInstanceOf(MemoryManager)
    })

    it('passes through MemoryManager instance unchanged', () => {
      const mm = new MemoryManager({ stores: [createMockStore('test')] })
      const agent = new Agent({ memoryManager: mm })
      expect(agent.memoryManager).toBe(mm)
    })

    it('sets memoryManager to undefined when not configured', () => {
      const agent = new Agent({})
      expect(agent.memoryManager).toBeUndefined()
    })
  })
})
