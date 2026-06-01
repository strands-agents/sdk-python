import { describe, it, expect } from 'vitest'
import { ContextManager, resolveContextManager } from '../context-manager.js'
import { ContextCompression } from '../compression/context-compression.js'
import { InMemoryStorage } from '../../vended-plugins/context-offloader/storage.js'
import { createMockAgent } from '../../__fixtures__/agent-helpers.js'
import type { Plugin } from '../../plugins/plugin.js'

describe('resolveContextManager', () => {
  it('with "auto" enables both compression and offloader', () => {
    const cm = resolveContextManager('auto')
    const subPlugins = (cm as any)._subPlugins as Plugin[]

    expect(subPlugins).toHaveLength(2)
    const names = subPlugins.map((p) => p.name)
    expect(names).toContain('strands:context-compression')
    expect(names).toContain('strands:context-offloader')
  })

  it('with config object is additive (omitted = disabled)', () => {
    const cm = resolveContextManager({ compression: true })
    const subPlugins = (cm as any)._subPlugins as Plugin[]

    const names = subPlugins.map((p) => p.name)
    expect(names).toContain('strands:context-compression')
    expect(names).not.toContain('strands:context-offloader')
  })

  it('with strategy: "auto" applies override semantics (omitted features stay enabled)', () => {
    const cm = resolveContextManager({ strategy: 'auto', compression: 'summarize' })
    const subPlugins = (cm as any)._subPlugins as Plugin[]

    const names = subPlugins.map((p) => p.name)
    // Both should be enabled because strategy: 'auto' defaults include offloader: true
    expect(names).toContain('strands:context-compression')
    expect(names).toContain('strands:context-offloader')

    // Compression should use summarize method
    const compression = subPlugins.find((p) => p.name === 'strands:context-compression') as ContextCompression
    expect((compression as any)._method).toBe('summarize')
  })

  it('with offloader: true uses default thresholds', () => {
    const cm = resolveContextManager({ offloader: true })
    const subPlugins = (cm as any)._subPlugins as Plugin[]

    const offloader = subPlugins.find((p) => p.name === 'strands:context-offloader')
    expect(offloader).toBeDefined()
  })

  it('with offloader config applies custom settings', () => {
    const cm = resolveContextManager({ offloader: { threshold: 5000, previewTokens: 1000 } })
    const subPlugins = (cm as any)._subPlugins as Plugin[]

    const offloader = subPlugins.find((p) => p.name === 'strands:context-offloader')
    expect(offloader).toBeDefined()
  })
})

describe('ContextManager._buildSubPlugins', () => {
  it('skips compression plugin when user already provides one', () => {
    const userCompression: Plugin = {
      name: 'strands:context-compression',
      initAgent: () => {},
      getTools: () => [],
    }

    const cm = resolveContextManager('auto', [userCompression])
    const subPlugins = (cm as any)._subPlugins as Plugin[]

    const compressionPlugins = subPlugins.filter((p) => p.name === 'strands:context-compression')
    expect(compressionPlugins).toHaveLength(0)
    // Offloader should still be present
    const offloaderPlugins = subPlugins.filter((p) => p.name === 'strands:context-offloader')
    expect(offloaderPlugins).toHaveLength(1)
  })

  it('skips offloader plugin when user already provides one', () => {
    const userOffloader: Plugin = {
      name: 'strands:context-offloader',
      initAgent: () => {},
      getTools: () => [],
    }

    const cm = resolveContextManager('auto', [userOffloader])
    const subPlugins = (cm as any)._subPlugins as Plugin[]

    const offloaderPlugins = subPlugins.filter((p) => p.name === 'strands:context-offloader')
    expect(offloaderPlugins).toHaveLength(0)
    // Compression should still be present
    const compressionPlugins = subPlugins.filter((p) => p.name === 'strands:context-compression')
    expect(compressionPlugins).toHaveLength(1)
  })
})

describe('ContextManager', () => {
  describe('constructor', () => {
    it('uses InMemoryStorage by default', () => {
      const cm = new ContextManager()
      expect(cm.storage).toBeInstanceOf(InMemoryStorage)
    })

    it('accepts custom storage', () => {
      const storage = new InMemoryStorage()
      const cm = new ContextManager({ storage })
      expect(cm.storage).toBe(storage)
    })

    it('has correct plugin name', () => {
      const cm = new ContextManager()
      expect(cm.name).toBe('strands:context-manager')
    })
  })

  describe('initAgent', () => {
    it('initializes sub-plugins', () => {
      const cm = new ContextManager({ compression: true, offloader: true })
      const agent = createMockAgent()

      cm.initAgent(agent)

      // Should have registered hooks from both sub-plugins
      expect(agent.trackedHooks.length).toBeGreaterThan(0)
    })

    it('builds sub-plugins if not already resolved', () => {
      const cm = new ContextManager({ compression: true })
      const agent = createMockAgent()

      // Don't call _resolveSubPlugins first
      cm.initAgent(agent)

      // Should still work and register hooks
      expect(agent.trackedHooks.length).toBeGreaterThan(0)
    })
  })

  describe('getTools', () => {
    it('returns tools from sub-plugins', () => {
      const cm = new ContextManager({ offloader: true })
      cm._resolveSubPlugins()

      const tools = cm.getTools()
      // ContextOffloader provides retrieval tool by default
      expect(tools.length).toBeGreaterThan(0)
      expect(tools[0]!.name).toBe('retrieve_offloaded_content')
    })

    it('returns empty array when no sub-plugins configured', () => {
      const cm = new ContextManager({})
      cm._resolveSubPlugins()

      const tools = cm.getTools()
      expect(tools).toHaveLength(0)
    })

    it('returns empty array when sub-plugins are not resolved yet', () => {
      const cm = new ContextManager({ offloader: true })
      // Don't resolve sub-plugins

      const tools = cm.getTools()
      expect(tools).toHaveLength(0)
    })
  })
})
