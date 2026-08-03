import { describe, it, expect } from 'vitest'
import { getCollection } from 'astro:content'
import {
  buildPythonApiSidebar,
  getDisplayName,
  type DocInfo,
  type SidebarEntry,
} from '../src/dynamic-sidebar'

describe('getDisplayName', () => {
  it('should capitalize single words', () => {
    expect(getDisplayName('agent')).toBe('Agent')
  })

  it('should convert snake_case to Title Case', () => {
    expect(getDisplayName('model_provider')).toBe('Model Provider')
  })

  it('should handle multiple underscores', () => {
    expect(getDisplayName('bidi_types_events')).toBe('Bidi Types Events')
  })
})

describe('buildPythonApiSidebar', () => {
  it('should create flat links for leaf modules', () => {
    const docs: DocInfo[] = [
      { id: 'docs/api/python/strands.interrupt.mdx', title: 'strands.interrupt' },
    ]

    const sidebar = buildPythonApiSidebar(docs, '')

    expect(sidebar).toHaveLength(1)
    expect(sidebar[0]).toMatchObject({
      type: 'link',
      label: 'Interrupt',
      href: '/docs/api/python/strands.interrupt.mdx/',
    })
  })

  it('should group modules by path segments', () => {
    const docs: DocInfo[] = [
      { id: 'docs/api/python/strands.agent.agent.mdx', title: 'strands.agent.agent' },
      { id: 'docs/api/python/strands.agent.base.mdx', title: 'strands.agent.base' },
    ]

    const sidebar = buildPythonApiSidebar(docs, '')

    expect(sidebar).toHaveLength(1)
    expect(sidebar[0]?.type).toBe('group')
    expect(sidebar[0]?.label).toBe('Agent')

    const group = sidebar[0] as Extract<SidebarEntry, { type: 'group' }>
    expect(group.entries).toHaveLength(2)
    expect(group.entries.map((e) => e.label)).toContain('Agent')
    expect(group.entries.map((e) => e.label)).toContain('Base')
  })

  it('should create nested groups for deep module paths', () => {
    const docs: DocInfo[] = [
      { id: 'docs/api/python/strands.experimental.bidi.types.events.mdx', title: 'strands.experimental.bidi.types.events' },
      { id: 'docs/api/python/strands.experimental.bidi.types.io.mdx', title: 'strands.experimental.bidi.types.io' },
    ]

    const sidebar = buildPythonApiSidebar(docs, '')

    // Should have Experimental group
    expect(sidebar).toHaveLength(1)
    expect(sidebar[0]?.type).toBe('group')
    expect(sidebar[0]?.label).toBe('Experimental')

    // Navigate to bidi > types
    const experimental = sidebar[0] as Extract<SidebarEntry, { type: 'group' }>
    const bidi = experimental.entries[0] as Extract<SidebarEntry, { type: 'group' }>
    expect(bidi.label).toBe('Bidi')

    const types = bidi.entries[0] as Extract<SidebarEntry, { type: 'group' }>
    expect(types.label).toBe('Types')

    // Should have events and io links
    expect(types.entries).toHaveLength(2)
    expect(types.entries.map((e) => e.label)).toContain('Events')
    expect(types.entries.map((e) => e.label)).toContain('Io')
  })

  it('should mark current page as isCurrent', () => {
    const docs: DocInfo[] = [
      { id: 'docs/api/python/strands.interrupt.mdx', title: 'strands.interrupt' },
    ]

    const sidebar = buildPythonApiSidebar(docs, 'docs/api/python/strands.interrupt.mdx')

    const link = sidebar[0] as Extract<SidebarEntry, { type: 'link' }>
    expect(link.isCurrent).toBe(true)
  })

  it('should filter out non-python-api docs', () => {
    const docs: DocInfo[] = [
      { id: 'docs/api/python/strands.agent.agent.mdx', title: 'strands.agent.agent' },
      { id: 'docs/user-guide/quickstart.mdx', title: 'Quickstart' },
      { id: 'docs/api/python/index', title: 'Python API Reference' },
    ]

    const sidebar = buildPythonApiSidebar(docs, '')

    // Should only have the agent group
    expect(sidebar).toHaveLength(1)
    expect(sidebar[0]?.label).toBe('Agent')
  })

  it('should sort groups before links', () => {
    const docs: DocInfo[] = [
      { id: 'docs/api/python/strands.interrupt.mdx', title: 'strands.interrupt' },
      { id: 'docs/api/python/strands.agent.agent.mdx', title: 'strands.agent.agent' },
    ]

    const sidebar = buildPythonApiSidebar(docs, '')

    // Agent group should come before Interrupt link
    expect(sidebar[0]?.type).toBe('group')
    expect(sidebar[0]?.label).toBe('Agent')
    expect(sidebar[1]?.type).toBe('link')
    expect(sidebar[1]?.label).toBe('Interrupt')
  })
})

describe('buildPythonApiSidebar with real collection', () => {
  it('should build sidebar from actual docs collection', async () => {
    const docs = await getCollection('docs')
    const docInfos: DocInfo[] = docs.map((doc) => ({
      id: doc.id,
      title: doc.data.title as string,
    }))

    const sidebar = buildPythonApiSidebar(docInfos, '')

    console.log('\n=== Python API Sidebar Structure ===\n')
    printSidebar(sidebar, 0)

    // Should have at least some entries
    expect(sidebar.length).toBeGreaterThan(0)
  })
})

function printSidebar(entries: SidebarEntry[], indent: number): void {
  const prefix = '  '.repeat(indent)
  for (const entry of entries) {
    if (entry.type === 'link') {
      console.log(`${prefix}- [link] ${entry.label} -> ${entry.href}`)
    } else if (entry.type === 'group') {
      console.log(`${prefix}- [group] ${entry.label}`)
      printSidebar(entry.entries, indent + 1)
    }
  }
}
