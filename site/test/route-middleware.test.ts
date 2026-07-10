import { describe, it, expect } from 'vitest'
import path from 'node:path'
import {
  findCurrentNavSection,
  filterSidebarByBasePath,
  applyCollapse,
} from '../src/route-middleware'
import { type NavLink } from '../src/config/navbar'
import { loadSidebarFromConfig, type StarlightSidebarItem } from '../src/sidebar'

// Sidebar entry types matching Starlight's runtime structure
type SidebarLink = { type: 'link'; label: string; href: string; isCurrent: boolean }
type SidebarGroup = { type: 'group'; label: string; entries: SidebarEntry[]; collapsed: boolean }
type SidebarEntry = SidebarLink | SidebarGroup

// Test nav links without base path (simulating no BASE_URL set)
const testNavLinks: NavLink[] = [
  { label: 'Home', href: '/' },
  { label: 'User Guide', href: '/docs/user-guide/quickstart/overview/', basePath: '/docs/user-guide/' },
  { label: 'Examples', href: '/docs/examples/', basePath: '/docs/examples/' },
  { label: 'Community', href: '/docs/community/community-packages/', basePath: '/docs/community/' },
  { label: 'Contribute', href: 'https://github.com/example', external: true },
]

/**
 * Convert build-time sidebar config to runtime format.
 * Starlight transforms { slug: "examples" } to { type: "link", href: "/examples/", ... }
 */
function convertToRuntimeFormat(items: StarlightSidebarItem[]): SidebarEntry[] {
  return items.map((item) => {
    if ('slug' in item) {
      return {
        type: 'link' as const,
        label: item.label || item.slug,
        href: '/' + item.slug + '/',
        isCurrent: false,
      }
    }
    if ('link' in item) {
      return {
        type: 'link' as const,
        label: item.label,
        href: item.link,
        isCurrent: false,
      }
    }
    if ('items' in item) {
      return {
        type: 'group' as const,
        label: item.label,
        entries: convertToRuntimeFormat(item.items),
        collapsed: item.collapsed ?? false,
      }
    }
    throw new Error('Unknown sidebar item type')
  })
}

// Helper to extract all links from sidebar structure
function getAllLinks(entries: SidebarEntry[]): SidebarLink[] {
  const links: SidebarLink[] = []
  for (const entry of entries) {
    if (entry.type === 'link') {
      links.push(entry)
    } else if (entry.type === 'group') {
      links.push(...getAllLinks(entry.entries))
    }
  }
  return links
}

describe('findCurrentNavSection', () => {
  it('should find Examples nav for /examples/ path', () => {
    const result = findCurrentNavSection('/docs/examples/', testNavLinks)
    expect(result).toBeDefined()
    expect(result?.label).toBe('Examples')
  })

  it('should find Examples nav for nested examples path', () => {
    const result = findCurrentNavSection('/docs/examples/python/weather_forecaster/', testNavLinks)
    expect(result).toBeDefined()
    expect(result?.label).toBe('Examples')
  })

  it('should find User Guide nav for user-guide paths', () => {
    const result = findCurrentNavSection('/docs/user-guide/quickstart/overview/', testNavLinks)
    expect(result).toBeDefined()
    expect(result?.label).toBe('User Guide')
  })

  it('should find Community nav for community paths', () => {
    const result = findCurrentNavSection('/docs/community/community-packages/', testNavLinks)
    expect(result).toBeDefined()
    expect(result?.label).toBe('Community')
  })

  it('should find Home nav for root path', () => {
    const result = findCurrentNavSection('/', testNavLinks)
    expect(result).toBeDefined()
    expect(result?.label).toBe('Home')
  })

  it('should not match external links', () => {
    const externalOnlyLinks: NavLink[] = [
      { label: 'Contribute', href: 'https://github.com/example', external: true },
    ]
    const result = findCurrentNavSection('/anything/', externalOnlyLinks)
    expect(result).toBeUndefined()
  })

  it('should work with base path in nav links', () => {
    const navWithBase: NavLink[] = [
      { label: 'Home', href: '/docs/' },
      { label: 'Examples', href: '/docs/examples/', basePath: '/docs/examples/' },
    ]
    const result = findCurrentNavSection('/docs/examples/python/', navWithBase)
    expect(result).toBeDefined()
    expect(result?.label).toBe('Examples')
  })
})

describe('Sidebar filtering with live navigation.yml data', () => {
  const configPath = path.resolve('./src/config/navigation.yml')
  const docsDir = path.resolve('./src/content')
  const buildTimeSidebar = loadSidebarFromConfig(configPath, docsDir)
  const runtimeSidebar = convertToRuntimeFormat(buildTimeSidebar)

  it('should have loaded sidebar from navigation.yml', () => {
    expect(runtimeSidebar.length).toBeGreaterThan(0)
    console.log(`\nLoaded ${runtimeSidebar.length} top-level sidebar items`)
  })

  it('should filter sidebar to only Examples items for /examples/ basePath', () => {
    const result = filterSidebarByBasePath(runtimeSidebar as any, '/docs/examples/')

    const allLinks = getAllLinks(result)
    console.log(`\nExamples section has ${allLinks.length} links:`)
    allLinks.slice(0, 5).forEach((link) => console.log(`  - ${link.href}`))
    if (allLinks.length > 5) console.log(`  ... and ${allLinks.length - 5} more`)

    expect(allLinks.length).toBeGreaterThan(0)
    allLinks.forEach((link) => {
      expect(link.href).toMatch(/^\/docs\/examples\//)
    })
  })

  it('should filter sidebar to only User Guide items for /user-guide/ basePath', () => {
    const result = filterSidebarByBasePath(runtimeSidebar as any, '/docs/user-guide/')

    const allLinks = getAllLinks(result)
    console.log(`\nUser Guide section has ${allLinks.length} links`)

    expect(allLinks.length).toBeGreaterThan(0)
    allLinks.forEach((link) => {
      expect(link.href).toMatch(/^\/docs\/user-guide\//)
    })
  })

  it('should filter sidebar to only Community items for /community/ basePath', () => {
    const result = filterSidebarByBasePath(runtimeSidebar as any, '/docs/community/')

    const allLinks = getAllLinks(result)
    console.log(`\nCommunity section has ${allLinks.length} links`)

    expect(allLinks.length).toBeGreaterThan(0)
    allLinks.forEach((link) => {
      expect(link.href).toMatch(/^\/docs\/community\//)
    })
  })

  it('should not include User Guide or Community items when filtering for Examples', () => {
    const result = filterSidebarByBasePath(runtimeSidebar as any, '/docs/examples/')

    const allLinks = getAllLinks(result)
    const nonExamplesLinks = allLinks.filter((link) => !link.href.startsWith('/docs/examples/'))

    if (nonExamplesLinks.length > 0) {
      console.log('\nUnexpected non-examples links found:')
      nonExamplesLinks.forEach((link) => console.log(`  - ${link.href}`))
    }

    expect(nonExamplesLinks).toEqual([])
  })
})

describe('Integration: Full filtering flow', () => {
  const configPath = path.resolve('./src/config/navigation.yml')
  const docsDir = path.resolve('./src/content')
  const buildTimeSidebar = loadSidebarFromConfig(configPath, docsDir)
  const runtimeSidebar = convertToRuntimeFormat(buildTimeSidebar)

  it('should correctly filter sidebar for /examples/ page', () => {
    const currentPath = '/docs/examples/'
    const currentNav = findCurrentNavSection(currentPath, testNavLinks)

    expect(currentNav).toBeDefined()
    expect(currentNav?.label).toBe('Examples')
    expect(currentNav?.basePath).toBe('/docs/examples/')

    const basePath = currentNav?.basePath || currentNav?.href || ''
    const filtered = filterSidebarByBasePath(runtimeSidebar as any, basePath)
    const result = applyCollapse(filtered)

    const allLinks = getAllLinks(result)
    console.log(`\n/docs/examples/ page should show ${allLinks.length} sidebar links`)

    expect(allLinks.length).toBeGreaterThan(0)
    allLinks.forEach((link) => {
      expect(link.href.startsWith('/docs/examples/')).toBe(true)
    })
  })

  it('should correctly filter sidebar for nested /examples/python/weather_forecaster/ page', () => {
    const currentPath = '/docs/examples/python/weather_forecaster/'
    const currentNav = findCurrentNavSection(currentPath, testNavLinks)

    expect(currentNav).toBeDefined()
    expect(currentNav?.label).toBe('Examples')

    const basePath = currentNav?.basePath || currentNav?.href || ''
    const filtered = filterSidebarByBasePath(runtimeSidebar as any, basePath)
    const result = applyCollapse(filtered)

    const allLinks = getAllLinks(result)
    expect(allLinks.length).toBeGreaterThan(0)
    allLinks.forEach((link) => {
      expect(link.href.startsWith('/docs/examples/')).toBe(true)
    })
  })

  it('should correctly filter sidebar for /community/community-packages/ page', () => {
    const currentPath = '/docs/community/community-packages/'
    const currentNav = findCurrentNavSection(currentPath, testNavLinks)

    expect(currentNav).toBeDefined()
    expect(currentNav?.label).toBe('Community')

    const basePath = currentNav?.basePath || currentNav?.href || ''
    const filtered = filterSidebarByBasePath(runtimeSidebar as any, basePath)
    const result = applyCollapse(filtered)

    const allLinks = getAllLinks(result)
    expect(allLinks.length).toBeGreaterThan(0)
    allLinks.forEach((link) => {
      expect(link.href.startsWith('/docs/community/')).toBe(true)
    })
  })
})

describe('applyCollapse', () => {
  it('should keep top-level groups expanded by default', () => {
    const input: SidebarEntry[] = [
      { type: 'group', label: 'Group 1', collapsed: false, entries: [] },
      { type: 'group', label: 'Group 2', collapsed: false, entries: [] },
    ]

    const result = applyCollapse(input as any)

    expect((result[0] as SidebarGroup).collapsed).toBe(false)
    expect((result[1] as SidebarGroup).collapsed).toBe(false)
  })

  it('should respect explicit collapsed: true at top level', () => {
    const input: SidebarEntry[] = [
      { type: 'group', label: 'Group 1', collapsed: true, entries: [] },
      { type: 'group', label: 'Group 2', collapsed: false, entries: [] },
    ]

    const result = applyCollapse(input as any)

    expect((result[0] as SidebarGroup).collapsed).toBe(true)
    expect((result[1] as SidebarGroup).collapsed).toBe(false)
  })

  it('should collapse nested groups by default when no explicit value', () => {
    // Note: in production Starlight pre-normalizes all unset collapsed to false,
    // so this path (no collapsed property) is only exercised outside Starlight.
    const nested = { type: 'group' as const, label: 'Nested', entries: [] }
    const input = [{ type: 'group' as const, label: 'Top', entries: [nested] }]

    const result = applyCollapse(input as any)

    const top = result[0] as SidebarGroup
    expect(top.collapsed).toBe(false)
    expect((top.entries[0] as SidebarGroup).collapsed).toBe(true)
  })

  it('should collapse nested groups when Starlight has normalized collapsed to false', () => {
    // Starlight normalizes unset collapsed to false before the middleware runs,
    // making collapsed: false indistinguishable from "not set". Depth-based
    // default still applies — only explicit collapsed: true in navigation.yml
    // can override it.
    const nested: SidebarGroup = { type: 'group', label: 'Nested', collapsed: false, entries: [] }
    const input: SidebarEntry[] = [{ type: 'group', label: 'Top', collapsed: false, entries: [nested] }]

    const result = applyCollapse(input as any)

    const top = result[0] as SidebarGroup
    expect(top.collapsed).toBe(false)
    expect((top.entries[0] as SidebarGroup).collapsed).toBe(true)
  })

  it('should not modify links', () => {
    const input: SidebarEntry[] = [{ type: 'link', label: 'Link', href: '/link/', isCurrent: false }]

    const result = applyCollapse(input as any)

    expect(result[0]!.type).toBe('link')
    expect((result[0] as SidebarLink).href).toBe('/link/')
  })
})

describe('filterSidebarByBasePath unwrapping', () => {
  it('should unwrap single top-level group to show its entries directly', () => {
    const sidebar: SidebarEntry[] = [
      {
        type: 'group',
        label: 'Examples',
        collapsed: false,
        entries: [
          { type: 'link', label: 'Overview', href: '/examples/', isCurrent: false },
          { type: 'link', label: 'Weather', href: '/examples/weather/', isCurrent: false },
        ],
      },
      {
        type: 'group',
        label: 'User Guide',
        collapsed: false,
        entries: [
          { type: 'link', label: 'Quickstart', href: '/user-guide/quickstart/', isCurrent: false },
        ],
      },
    ]

    const result = filterSidebarByBasePath(sidebar as any, '/examples/')

    // Should unwrap the single "Examples" group and return its entries directly
    expect(result.length).toBe(2)
    expect(result[0]!.type).toBe('link')
    expect((result[0] as SidebarLink).href).toBe('/examples/')
    expect((result[1] as SidebarLink).href).toBe('/examples/weather/')
  })

  it('should not unwrap when multiple top-level items match', () => {
    const sidebar: SidebarEntry[] = [
      { type: 'link', label: 'Examples Home', href: '/examples/', isCurrent: false },
      {
        type: 'group',
        label: 'Python Examples',
        collapsed: false,
        entries: [
          { type: 'link', label: 'Weather', href: '/examples/python/weather/', isCurrent: false },
        ],
      },
    ]

    const result = filterSidebarByBasePath(sidebar as any, '/examples/')

    // Should keep both items since there are multiple matches
    expect(result.length).toBe(2)
    expect(result[0]!.type).toBe('link')
    expect(result[1]!.type).toBe('group')
  })

  it('should not unwrap a single top-level link', () => {
    const sidebar: SidebarEntry[] = [
      { type: 'link', label: 'Examples', href: '/examples/', isCurrent: false },
      { type: 'link', label: 'Other', href: '/other/', isCurrent: false },
    ]

    const result = filterSidebarByBasePath(sidebar as any, '/examples/')

    // Single link should remain as-is
    expect(result.length).toBe(1)
    expect(result[0]!.type).toBe('link')
  })
})

describe('filterSidebarByBasePath with base path in URLs', () => {
  it('should filter sidebar entries that include base path', () => {
    const sidebar: SidebarEntry[] = [
      { type: 'link', label: 'Examples', href: '/docs/examples/', isCurrent: false },
      { type: 'link', label: 'User Guide', href: '/docs/user-guide/', isCurrent: false },
    ]

    // When base path is /docs, nav basePath would be /docs/examples/
    const result = filterSidebarByBasePath(sidebar as any, '/docs/examples/')

    expect(result.length).toBe(1)
    expect((result[0] as SidebarLink).href).toBe('/docs/examples/')
  })

  it('should handle nested groups with base path', () => {
    const sidebar: SidebarEntry[] = [
      {
        type: 'group',
        label: 'Examples',
        collapsed: false,
        entries: [
          { type: 'link', label: 'Overview', href: '/docs/examples/', isCurrent: false },
          { type: 'link', label: 'Python', href: '/docs/examples/python/', isCurrent: false },
        ],
      },
      {
        type: 'group',
        label: 'User Guide',
        collapsed: false,
        entries: [
          { type: 'link', label: 'Quickstart', href: '/docs/user-guide/quickstart/', isCurrent: false },
        ],
      },
    ]

    const result = filterSidebarByBasePath(sidebar as any, '/docs/examples/')

    // Should unwrap single matching group
    expect(result.length).toBe(2)
    expect((result[0] as SidebarLink).href).toBe('/docs/examples/')
    expect((result[1] as SidebarLink).href).toBe('/docs/examples/python/')
  })
})
