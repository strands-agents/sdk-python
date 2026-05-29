import { describe, it, expect } from 'vitest'
import path from 'node:path'
import {
  loadSidebarFromConfig,
  loadNavigationConfig,
  loadNavbarFromConfig,
  loadGitHubSectionsFromConfig,
  type StarlightSidebarItem,
} from '../src/sidebar'

const pathToNavigationYml = path.resolve('./src/config/navigation.yml')

describe('Sidebar Generation from navigation.yml', () => {
  it('should generate sidebar structure from navigation.yml', () => {
    const sidebar = loadSidebarFromConfig(pathToNavigationYml)

    console.log('\n=== Generated Sidebar Structure ===\n')
    console.log(JSON.stringify(sidebar, null, 2))

    expect(sidebar).toBeDefined()
    expect(Array.isArray(sidebar)).toBe(true)
    expect(sidebar.length).toBeGreaterThan(0)
  })

  it('should load navigation config with all sections', () => {
    const config = loadNavigationConfig(pathToNavigationYml)

    expect(config).toBeDefined()
    expect(config.navbar).toBeDefined()
    expect(Array.isArray(config.navbar)).toBe(true)
    expect(config.sidebar).toBeDefined()
    expect(Array.isArray(config.sidebar)).toBe(true)
    expect(config.github).toBeDefined()
    expect(config.github.sections).toBeDefined()
  })

  it('should load navbar links', () => {
    const navbar = loadNavbarFromConfig(pathToNavigationYml)

    expect(navbar).toBeDefined()
    expect(Array.isArray(navbar)).toBe(true)
    expect(navbar.length).toBeGreaterThan(0)

    // Check that navbar links have required properties
    const firstLink = navbar[0]
    expect(firstLink).toHaveProperty('label')
    expect(firstLink).toHaveProperty('href')
  })

  it('should load GitHub sections', () => {
    const sections = loadGitHubSectionsFromConfig(pathToNavigationYml)

    expect(sections).toBeDefined()
    expect(Array.isArray(sections)).toBe(true)
    expect(sections.length).toBeGreaterThan(0)

    // Check that sections have required structure
    const firstSection = sections[0]
    expect(firstSection).toHaveProperty('title')
    expect(firstSection).toHaveProperty('links')
    expect(Array.isArray(firstSection.links)).toBe(true)
  })

  it('should have correct top-level sidebar sections', () => {
    const sidebar = loadSidebarFromConfig(pathToNavigationYml)

    // Check that we have the expected top-level sections
    const topLevelLabels = sidebar
      .filter((item): item is StarlightSidebarItem & { label: string } => 'label' in item)
      .map((item) => item.label)

    expect(topLevelLabels).toContain('Docs')
    expect(topLevelLabels).toContain('Examples')
    expect(topLevelLabels).toContain('Community')
  })

  it('should not set collapsed on groups unless explicitly specified in YAML', () => {
    const sidebar = loadSidebarFromConfig(pathToNavigationYml)

    // Find the Docs section
    const docs = sidebar.find(
      (item): item is StarlightSidebarItem & { label: string; items: StarlightSidebarItem[] } =>
        'label' in item && item.label === 'Docs'
    )

    expect(docs).toBeDefined()
    if (docs) {
      // Top level should not have collapsed set (middleware handles depth-based defaults)
      expect(docs).not.toHaveProperty('collapsed')

      // Find a nested group (like "Get Started") — no explicit collapsed in YAML
      const getStarted = docs.items.find(
        (item): item is StarlightSidebarItem & { label: string } => 'label' in item && item.label === 'Get Started'
      )

      // Nested groups without explicit YAML collapsed flag should also lack the property
      expect(getStarted).not.toHaveProperty('collapsed')
    }
  })

  it('should support both labeled and unlabeled leaf items', () => {
    const sidebar = loadSidebarFromConfig(pathToNavigationYml)

    // Collect all leaf items
    function findLeafItems(items: StarlightSidebarItem[]): StarlightSidebarItem[] {
      const leaves: StarlightSidebarItem[] = []
      for (const item of items) {
        if ('slug' in item && !('items' in item)) {
          leaves.push(item)
        }
        if ('items' in item) {
          leaves.push(...findLeafItems(item.items as StarlightSidebarItem[]))
        }
      }
      return leaves
    }

    const leaves = findLeafItems(sidebar)
    expect(leaves.length).toBeGreaterThan(0)

    // Some leaves should have labels (Build section items)
    const labeled = leaves.filter((item) => 'label' in item)
    expect(labeled.length).toBeGreaterThan(0)

    // Some leaves should not have labels (plain slug items)
    const unlabeled = leaves.filter((item) => !('label' in item))
    expect(unlabeled.length).toBeGreaterThan(0)
  })

  it('should include Labs and Contribute under Community', () => {
    const sidebar = loadSidebarFromConfig(pathToNavigationYml)

    // Find the Community section
    const community = sidebar.find(
      (item): item is StarlightSidebarItem & { label: string; items: StarlightSidebarItem[] } =>
        'label' in item && item.label === 'Community'
    )

    expect(community).toBeDefined()
    if (community) {
      const subLabels = community.items
        .filter((item): item is StarlightSidebarItem & { label: string } => 'label' in item)
        .map((item) => item.label)

      expect(subLabels).toContain('Labs')
      expect(subLabels).toContain('Contribute')
    }
  })
})
