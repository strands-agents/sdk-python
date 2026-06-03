import fs from 'node:fs'
import path from 'node:path'
import yaml from 'js-yaml'

// Starlight sidebar item types
export type StarlightSidebarItem =
  | { slug: string; label?: string; attrs?: Record<string, string> } // Internal link
  | { label: string; link: string; attrs?: Record<string, string> } // External link
  | { label: string; items: StarlightSidebarItem[]; collapsed?: boolean } // Group

// Navigation config types
interface NavConfigItem {
  label?: string
  items?: NavConfigItem[]
  slug?: string // For labeled leaf items "Adding Tools"
  collapsed?: boolean // Explicit collapse state for groups (overrides auto-collapse)
}
type NavConfigEntry = string | NavConfigItem

interface NavbarLink {
  label: string
  href: string
  basePath?: string
  external?: boolean
}

export interface GitHubLink {
  label: string
  href: string
  icon?: string
}

export interface GitHubSection {
  title: string
  links: GitHubLink[]
}

interface NavigationConfig {
  navbar: NavbarLink[]
  sidebar: NavConfigItem[]
  github: {
    sections: GitHubSection[]
  }
}

interface ConvertContext {
  contentDir: string
}

/**
 * Check if a content file exists for a given slug
 */
function contentExists(slug: string, contentDir: string): boolean {
  if (!contentDir) return true

  const candidates = [
    path.join(contentDir, `${slug}.md`),
    path.join(contentDir, `${slug}.mdx`),
    path.join(contentDir, slug, 'index.md'),
    path.join(contentDir, slug, 'index.mdx'),
  ]

  return candidates.some((p) => fs.existsSync(p))
}

/**
 * Convert a navigation config item to Starlight sidebar format
 */
function convertConfigItem(item: NavConfigEntry, ctx: ConvertContext): StarlightSidebarItem | null {
  // String: slug only (e.g., "docs/user-guide/quickstart/overview")
  if (typeof item === 'string') {
    if (!contentExists(item, ctx.contentDir)) return null
    return { slug: item }
  }

  // Object with label and items (group)
  if (typeof item === 'object' && item !== null) {
    if (item.label && item.items) {
      const children = item.items
        .map((child) => convertConfigItem(child, ctx))
        .filter((c): c is StarlightSidebarItem => c !== null)

      if (children.length === 0) return null

      return {
        label: item.label,
        items: children,
        ...(typeof item.collapsed === 'boolean' && { collapsed: item.collapsed }),
      }
    }

    // Object with label and slug (labeled leaf item)
    if (item.label && item.slug) {
      if (!contentExists(item.slug, ctx.contentDir)) return null
      return { slug: item.slug, label: item.label }
    }
  }

  return null
}

/**
 * Load navigation configuration from YAML file
 */
export function loadNavigationConfig(configPath: string): NavigationConfig {
  const content = fs.readFileSync(configPath, 'utf-8')
  return yaml.load(content) as NavigationConfig
}

/**
 * Load sidebar from navigation.yml config
 */
export function loadSidebarFromConfig(configPath: string, docsContentDir?: string): StarlightSidebarItem[] {
  const config = loadNavigationConfig(configPath)
  if (!config.sidebar) return []

  const ctx: ConvertContext = { contentDir: docsContentDir || '' }

  const items = config.sidebar
    .map((item) => convertConfigItem(item, ctx))
    .filter((i): i is StarlightSidebarItem => i !== null)

  return items
}

/**
 * Load navbar links from navigation.yml config
 */
export function loadNavbarFromConfig(configPath: string): NavbarLink[] {
  const config = loadNavigationConfig(configPath)
  return config.navbar || []
}

/**
 * Load GitHub sections from navigation.yml config
 */
export function loadGitHubSectionsFromConfig(configPath: string): GitHubSection[] {
  const config = loadNavigationConfig(configPath)
  return config.github?.sections || []
}
