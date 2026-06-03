import { describe, it, expect } from 'vitest'
import { getCollection } from 'astro:content'
import path from 'node:path'
import { loadSidebarFromConfig, type StarlightSidebarItem } from '../src/sidebar'

describe('Content Collections', () => {
  it('should list all doc contents', async () => {
    const docs = await getCollection('docs')

    console.log('\n=== All Docs ===\n')
    console.log(`Total: ${docs.length} documents\n`)

    for (const doc of docs) {
      console.log(`- ${doc.id}`)
      console.log(`  Title: ${doc.data.title}`)
      if (doc.data.languages) console.log(`  Languages: ${doc.data.languages}`)
      if (doc.data.community) console.log(`  Community: ${doc.data.community}`)
      if (doc.data.experimental) console.log(`  Experimental: ${doc.data.experimental}`)
    }

    expect(docs.length).toBeGreaterThan(0)
  })

  it('should have valid slugs for all sidebar items', async () => {
    const configPath = path.resolve('./src/config/navigation.yml')
    const docsDir = path.resolve('./docs')
    const sidebar = loadSidebarFromConfig(configPath, docsDir)
    const docs = await getCollection('docs')

    // Build a set of valid doc IDs
    const validIds = new Set(docs.map((doc) => doc.id))

    // Extract all slugs from the sidebar
    function extractSlugs(items: StarlightSidebarItem[]): string[] {
      const slugs: string[] = []
      for (const item of items) {
        if ('slug' in item && item.slug) {
          slugs.push(item.slug)
        }
        if ('items' in item) {
          slugs.push(...extractSlugs(item.items))
        }
      }
      return slugs
    }

    const slugs = extractSlugs(sidebar)

    const invalidSlugs: string[] = []
    for (const slug of slugs) {
      if (!validIds.has(slug)) {
        invalidSlugs.push(slug)
      }
    }

    if (invalidSlugs.length > 0) {
      console.log('\n=== Invalid Sidebar Slugs (no content entry found) ===\n')
      for (const slug of invalidSlugs) {
        console.log(`- ${slug}`)
      }
    }

    expect(invalidSlugs).toEqual([])
  })

  it('has unique titles across user-guide pages', async () => {
    // Related-pages output and JSON-LD `headline` use `title` verbatim. Two
    // user-guide pages with the same title produce ambiguous links that look
    // like duplicates (e.g. a "Hooks" that could mean the agents or bidi one).
    // Set `sidebar.label` if you want the sidebar to stay terse while the page
    // title remains unambiguous out-of-context.
    const docs = await getCollection('docs')
    const userGuide = docs.filter((d) => d.id.startsWith('docs/user-guide/'))

    const titleMap = new Map<string, string[]>()
    for (const doc of userGuide) {
      const slugs = titleMap.get(doc.data.title) ?? []
      slugs.push(doc.id)
      titleMap.set(doc.data.title, slugs)
    }

    const collisions = [...titleMap.entries()].filter(([, slugs]) => slugs.length > 1)
    if (collisions.length > 0) {
      console.log('\n=== Duplicate user-guide titles ===\n')
      for (const [title, slugs] of collisions) {
        console.log(`"${title}"`)
        for (const s of slugs) console.log(`  ${s}`)
      }
    }

    expect(collisions).toEqual([])
  })
})
