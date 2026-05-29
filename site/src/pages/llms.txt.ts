import type { APIRoute } from 'astro'
import type { CollectionEntry } from 'astro:content'
import { getCollection } from 'astro:content'
import { getBase, getSiteOrigin } from '@util/links'
import { loadSidebarFromConfig, type StarlightSidebarItem } from '../sidebar'
import path from 'node:path'

// Sections to pull from sidebar (with their nav labels)
const SIDEBAR_SECTIONS = ['Docs', 'Examples', 'Community']

/**
 * Recursively extract links from sidebar items
 */
function extractLinks(
  items: StarlightSidebarItem[],
  base: string,
  depth: number = 0
): string[] {
  const lines: string[] = []
  const indent = '  '.repeat(depth)

  for (const item of items) {
    if ('slug' in item && item.slug) {
      // Internal link
      const url = `${base}/${item.slug}/index.md`
      const label = item.label || item.slug.split('/').pop() || item.slug
      lines.push(`${indent}- [${label}](${url})`)
    } else if ('link' in item && item.link) {
      // External link - skip or include as-is
      if (!item.link.startsWith('http')) {
        lines.push(`${indent}- [${item.label}](${item.link})`)
      }
    } else if ('items' in item && item.items) {
      // Group - add label and recurse
      lines.push(`${indent}- ${item.label}`)
      lines.push(...extractLinks(item.items, base, depth + 1))
    }
  }

  return lines
}

function buildLlmsTxt(docs: CollectionEntry<'docs'>[], sidebar: StarlightSidebarItem[], blogPosts: CollectionEntry<'blog'>[]): string {
  const base = getSiteOrigin() + getBase()
  const lines: string[] = []

  lines.push('# Strands Agents')
  lines.push('')
  lines.push('> Strands Agents is a simple yet powerful SDK that takes a model-driven approach to building and running AI agents. From simple conversational assistants to complex autonomous workflows, from local development to production deployment, Strands Agents scales with your needs.')
  lines.push('')

  // Process sidebar sections (User Guide, Examples, Community)
  for (const sectionName of SIDEBAR_SECTIONS) {
    const section = sidebar.find(
      (item) => 'label' in item && item.label === sectionName
    )

    if (section && 'items' in section) {
      lines.push(`## ${sectionName}`)
      lines.push('')
      lines.push(...extractLinks(section.items, base, 0))
      lines.push('')
    }
  }

  // API sections - group by python/typescript
  const apiDocs = docs.filter((doc) => doc.id.startsWith('docs/api/'))
  const pythonApi = apiDocs.filter((doc) => doc.id.startsWith('docs/api/python/'))
  const typescriptApi = apiDocs.filter((doc) => doc.id.startsWith('docs/api/typescript/'))

  if (pythonApi.length > 0) {
    lines.push(`## Api Python`)
    lines.push('')
    for (const doc of pythonApi) {
      const url = `${base}/${doc.id}/index.md`
      const title = doc.data.title || doc.id
      lines.push(`- [${title}](${url})`)
    }
    lines.push('')
  }

  if (typescriptApi.length > 0) {
    lines.push(`## Api TypeScript`)
    lines.push('')
    for (const doc of typescriptApi) {
      const url = `${base}/${doc.id}/index.md`
      const title = doc.data.title || doc.id
      lines.push(`- [${title}](${url})`)
    }
    lines.push('')
  }

  // Blog posts - sorted by date descending
  if (blogPosts.length > 0) {
    const sorted = [...blogPosts].sort((a, b) => b.data.date.getTime() - a.data.date.getTime())
    lines.push(`## Blog`)
    lines.push('')
    for (const post of sorted) {
      const url = `${base}/blog/${post.id}/index.md`
      lines.push(`- [${post.data.title}](${url})`)
    }
    lines.push('')
  }

  return lines.join('\n')
}

export const GET: APIRoute = async () => {
  const docs = await getCollection('docs')
  const sidebar = loadSidebarFromConfig(
    path.resolve('./src/config/navigation.yml'),
    path.resolve('./src/content')
  )

  const blogPosts = await getCollection('blog', ({ data }) => !data.draft)
  const content = buildLlmsTxt(docs, sidebar, blogPosts)

  return new Response(content, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
    },
  })
}
