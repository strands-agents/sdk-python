import type { APIRoute } from 'astro'
import { getCollection } from 'astro:content'
import { renderEntryToMarkdown } from '@util/render-to-markdown'
import { getBase, getSiteOrigin } from '@util/links'

export const GET: APIRoute = async () => {
  const allDocs = await getCollection('docs')
  // Exclude API documentation from full content
  const docs = allDocs.filter((doc) => !doc.id.startsWith('docs/api/'))
  const base = getSiteOrigin() + getBase()
  const lines: string[] = []

  lines.push('# Strands Agents')
  lines.push('')
  lines.push('> Strands Agents is a simple yet powerful SDK that takes a model-driven approach to building and running AI agents. From simple conversational assistants to complex autonomous workflows, from local development to production deployment, Strands Agents scales with your needs.')
  lines.push('')

  // Render each doc's full content
  for (const doc of docs) {
    const title = doc.data.title || doc.id
    lines.push(`## ${title}`)
    lines.push('')

    const { markdown } = await renderEntryToMarkdown(doc)
    lines.push(markdown.trim())

    lines.push('')
    lines.push(`Source: ${base}/${doc.id}/index.md`)
    lines.push('')
    lines.push('---')
    lines.push('')
  }

  // Render blog posts
  const blogPosts = await getCollection('blog', ({ data }) => !data.draft)
  const sortedPosts = [...blogPosts].sort((a, b) => b.data.date.getTime() - a.data.date.getTime())

  for (const post of sortedPosts) {
    lines.push(`## ${post.data.title}`)
    lines.push('')
    lines.push(`Date: ${post.data.date.toISOString()}`)
    lines.push(`Tags: ${post.data.tags.join(', ')}`)
    lines.push('')

    const { markdown } = await renderEntryToMarkdown(post, `/blog/${post.id}/`)
    lines.push(markdown.trim())

    lines.push('')
    lines.push(`Source: ${base}/blog/${post.id}/index.md`)
    lines.push('')
    lines.push('---')
    lines.push('')
  }

  return new Response(lines.join('\n'), {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
    },
  })
}
