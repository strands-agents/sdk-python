/**
 * HTML → markdown extraction for the web fetch tool.
 *
 * Delegates HTML-to-markdown conversion to turndown with ATX headings and
 * fenced code blocks. Non-content elements (scripts, styles, media, forms,
 * etc.) are dropped entirely. `data:` URI images are replaced with their alt
 * text (or nothing) to avoid enormous blobs in the output. `javascript:`
 * hrefs and src values are stripped silently.
 */

import TurndownService from 'turndown'

// Elements whose entire subtree is discarded.
// Cast needed because svg/noscript/template are absent from HTMLElementTagNameMap.
const DROPPED_ELEMENTS = [
  'script',
  'style',
  'noscript',
  'template',
  'svg',
  'canvas',
  'iframe',
  'object',
  'embed',
  'video',
  'audio',
  'form',
  'input',
  'button',
  'select',
  'textarea',
] as unknown as TurndownService.Filter

function getListDepth(node: HTMLElement): number {
  let depth = 0
  let current: HTMLElement | null = node
  while (current) {
    const tag = current.nodeName.toLowerCase()
    if (tag === 'ul' || tag === 'ol') depth += 1
    current = current.parentNode as HTMLElement | null
  }
  return depth
}

function buildTurndownService(): TurndownService {
  const td = new TurndownService({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced',
    bulletListMarker: '-',
    emDelimiter: '*',
    strongDelimiter: '**',
    linkStyle: 'inlined',
  })

  td.remove(DROPPED_ELEMENTS)

  // Replace data: URI images with alt text only (they can be enormous blobs).
  // Strip javascript: src values entirely.
  td.addRule('safeImage', {
    filter: 'img',
    replacement: (_content, node) => {
      const element = node as HTMLImageElement
      const src = (element.getAttribute('src') ?? '').trim()
      const alt = element.getAttribute('alt') ?? ''
      if (!src) return alt
      if (src.trimStart().toLowerCase().startsWith('javascript:')) return ''
      if (src.trimStart().toLowerCase().startsWith('data:')) return alt
      return `![${alt}](${src})`
    },
  })

  // Strip javascript: hrefs — render link text only.
  td.addRule('safeLink', {
    filter: (node) => {
      if (node.nodeName !== 'A') return false
      const href = ((node as HTMLAnchorElement).getAttribute('href') ?? '').trim()
      return href.trimStart().toLowerCase().startsWith('javascript:')
    },
    replacement: (content) => content,
  })

  // Compact list-item markers: "- item" / "1. item" rather than turndown's
  // default padded form ("-   item" / "1.  item").
  td.addRule('compactListItem', {
    filter: 'li',
    replacement: (content, node) => {
      content = content.replace(/^\n+/, '').replace(/\n+$/, '\n')
      const parentName = (node.parentNode?.nodeName ?? '').toLowerCase()
      const prefix =
        parentName === 'ol'
          ? `${Array.from((node.parentNode as HTMLElement).children).indexOf(node as HTMLElement) + 1}. `
          : '- '
      const indent = '  '.repeat(Math.max(0, getListDepth(node as HTMLElement) - 1))
      return `${indent}${prefix}${content.replace(/\n/g, `\n${indent}  `)}\n`
    },
  })

  return td
}

/**
 * Convert HTML to markdown suitable for a model to read.
 *
 * @param html - Raw HTML string to convert.
 * @returns `{ title, markdown }` — both trimmed of surrounding whitespace.
 */
export function htmlToMarkdown(html: string): { title: string; markdown: string } {
  // Extract <title> before handing to turndown, which would otherwise include
  // it in the output since <title> is a head element turndown doesn't handle.
  const titleMatch = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html)
  const title = titleMatch ? titleMatch[1]?.replace(/\s+/g, ' ').trim() ?? '' : ''

  const td = buildTurndownService()
  const raw = td.turndown(html)

  // Collapse runs of more than one blank line down to a single blank line,
  // matching the original output's blank-line normalization.
  const collapsed = raw.replace(/\n{3,}/g, '\n\n').trim()
  const markdown = collapsed ? collapsed + '\n' : ''

  return { title, markdown }
}
