import TurndownService from 'turndown'
import { gfm } from 'turndown-plugin-gfm'
import { isLocalLink, toRawMarkdownUrl } from './links'

export interface HtmlToMarkdownOptions {
  /** Heading style: 'setext' (underlined) or 'atx' (# prefixed) */
  headingStyle?: 'setext' | 'atx'
  /** Horizontal rule character */
  hr?: string
  /** Bullet list marker */
  bulletListMarker?: '-' | '+' | '*'
  /** Code block style: 'indented' or 'fenced' */
  codeBlockStyle?: 'indented' | 'fenced'
  /** Fence character for code blocks */
  fence?: '```' | '~~~'
  /** Emphasis delimiter */
  emDelimiter?: '_' | '*'
  /** Strong delimiter */
  strongDelimiter?: '__' | '**'
  /** Link style: 'inlined' or 'referenced' */
  linkStyle?: 'inlined' | 'referenced'
  /** Link reference style */
  linkReferenceStyle?: 'full' | 'collapsed' | 'shortcut'
}

/**
 * Creates a configured TurndownService instance for HTML to Markdown conversion.
 * Returns the service so you can add custom rules before converting.
 */
export function createTurndownService(options: HtmlToMarkdownOptions = {}): TurndownService {
  const service = new TurndownService({
    headingStyle: options.headingStyle ?? 'atx',
    hr: options.hr ?? '---',
    bulletListMarker: options.bulletListMarker ?? '-',
    codeBlockStyle: options.codeBlockStyle ?? 'fenced',
    fence: options.fence ?? '```',
    emDelimiter: options.emDelimiter ?? '*',
    strongDelimiter: options.strongDelimiter ?? '**',
    linkStyle: options.linkStyle ?? 'inlined',
    linkReferenceStyle: options.linkReferenceStyle ?? 'full',
  })

  // Add GFM plugin for tables, strikethrough, and task lists
  service.use(gfm)

  // Remove screen-reader-only elements (e.g., "Section titled X" links)
  service.addRule('removeSrOnly', {
    filter: (node) => {
      if (node.nodeType !== 1) return false
      const el = node as Element
      const className = el.getAttribute?.('class') || ''
      return className.includes('sr-only')
    },
    replacement: () => '',
  })

  // Remove script tags (JavaScript code shouldn't appear in markdown)
  service.addRule('removeScripts', {
    filter: 'script',
    replacement: () => '',
  })

  // Remove empty anchor links (e.g., <a href="#section"><svg>...</svg></a> anchor icons)
  // These are typically section anchor links with only icons, no text
  service.addRule('removeAnchorLinks', {
    filter: (node) => {
      if (node.nodeName !== 'A') return false
      const el = node as Element
      const className = el.getAttribute?.('class') || ''
      // Remove Starlight anchor links (icon-only links next to headings)
      if (className.includes('sl-anchor-link')) return true
      // Also remove any anchor link that only contains whitespace/icons
      const href = el.getAttribute?.('href') || ''
      if (href.startsWith('#')) {
        // Check if there's any actual text (not just whitespace)
        const text = el.textContent?.replace(/\s/g, '') || ''
        return text === ''
      }
      return false
    },
    replacement: () => '',
  })

  // Remove tab navigation lists (the Python/TypeScript tab buttons)
  // These render as "- [Python](#tab-panel-xxx)" lists which aren't useful in markdown
  service.addRule('removeTabList', {
    filter: (node) => {
      if (node.nodeName !== 'UL') return false
      const el = node as Element
      return el.getAttribute?.('role') === 'tablist'
    },
    replacement: () => '',
  })

  // Wrap tab panels with markers so readers know which language section they're in
  // Input: <div id="tab-panel-xxx" role="tabpanel" aria-labelledby="tab-xxx">...</div>
  // The tab label is found in the corresponding <a id="tab-xxx">Label</a>
  service.addRule('tabPanel', {
    filter: (node) => {
      if (node.nodeName !== 'DIV') return false
      const el = node as Element
      return el.getAttribute?.('role') === 'tabpanel'
    },
    replacement: (content, node) => {
      const el = node as Element
      const labelledBy = el.getAttribute?.('aria-labelledby') || ''

      // Find the tab label by looking for the corresponding tab link
      // The tab link has id matching aria-labelledby and contains the label text
      let tabLabel = ''
      if (labelledBy) {
        // Look for the tab element in the parent starlight-tabs
        const parent = el.parentElement
        if (parent) {
          const tabLink = parent.querySelector?.(`#${labelledBy}`)
          if (tabLink) {
            tabLabel = tabLink.textContent?.trim() || ''
          }
        }
      }

      if (tabLabel) {
        return `\n\n(( tab "${tabLabel}" ))\n${content.trim()}\n(( /tab "${tabLabel}" ))\n\n`
      }
      return content
    },
  })

  // Standard fenced code block rule (for non-expressive-code blocks)
  // This must be added BEFORE expressiveCodeBlock so that expressiveCodeBlock takes precedence
  // (Turndown checks rules in reverse order of addition - last added = first checked)
  service.addRule('fencedCodeBlock', {
    filter: (node, options) => {
      return (
        options.codeBlockStyle === 'fenced' &&
        node.nodeName === 'PRE' &&
        node.firstChild !== null &&
        node.firstChild.nodeName === 'CODE'
      )
    },
    replacement: (_content, node, options) => {
      const codeNode = node.firstChild as Element
      const className = codeNode.getAttribute?.('class') || ''
      // Extract language from class like "language-typescript" or "lang-ts"
      const langMatch = className.match(/(?:language-|lang-)(\w+)/)
      const language = langMatch ? langMatch[1] : ''
      const code = codeNode.textContent || ''

      const fence = options.fence || '```'
      return `\n\n${fence}${language}\n${code.replace(/\n$/, '')}\n${fence}\n\n`
    },
  })

  // Custom rule for syntax-highlighted code blocks (expressive-code format)
  // These have: <pre data-language="python"><code><div class="ec-line">...</div></code></pre>
  // This rule is added AFTER fencedCodeBlock so it takes precedence (Turndown checks last-added first)
  service.addRule('expressiveCodeBlock', {
    filter: (node) => {
      if (node.nodeName !== 'PRE') return false
      const lang = node.getAttribute?.('data-language')
      return lang != null
    },
    replacement: (_content, node, options) => {
      const language = node.getAttribute?.('data-language') || ''
      const fence = options.fence || '```'

      // Extract lines from ec-line divs
      const lines: string[] = []
      function walk(el: Element | ChildNode) {
        if (el.nodeType === 1) {
          const element = el as Element
          const className = element.getAttribute?.('class') || ''
          if (className.includes('ec-line')) {
            lines.push(element.textContent?.replace(/\n/g, '') || '')
          } else {
            const children = element.childNodes || []
            for (let i = 0; i < children.length; i++) {
              walk(children[i] as Element)
            }
          }
        }
      }
      walk(node as Element)

      const code = lines.length > 0 ? lines.join('\n') : (node.textContent || '')
      return `\n\n${fence}${language}\n${code}\n${fence}\n\n`
    },
  })

  // Rewrite local links to point to raw.md endpoints
  // Transforms: <a href="/user-guide/foo/"> -> [text](/user-guide/foo/raw.md)
  service.addRule('rewriteLocalLinks', {
    filter: (node) => {
      if (node.nodeName !== 'A') return false
      const el = node as Element
      const href = el.getAttribute?.('href') || ''
      return isLocalLink(href)
    },
    replacement: (content, node) => {
      const el = node as Element
      const href = el.getAttribute?.('href') || ''
      const title = el.getAttribute?.('title')
      const newHref = toRawMarkdownUrl(href)

      return title ? `[${content}](${newHref} "${title}")` : `[${content}](${newHref})`
    },
  })

  return service
}

/**
 * Converts HTML string to Markdown.
 */
export function htmlToMarkdown(html: string, options: HtmlToMarkdownOptions = {}): string {
  const service = createTurndownService(options)
  return service.turndown(html)
}

/**
 * Converts HTML to Markdown with custom rules.
 * Use this when you need to add custom transformation rules.
 *
 * @example
 * ```ts
 * const markdown = htmlToMarkdownWithRules(html, (service) => {
 *   service.addRule('customDiv', {
 *     filter: (node) => node.nodeName === 'DIV' && node.classList.contains('note'),
 *     replacement: (content) => `:::note\n${content}\n:::\n`
 *   })
 * })
 * ```
 */
export function htmlToMarkdownWithRules(
  html: string,
  configureService: (service: TurndownService) => void,
  options: HtmlToMarkdownOptions = {}
): string {
  const service = createTurndownService(options)
  configureService(service)
  return service.turndown(html)
}
