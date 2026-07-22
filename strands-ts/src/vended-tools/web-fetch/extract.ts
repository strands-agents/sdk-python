/**
 * HTML → markdown extraction for the web fetch tool.
 *
 * The goal is not a perfect readability heuristic — it is producing output
 * that is safe and useful for a model to read. We strip script, style, and
 * other non-content elements entirely, drop `data:` URI images (which can be
 * huge blobs), and preserve headings, links, lists, blockquotes, code, and
 * paragraph text as GitHub-flavored markdown.
 *
 * No third-party dependency: we hand-tokenize the input. Because we never
 * execute anything, the parser is safe against script content, event handler
 * attributes, and other active HTML features by construction — we simply
 * ignore anything we don't recognize.
 */

// Elements whose content is discarded entirely, not just unwrapped.
const DROPPED_ELEMENTS = new Set([
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
])

// HTML void elements never emit a close tag; they must not push depth on the
// drop counter, or a void child inside a dropped element (e.g. <input> inside
// <form>) would leak the drop state past its enclosing </form>.
const VOID_ELEMENTS = new Set([
  'area',
  'base',
  'br',
  'col',
  'embed',
  'hr',
  'img',
  'input',
  'link',
  'meta',
  'param',
  'source',
  'track',
  'wbr',
])

const HEADING_LEVELS: Record<string, number> = {
  h1: 1,
  h2: 2,
  h3: 3,
  h4: 4,
  h5: 5,
  h6: 6,
}

interface Attr {
  name: string
  value: string | null
}

interface TagToken {
  kind: 'tag'
  name: string
  attrs: Attr[]
  closing: boolean
  selfClosing: boolean
}

interface TextToken {
  kind: 'text'
  value: string
}

type Token = TagToken | TextToken

const NAMED_ENTITIES: Record<string, string> = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  nbsp: ' ',
}

function decodeEntities(input: string): string {
  return input.replace(/&(#x[0-9a-fA-F]+|#[0-9]+|[a-zA-Z]+);/g, (match, body: string) => {
    if (body.startsWith('#x') || body.startsWith('#X')) {
      const code = Number.parseInt(body.slice(2), 16)
      return Number.isFinite(code) ? safeCodePoint(code) : match
    }
    if (body.startsWith('#')) {
      const code = Number.parseInt(body.slice(1), 10)
      return Number.isFinite(code) ? safeCodePoint(code) : match
    }
    const named = NAMED_ENTITIES[body.toLowerCase()]
    return named ?? match
  })
}

function safeCodePoint(code: number): string {
  if (code < 0 || code > 0x10ffff) return ''
  try {
    return String.fromCodePoint(code)
  } catch {
    return ''
  }
}

/**
 * Tokenize HTML into a flat stream of `<tag>` and text tokens.
 *
 * Not a general HTML5 tokenizer — good enough for the block/inline elements
 * we render. Script/style bodies are consumed as a single text run and
 * discarded by the caller, which prevents accidental script text ending up in
 * the output.
 */
function tokenize(html: string): Token[] {
  const tokens: Token[] = []
  let i = 0
  while (i < html.length) {
    if (html[i] === '<') {
      // Comment
      if (html.startsWith('<!--', i)) {
        const end = html.indexOf('-->', i + 4)
        i = end === -1 ? html.length : end + 3
        continue
      }
      // Doctype / CDATA / other markup declaration
      if (html.startsWith('<!', i) || html.startsWith('<?', i)) {
        const end = html.indexOf('>', i + 2)
        i = end === -1 ? html.length : end + 1
        continue
      }
      // Tag
      const closing = html.charAt(i + 1) === '/'
      const nameStart = i + (closing ? 2 : 1)
      let nameEnd = nameStart
      while (nameEnd < html.length && /[A-Za-z0-9]/.test(html.charAt(nameEnd))) {
        nameEnd += 1
      }
      const name = html.slice(nameStart, nameEnd).toLowerCase()
      if (!name) {
        // "<" that isn't a real tag — treat as text.
        tokens.push({ kind: 'text', value: '<' })
        i += 1
        continue
      }

      // Parse attributes until we hit '>' or self-close.
      let cursor = nameEnd
      const attrs: Attr[] = []
      let selfClosing = false
      while (cursor < html.length && html.charAt(cursor) !== '>') {
        // Skip whitespace
        while (cursor < html.length && /\s/.test(html.charAt(cursor))) cursor += 1
        if (cursor >= html.length || html.charAt(cursor) === '>') break
        if (html.charAt(cursor) === '/') {
          selfClosing = true
          cursor += 1
          continue
        }
        // Attribute name
        const attrStart = cursor
        while (cursor < html.length && !/[\s/=>]/.test(html.charAt(cursor))) cursor += 1
        const attrName = html.slice(attrStart, cursor).toLowerCase()
        // Skip whitespace, optional '='
        while (cursor < html.length && /\s/.test(html.charAt(cursor))) cursor += 1
        let attrValue: string | null = null
        if (html.charAt(cursor) === '=') {
          cursor += 1
          while (cursor < html.length && /\s/.test(html.charAt(cursor))) cursor += 1
          const quote = html.charAt(cursor)
          if (quote === '"' || quote === "'") {
            cursor += 1
            const valueStart = cursor
            while (cursor < html.length && html.charAt(cursor) !== quote) cursor += 1
            attrValue = html.slice(valueStart, cursor)
            if (cursor < html.length) cursor += 1
          } else {
            const valueStart = cursor
            while (cursor < html.length && !/[\s>]/.test(html.charAt(cursor))) cursor += 1
            attrValue = html.slice(valueStart, cursor)
          }
          attrValue = decodeEntities(attrValue)
        }
        if (attrName) attrs.push({ name: attrName, value: attrValue })
      }
      // Consume '>'
      if (html.charAt(cursor) === '>') cursor += 1
      i = cursor

      tokens.push({ kind: 'tag', name, attrs, closing, selfClosing })

      // For script/style, consume raw content up to matching close tag so we
      // don't accidentally interpret its contents as HTML.
      if (!closing && !selfClosing && (name === 'script' || name === 'style')) {
        const closeTag = `</${name}`
        const lower = html.toLowerCase()
        const closeIndex = lower.indexOf(closeTag, i)
        if (closeIndex === -1) {
          // Unterminated -- discard the rest.
          i = html.length
        } else {
          // Push the raw content as text so DROPPED_ELEMENTS logic in the
          // renderer discards it. Do NOT decode entities in <script>/<style>.
          tokens.push({ kind: 'text', value: html.slice(i, closeIndex) })
          i = closeIndex
        }
      }
      continue
    }

    // Text run
    const next = html.indexOf('<', i)
    const end = next === -1 ? html.length : next
    const raw = html.slice(i, end)
    tokens.push({ kind: 'text', value: decodeEntities(raw) })
    i = end
  }
  return tokens
}

function attr(attrs: Attr[], name: string): string | null {
  for (const a of attrs) if (a.name === name) return a.value
  return null
}

/**
 * Convert HTML to markdown suitable for a model to read.
 *
 * @returns `{ title, markdown }` — both trimmed of surrounding whitespace.
 */
// Sentinel markers used during extraction to record where a blockquote begins
// and ends. A post-processing pass prefixes every line between the markers
// with `> ` so nested block elements (e.g. `<blockquote><p>...</p></blockquote>`)
// stay inside the blockquote in the emitted markdown.
const BQ_OPEN = '\x00BQ_OPEN\x00'
const BQ_CLOSE = '\x00BQ_CLOSE\x00'

export function htmlToMarkdown(html: string): { title: string; markdown: string } {
  const tokens = tokenize(html)
  const out: string[] = []
  const inline: string[] = []
  const listStack: ('ul' | 'ol')[] = []
  const olCounters: number[] = []
  const linkHrefs: (string | null)[] = []
  let dropDepth = 0
  let preDepth = 0
  let codeDepth = 0
  let inTitle = false
  const titleParts: string[] = []

  const flushInline = (): void => {
    if (inline.length === 0) return
    const text = inline.join('').trim()
    inline.length = 0
    if (text) out.push(text)
  }

  const pushImg = (attrs: Attr[]): void => {
    const src = (attr(attrs, 'src') ?? '').trim()
    const altText = attr(attrs, 'alt') ?? ''
    if (!src) return
    const lower = src.trimStart().toLowerCase()
    // data: URIs can be enormous blobs — dropping them is a size defense as
    // well as noise reduction.
    if (lower.startsWith('data:')) {
      if (altText) inline.push(altText)
      return
    }
    if (lower.startsWith('javascript:')) return
    inline.push(`![${altText}](${src})`)
  }

  for (const tok of tokens) {
    if (tok.kind === 'text') {
      if (dropDepth > 0) continue
      if (inTitle) {
        titleParts.push(tok.value)
        continue
      }
      if (preDepth > 0) {
        out.push(tok.value)
        continue
      }
      if (codeDepth > 0) {
        inline.push(tok.value.replace(/`/g, ''))
        continue
      }
      const raw = tok.value
      const normalized = raw.replace(/\s+/g, ' ')
      if (normalized === ' ' || normalized === '') {
        if (raw.length > 0 && /\s/.test(raw.charAt(0)) && inline.length > 0) {
          const last = inline[inline.length - 1] ?? ''
          if (!last.endsWith(' ')) inline.push(' ')
        }
        continue
      }
      const leading = /^\s/.test(raw) && inline.length > 0 ? ' ' : ''
      const trailing = /\s$/.test(raw) ? ' ' : ''
      inline.push(`${leading}${normalized.trim()}${trailing}`)
      continue
    }

    // tag token
    const { name, attrs, closing, selfClosing } = tok

    if (name === 'title') {
      inTitle = !closing
      continue
    }

    if (dropDepth > 0) {
      // Only adjust depth on non-void dropped elements; a void tag like <input>
      // never fires a matching close, and self-closing tags are one-shot.
      if (DROPPED_ELEMENTS.has(name) && !VOID_ELEMENTS.has(name) && !selfClosing) {
        if (closing) dropDepth -= 1
        else dropDepth += 1
      }
      continue
    }
    if (!closing && !selfClosing && DROPPED_ELEMENTS.has(name) && !VOID_ELEMENTS.has(name)) {
      dropDepth = 1
      continue
    }
    // Void + dropped (e.g. <input>) outside a drop -- nothing to do.
    if (!closing && DROPPED_ELEMENTS.has(name) && VOID_ELEMENTS.has(name)) {
      continue
    }

    if (name === 'br') {
      if (!closing) inline.push('  \n')
      continue
    }
    if (name === 'hr') {
      if (!closing) {
        flushInline()
        out.push('\n---\n\n')
      }
      continue
    }
    if (name === 'img') {
      if (!closing) pushImg(attrs)
      continue
    }

    const headingLevel = HEADING_LEVELS[name]
    if (headingLevel !== undefined) {
      flushInline()
      if (!closing) out.push('\n' + '#'.repeat(headingLevel) + ' ')
      else out.push('\n\n')
      continue
    }
    if (name === 'p') {
      flushInline()
      out.push(closing ? '\n\n' : '\n')
      continue
    }
    if (name === 'blockquote') {
      flushInline()
      // Emit sentinels rather than a bare `> ` prefix. A post-processing pass
      // rewrites every line between BQ_OPEN and BQ_CLOSE with a `> ` prefix,
      // which correctly quotes nested block elements like <p> and lists.
      out.push(closing ? `\n${BQ_CLOSE}\n\n` : `\n${BQ_OPEN}\n`)
      continue
    }
    if (name === 'ul') {
      flushInline()
      if (!closing) listStack.push('ul')
      else {
        if (listStack[listStack.length - 1] === 'ul') listStack.pop()
        out.push('\n')
      }
      continue
    }
    if (name === 'ol') {
      flushInline()
      if (!closing) {
        listStack.push('ol')
        olCounters.push(0)
      } else {
        if (listStack[listStack.length - 1] === 'ol') {
          listStack.pop()
          olCounters.pop()
        }
        out.push('\n')
      }
      continue
    }
    if (name === 'li') {
      flushInline()
      if (!closing) {
        const depth = Math.max(0, listStack.length - 1)
        const indent = '  '.repeat(depth)
        if (listStack[listStack.length - 1] === 'ol') {
          const idx = olCounters.length - 1
          const next = (olCounters[idx] ?? 0) + 1
          olCounters[idx] = next
          out.push(`${indent}${next}. `)
        } else {
          out.push(`${indent}- `)
        }
      } else {
        out.push('\n')
      }
      continue
    }
    if (name === 'pre') {
      flushInline()
      if (!closing) {
        preDepth += 1
        out.push('\n```\n')
      } else {
        if (preDepth > 0) preDepth -= 1
        out.push('```\n\n')
      }
      continue
    }
    if (name === 'code') {
      if (!closing) codeDepth += 1
      else if (codeDepth > 0) codeDepth -= 1
      if (preDepth === 0) inline.push('`')
      continue
    }
    if (name === 'a') {
      if (!closing) {
        const href = (attr(attrs, 'href') ?? '').trim()
        if (href && !href.trimStart().toLowerCase().startsWith('javascript:')) {
          linkHrefs.push(href)
          inline.push('[')
        } else {
          linkHrefs.push(null)
        }
      } else {
        const href = linkHrefs.pop()
        if (href !== null && href !== undefined) inline.push(`](${href})`)
      }
      continue
    }
    if (name === 'strong' || name === 'b') {
      inline.push('**')
      continue
    }
    if (name === 'em' || name === 'i') {
      inline.push('*')
      continue
    }
    // Unrecognized tag → treat as a transparent wrapper (render children inline).
  }

  flushInline()

  const rawMd = out.join('')

  // Walk lines and prefix everything between BQ_OPEN/BQ_CLOSE markers with
  // `> `. Nesting increases the number of prefixes so `<blockquote><blockquote>`
  // yields `> > `.
  const prefixed: string[] = []
  let bqDepth = 0
  for (const line of rawMd.split('\n')) {
    if (line === BQ_OPEN) {
      bqDepth += 1
      continue
    }
    if (line === BQ_CLOSE) {
      if (bqDepth > 0) bqDepth -= 1
      continue
    }
    if (bqDepth > 0) {
      const prefix = '> '.repeat(bqDepth)
      // Blank lines inside a blockquote still get the marker so the block stays
      // visually connected in rendered markdown.
      prefixed.push(line === '' ? prefix.trimEnd() : `${prefix}${line}`)
    } else {
      prefixed.push(line)
    }
  }

  const collapsed: string[] = []
  let blank = 0
  for (const line of prefixed) {
    if (line.trim() === '') {
      blank += 1
      if (blank <= 1) collapsed.push('')
    } else {
      blank = 0
      collapsed.push(line.replace(/\s+$/, ''))
    }
  }
  const markdown = collapsed.join('\n').trim()
  const title = titleParts.join('').trim()
  return { title, markdown: markdown ? markdown + '\n' : '' }
}
