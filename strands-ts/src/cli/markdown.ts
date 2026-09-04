import { marked } from 'marked'
import type { Tokens } from 'marked'
import type { Theme } from './theme.js'

/**
 * Markdown-to-terminal renderer. Uses the marked lexer and walks the token
 * tree directly so rendering is independent of marked's renderer API, which
 * changes between major versions.
 */

export interface MarkdownOptions {
  theme: Theme
}

function renderInlineToken(token: Tokens.Generic, theme: Theme): string {
  switch (token.type) {
    case 'strong':
      return theme.bold(renderInline((token as Tokens.Strong).tokens ?? [], theme))
    case 'em':
      return renderInline((token as Tokens.Em).tokens ?? [], theme)
    case 'del':
      return renderInline((token as Tokens.Del).tokens ?? [], theme)
    case 'codespan':
      return theme.code((token as Tokens.Codespan).text)
    case 'link': {
      const link = token as Tokens.Link
      return theme.link(renderInline(link.tokens ?? [], theme) || link.href, link.href)
    }
    case 'image':
      return theme.dim(`[image: ${(token as Tokens.Image).text}]`)
    case 'br':
      return '\n'
    case 'escape':
    case 'text':
      return renderTextToken(token, theme)
    case 'html':
      return ''
    case 'paragraph':
      return renderInline((token as Tokens.Paragraph).tokens ?? [], theme)
    default:
      return 'raw' in token ? String((token as { raw: string }).raw) : ''
  }
}

function renderTextToken(token: Tokens.Generic, theme: Theme): string {
  const textToken = token as Tokens.Text
  if (textToken.tokens && textToken.tokens.length > 0) {
    return renderInline(textToken.tokens, theme)
  }
  return textToken.text
}

function renderInline(tokens: Tokens.Generic[], theme: Theme): string {
  return tokens.map((token) => renderInlineToken(token, theme)).join('')
}

function renderList(token: Tokens.List, theme: Theme, indent: string): string[] {
  const lines: string[] = []
  let index = Number(token.start ?? 1)
  for (const item of token.items) {
    const marker = token.ordered ? `${index}. ` : '- '
    if (token.ordered) {
      index += 1
    }
    const itemLines = renderBlocks(item.tokens, theme, '')
    if (itemLines.length === 0) {
      continue
    }
    lines.push(indent + theme.dim(marker) + itemLines[0])
    const continuation = indent + '  '
    for (const line of itemLines.slice(1)) {
      lines.push(line === '' ? '' : continuation + line)
    }
  }
  return lines
}

function renderTable(token: Tokens.Table, theme: Theme): string[] {
  const headers = token.header.map((cell) => renderInline(cell.tokens ?? [], theme))
  const rows = token.rows.map((row) => row.map((cell) => renderInline(cell.tokens ?? [], theme)))
  const widths = headers.map((header, column) =>
    Math.max(stripLength(header), ...rows.map((row) => stripLength(row[column] ?? '')))
  )
  const border = theme.dim('+' + widths.map((width) => '-'.repeat(width + 2)).join('+') + '+')
  const rowLine = (cells: string[]): string =>
    theme.dim('|') +
    cells
      .map((cell, column) => ` ${cell.padEnd((widths[column] ?? 0) + cell.length - stripLength(cell))} `)
      .join(theme.dim('|')) +
    theme.dim('|')

  const lines = [border, rowLine(headers), border]
  for (const row of rows) {
    lines.push(rowLine(row))
  }
  lines.push(border)
  return lines
}

// eslint-disable-next-line no-control-regex -- match ANSI escape sequences
const ansiEscape = /\u001b\[[0-9;]*m/g

function stripLength(text: string): number {
  return text.replace(ansiEscape, '').length
}

function renderCode(token: Tokens.Code, theme: Theme): string[] {
  const label = token.lang ? theme.dim(` ${token.lang} `) : ''
  const body = token.text.split('\n').map((line) => '  ' + theme.code(line))
  return [theme.dim('┌' + label), ...body, theme.dim('└')]
}

function renderBlocks(tokens: Tokens.Generic[], theme: Theme, indent: string): string[] {
  const lines: string[] = []
  for (const token of tokens) {
    switch (token.type) {
      case 'heading': {
        const text = renderInline((token as Tokens.Heading).tokens ?? [], theme)
        const underline = theme.dim('─'.repeat(Math.min(text.length, 40)))
        lines.push(theme.heading(text), underline)
        break
      }
      case 'paragraph':
        lines.push(indent + renderInline((token as Tokens.Paragraph).tokens ?? [], theme))
        break
      case 'code':
        lines.push(...renderCode(token as Tokens.Code, theme))
        break
      case 'list':
        lines.push(...renderList(token as Tokens.List, theme, indent))
        break
      case 'table':
        lines.push(...renderTable(token as Tokens.Table, theme))
        break
      case 'blockquote':
        for (const line of renderBlocks((token as Tokens.Blockquote).tokens ?? [], theme, '')) {
          lines.push(theme.dim('│ ') + line)
        }
        break
      case 'hr':
        lines.push(theme.dim('─'.repeat(40)))
        break
      case 'space':
        break
      case 'text': {
        const textToken = token as Tokens.Text
        const text =
          textToken.tokens && textToken.tokens.length > 0 ? renderInline(textToken.tokens, theme) : textToken.text
        if (text !== '') {
          lines.push(indent + text)
        }
        break
      }
      default:
        if ('raw' in token && token.type !== 'html') {
          lines.push(indent + String((token as { raw: string }).raw).trim())
        }
        break
    }
  }
  return lines
}

/** Renders a Markdown string as styled terminal text. */
export function renderMarkdown(markdown: string, options: MarkdownOptions): string {
  const tokens = marked.lexer(markdown)
  const rendered = renderBlocks(tokens, options.theme, '')
  return rendered.filter((line, index) => line.trim() !== '' || index < rendered.length - 1).join('\n')
}
