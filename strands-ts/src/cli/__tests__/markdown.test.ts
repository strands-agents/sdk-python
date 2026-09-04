import { describe, it, expect } from 'vitest'
import { createTheme, stripAnsi } from '../theme.js'
import { renderMarkdown } from '../markdown.js'
import chalk from 'chalk'

// Chalk disables colors when stdout is not a TTY, which is the case under
// vitest. Force a color level so styled-output assertions are deterministic.
chalk.level = 1

const theme = createTheme({ plain: false })
const plainTheme = createTheme({ plain: true })

describe('createTheme', () => {
  it('returns a pass-through theme in plain mode', () => {
    expect(plainTheme.plain).toBe(true)
    expect(plainTheme.error('boom')).toBe('boom')
    expect(plainTheme.heading('Title')).toBe('Title')
  })

  it('returns a styled theme otherwise', () => {
    expect(theme.plain).toBe(false)
    expect(theme.error('boom')).not.toBe('boom')
    expect(stripAnsi(theme.error('boom'))).toBe('boom')
  })

  it('appends the URL to link text', () => {
    const styled = theme.link('docs', 'https://example.com')
    expect(stripAnsi(styled)).toBe('docs (https://example.com)')
    expect(plainTheme.link('https://example.com', 'https://example.com')).toBe('https://example.com')
  })

  it('strips ANSI escape sequences', () => {
    expect(stripAnsi('[31mred[0m')).toBe('red')
  })
})

describe('renderMarkdown', () => {
  it('renders headings with a rule in styled mode', () => {
    const output = renderMarkdown('# Title', { theme })
    const lines = output.split('\n')
    expect(stripAnsi(lines[0] ?? '')).toBe('Title')
    expect(stripAnsi(lines[1] ?? '')).toMatch(/^─+$/)
  })

  it('renders lists with markers and indentation', () => {
    const output = stripAnsi(renderMarkdown('- one\n- two\n  - nested', { theme }))
    expect(output).toContain('- one')
    expect(output).toContain('- two')
    expect(output).toContain('- nested')
    const nestedLine = output.split('\n').find((line) => line.includes('nested'))
    expect(nestedLine?.startsWith('  ')).toBe(true)
  })

  it('renders ordered lists with incrementing numbers', () => {
    const output = stripAnsi(renderMarkdown('1. first\n2. second', { theme }))
    expect(output).toContain('1. first')
    expect(output).toContain('2. second')
  })

  it('renders code blocks in a frame', () => {
    const output = stripAnsi(renderMarkdown('```ts\nconst x = 1\n```', { theme }))
    expect(output).toContain(' ts ')
    expect(output).toContain('  const x = 1')
    expect(output).toContain('┌')
    expect(output).toContain('└')
  })

  it('renders links as text plus URL', () => {
    const output = stripAnsi(renderMarkdown('[Strands](https://example.com)', { theme }))
    expect(output).toContain('Strands (https://example.com)')
  })

  it('renders tables with aligned columns and borders', () => {
    const output = stripAnsi(renderMarkdown('| a | b |\n| --- | --- |\n| 1 | 2 |', { theme }))
    expect(output).toContain('|')
    expect(output).toContain('-')
    expect(output).toContain(' a ')
    expect(output).toContain(' 1 ')
  })

  it('renders bold and inline code', () => {
    const output = stripAnsi(renderMarkdown('use **the** `tool`', { theme }))
    expect(output).toContain('use the tool')
    const styled = renderMarkdown('use **the** `tool`', { theme })
    expect(styled).not.toBe(stripAnsi(styled))
  })

  it('renders blockquotes with a vertical rule', () => {
    const output = stripAnsi(renderMarkdown('> quoted', { theme }))
    expect(output).toContain('│ quoted')
  })

  it('produces raw text in plain mode', () => {
    const markdown = '# Title\n\n- one\n- two\n\n```ts\nconst x = 1\n```'
    const output = renderMarkdown(markdown, { theme: plainTheme })
    expect(output).toBe(stripAnsi(output))
    expect(output).toContain('Title')
    expect(output).toContain('- one')
    expect(output).toContain('const x = 1')
  })
})
