import { describe, it, expect } from 'vitest'
import { htmlToMarkdown } from '../extract.js'

describe('htmlToMarkdown', () => {
  it('strips script and style content', async () => {
    const html = `
      <html><head><title>Hi</title>
      <style>body{color:red}</style>
      </head><body>
      <p>Hello world.</p>
      <script>alert('xss')</script>
      <p>After script.</p>
      </body></html>
    `
    const result = await htmlToMarkdown(html)
    expect(result).toMatch(/# Hi/)
    expect(result).not.toMatch(/alert/)
    expect(result).not.toMatch(/color:red/)
    expect(result).toMatch(/Hello world\./)
    expect(result).toMatch(/After script\./)
  })

  it('strips data-URI image blobs', async () => {
    const bigBlob = 'A'.repeat(10000)
    const result = await htmlToMarkdown(`<p>text</p><img src="data:image/png;base64,${bigBlob}" alt="alt text">`)
    expect(result).not.toContain(bigBlob)
    expect(result).not.toMatch(/data:/)
    expect(result).toMatch(/alt text/)
  })

  it('preserves regular images and relative hrefs', async () => {
    expect(await htmlToMarkdown('<img src="https://example.com/pic.png" alt="pic">')).toContain(
      '![pic](https://example.com/pic.png)'
    )
    expect(await htmlToMarkdown('<a href="/about">About</a>')).toContain('[About](/about)')
  })

  it('drops javascript: hrefs but keeps link text', async () => {
    const result = await htmlToMarkdown('<a href="javascript:alert(1)">click</a>')
    expect(result).not.toMatch(/javascript:/)
    expect(result).toMatch(/click/)
  })

  it.each(['\u200b', '\u00ad', '\ufeff', ' '])('drops javascript: href with invisible prefix %s', async (prefix) => {
    const result = await htmlToMarkdown(`<a href="${prefix}javascript:alert(1)">click</a>`)
    expect(result).not.toMatch(/javascript:/)
    expect(result).toContain('click')
  })

  it('drops javascript: img srcs', async () => {
    expect(await htmlToMarkdown('<img src="javascript:alert(1)" alt="x">')).not.toMatch(/javascript:/)
  })

  it.each(['\u200b', '\u00ad', '\ufeff'])('drops javascript: img src with invisible prefix %s', async (prefix) => {
    expect(await htmlToMarkdown(`<img src="${prefix}javascript:alert(1)" alt="x">`)).not.toMatch(/javascript:/)
  })

  it('strips data-URI image blob with invisible prefix', async () => {
    const blob = 'A'.repeat(100)
    const result = await htmlToMarkdown(`<img src="\u200bdata:image/png;base64,${blob}" alt="alt">`)
    expect(result).not.toContain(blob)
    expect(result).toContain('alt')
  })

  it('preserves headings, lists, and links', async () => {
    const html = `
      <h1>Title</h1>
      <p>Intro paragraph with a <a href="https://ex.com/x">link</a>.</p>
      <ul><li>one</li><li>two</li></ul>
      <ol><li>first</li><li>second</li></ol>
    `
    const result = await htmlToMarkdown(html)
    expect(result).toMatch(/# Title/)
    expect(result).toContain('[link](https://ex.com/x)')
    expect(result).toMatch(/- +one/)
    expect(result).toMatch(/1\. +first/)
  })

  it('preserves code blocks', async () => {
    const result = await htmlToMarkdown('<pre><code>def f():\n    return 1</code></pre>')
    expect(result).toContain('```')
    expect(result).toContain('def f():')
  })

  it('survives malformed HTML', async () => {
    const result = await htmlToMarkdown('<p>ok <b>bold <em>and italic</p>')
    expect(result).toContain('ok')
    expect(result).toContain('bold')
  })

  it('preserves blockquote prefix on nested block content', async () => {
    const result = await htmlToMarkdown('<blockquote><p>quoted line one.</p><p>quoted line two.</p></blockquote>')
    const bqLines = result.split('\n').filter((line: string) => line.includes('quoted line'))
    expect(bqLines).toHaveLength(2)
    for (const line of bqLines) {
      expect(line.startsWith('> ')).toBe(true)
    }
  })

  it('void tag inside a dropped element does not swallow following content', async () => {
    expect(await htmlToMarkdown('<form><input></form><p>after</p>')).toContain('after')
  })

  it('embeds the page title as a top-level heading', async () => {
    const result = await htmlToMarkdown('<html><head><title>My Page</title></head><body><p>body</p></body></html>')
    expect(result).toMatch(/^# My Page\n/)
    expect(result).toContain('body')
  })

  it('produces no heading when there is no title', async () => {
    expect(await htmlToMarkdown('<p>no title here</p>')).not.toMatch(/^#/)
  })

  it('returns original input on conversion failure', async () => {
    expect(await htmlToMarkdown(null as unknown as string)).toBe(null)
  })
})
