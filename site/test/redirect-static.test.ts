import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { buildStaticRedirects } from '../src/util/redirect.static'
import { pathToDocsSlug } from '../src/util/links'

/**
 * Each test gets a fresh temp content dir and passes its own static-redirect
 * rules, so tests are independent of each other and of the production
 * STATIC_SLUG_REDIRECTS entries (adding a rename must not break this suite).
 * buildStaticRedirects runs at config time, before astro:content exists, so
 * the fixtures exercise the real disk-reading path.
 */
function writeDoc(contentDir: string, relativePath: string, frontmatter: string): void {
  const filePath = path.join(contentDir, relativePath)
  fs.mkdirSync(path.dirname(filePath), { recursive: true })
  fs.writeFileSync(filePath, `---\n${frontmatter}\n---\n\n# Page\n`)
}

describe('buildStaticRedirects', () => {
  let contentDir: string

  beforeEach(() => {
    contentDir = fs.mkdtempSync(path.join(os.tmpdir(), 'redirect-static-test-'))
  })

  afterEach(() => {
    fs.rmSync(contentDir, { recursive: true, force: true })
  })

  it('maps a redirectFrom frontmatter entry to its page URL', () => {
    writeDoc(contentDir, 'docs/user-guide/state.mdx', 'title: State\nredirectFrom:\n  - docs/old/state')

    const redirects = buildStaticRedirects(contentDir, '/', {})

    expect(redirects).toEqual({ '/docs/old/state': '/docs/user-guide/state/' })
  })

  it('respects a frontmatter slug override on the target page', () => {
    writeDoc(
      contentDir,
      'docs/user-guide/renamed.mdx',
      'title: Renamed\nslug: docs/user-guide/custom-slug\nredirectFrom:\n  - docs/old/renamed'
    )

    const redirects = buildStaticRedirects(contentDir, '/', {})

    expect(redirects).toEqual({ '/docs/old/renamed': '/docs/user-guide/custom-slug/' })
  })

  it('resolves a rename rule against slugified page ids', () => {
    // The file name slugifies (Custom_Tools.mdx → custom_tools), so a rule
    // targeting the collection id only validates if ids — not raw file
    // names — are what the builder checks against.
    writeDoc(contentDir, 'docs/user-guide/Custom_Tools.mdx', 'title: Custom Tools')

    const redirects = buildStaticRedirects(contentDir, '/', {
      'docs/user-guide/python-tools': 'docs/user-guide/custom_tools',
    })

    expect(redirects).toEqual({ '/docs/user-guide/python-tools': '/docs/user-guide/custom_tools/' })
  })

  it('passes external targets through unchanged', () => {
    const redirects = buildStaticRedirects(contentDir, '/', { discord: 'https://discord.gg/strands' })

    expect(redirects).toEqual({ '/discord': 'https://discord.gg/strands' })
  })

  it('prefixes internal destinations with the base path', () => {
    writeDoc(contentDir, 'docs/user-guide/state.mdx', 'title: State\nredirectFrom:\n  - docs/old/state')

    const redirects = buildStaticRedirects(contentDir, '/pr-preview/', {})

    expect(redirects).toEqual({ '/docs/old/state': '/pr-preview/docs/user-guide/state/' })
  })

  it('throws when a redirect source collides with an existing content file', () => {
    writeDoc(contentDir, 'docs/user-guide/state.mdx', 'title: State')
    writeDoc(contentDir, 'docs/user-guide/other.mdx', 'title: Other\nredirectFrom:\n  - docs/user-guide/state')

    expect(() => buildStaticRedirects(contentDir, '/', {})).toThrow(/collides with an existing content file/)
  })

  it('throws when duplicate redirectFrom slugs point at different targets', () => {
    writeDoc(contentDir, 'docs/user-guide/dupe-a.mdx', 'title: A\nredirectFrom:\n  - docs/old/dupe')
    writeDoc(contentDir, 'docs/user-guide/dupe-b.mdx', 'title: B\nredirectFrom:\n  - docs/old/dupe')

    expect(() => buildStaticRedirects(contentDir, '/', {})).toThrow(
      /duplicate redirectFrom slug "docs\/old\/dupe"/
    )
  })

  it('throws when an internal redirect target has no content file', () => {
    expect(() => buildStaticRedirects(contentDir, '/', { 'docs/old/page': 'docs/new/missing' })).toThrow(
      /has no content file/
    )
  })

  it('throws when a redirectFrom entry conflicts with a static rename rule', () => {
    writeDoc(contentDir, 'docs/user-guide/state.mdx', 'title: State\nredirectFrom:\n  - docs/old/page')
    writeDoc(contentDir, 'docs/user-guide/other.mdx', 'title: Other')

    expect(() =>
      buildStaticRedirects(contentDir, '/', { 'docs/old/page': 'docs/user-guide/other' })
    ).toThrow(/conflicts with a static rename rule/)
  })

  it('allows a redirectFrom entry that agrees with a static rename rule', () => {
    writeDoc(contentDir, 'docs/user-guide/state.mdx', 'title: State\nredirectFrom:\n  - docs/old/page')

    const redirects = buildStaticRedirects(contentDir, '/', { 'docs/old/page': 'docs/user-guide/state' })

    expect(redirects).toEqual({ '/docs/old/page': '/docs/user-guide/state/' })
  })

  it('validates the production redirect map against the real content dir', () => {
    // The same invocation astro.config.mjs makes — catches a production rename
    // rule whose target page was deleted or moved, without pinning any
    // specific entry.
    const redirects = buildStaticRedirects(path.resolve(__dirname, '../src/content'))

    expect(Object.keys(redirects).length).toBeGreaterThan(0)
  })
})

describe('pathToDocsSlug', () => {
  // redirect.static.ts derives redirect targets with pathToDocsSlug, and the
  // docs collection derives entry ids from it (generateDocsId) — these cases
  // pin the shared behavior both sides depend on.
  it.each([
    ['plain path', 'docs/user-guide/state.mdx', undefined, 'docs/user-guide/state'],
    ['index collapses to parent', 'docs/examples/index.mdx', undefined, 'docs/examples'],
    ['README collapses to parent', 'docs/examples/README.mdx', undefined, 'docs/examples'],
    ['segments are slugified', 'docs/user-guide/Deploy_To_AWS.mdx', undefined, 'docs/user-guide/deploy_to_aws'],
    ['root index becomes index', 'index.mdx', undefined, 'index'],
    ['frontmatter slug wins', 'docs/user-guide/state.mdx', 'custom/slug', 'custom/slug'],
  ])('%s', (_description, entryPath, slugOverride, expected) => {
    expect(pathToDocsSlug(entryPath, slugOverride)).toBe(expected)
  })
})
