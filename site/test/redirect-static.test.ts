import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { buildStaticRedirects } from '../src/util/redirect.static'
import { pathToDocsSlug } from '../src/util/links'

/**
 * Fixtures live in a temp content dir so tests exercise the real
 * disk-reading path (buildStaticRedirects runs at config time, before
 * astro:content exists).
 */
function writeDoc(contentDir: string, relativePath: string, frontmatter: string): void {
  const filePath = path.join(contentDir, relativePath)
  fs.mkdirSync(path.dirname(filePath), { recursive: true })
  fs.writeFileSync(filePath, `---\n${frontmatter}\n---\n\n# Page\n`)
}

describe('buildStaticRedirects', () => {
  let contentDir: string

  beforeAll(() => {
    contentDir = fs.mkdtempSync(path.join(os.tmpdir(), 'redirect-static-test-'))
    // Targets for the STATIC_SLUG_REDIRECTS entries must exist, or the builder
    // throws — create the current production set.
    writeDoc(contentDir, 'docs/user-guide/concepts/model-providers/google.mdx', 'title: Google')
    writeDoc(contentDir, 'docs/user-guide/concepts/tools/custom-tools.mdx', 'title: Custom Tools')
    writeDoc(
      contentDir,
      'docs/examples/python/multi_agent_example/multi_agent_example.mdx',
      'title: Multi-Agent Example'
    )
    writeDoc(contentDir, 'docs/examples/README.mdx', 'title: Examples')
  })

  afterAll(() => {
    fs.rmSync(contentDir, { recursive: true, force: true })
  })

  it('maps a redirectFrom frontmatter entry to its page URL', () => {
    writeDoc(contentDir, 'docs/user-guide/state.mdx', 'title: State\nredirectFrom:\n  - docs/old/state')

    const redirects = buildStaticRedirects(contentDir)

    expect(redirects['/docs/old/state']).toBe('/docs/user-guide/state/')
  })

  it('respects a frontmatter slug override on the target page', () => {
    writeDoc(
      contentDir,
      'docs/user-guide/renamed.mdx',
      'title: Renamed\nslug: docs/user-guide/custom-slug\nredirectFrom:\n  - docs/old/renamed'
    )

    const redirects = buildStaticRedirects(contentDir)

    expect(redirects['/docs/old/renamed']).toBe('/docs/user-guide/custom-slug/')
  })

  it('passes external STATIC_SLUG_REDIRECTS targets through unchanged', () => {
    const redirects = buildStaticRedirects(contentDir)

    expect(redirects['/discord']).toBe('https://discord.gg/strands')
  })

  it('prefixes internal destinations with the base path', () => {
    const redirects = buildStaticRedirects(contentDir, '/pr-preview/')

    expect(redirects['/docs/user-guide/concepts/model-providers/gemini']).toBe(
      '/pr-preview/docs/user-guide/concepts/model-providers/google/'
    )
  })

  it('throws when a redirect source collides with an existing content file', () => {
    writeDoc(
      contentDir,
      'docs/user-guide/collision.mdx',
      'title: Collision\nredirectFrom:\n  - docs/user-guide/state'
    )

    expect(() => buildStaticRedirects(contentDir)).toThrow(/collides with an existing content file/)

    fs.rmSync(path.join(contentDir, 'docs/user-guide/collision.mdx'))
  })

  it('throws when duplicate redirectFrom slugs point at different targets', () => {
    writeDoc(contentDir, 'docs/user-guide/dupe-a.mdx', 'title: A\nredirectFrom:\n  - docs/old/dupe')
    writeDoc(contentDir, 'docs/user-guide/dupe-b.mdx', 'title: B\nredirectFrom:\n  - docs/old/dupe')

    expect(() => buildStaticRedirects(contentDir)).toThrow(/duplicate redirectFrom slug "docs\/old\/dupe"/)

    fs.rmSync(path.join(contentDir, 'docs/user-guide/dupe-a.mdx'))
    fs.rmSync(path.join(contentDir, 'docs/user-guide/dupe-b.mdx'))
  })

  it('throws when an internal redirect target has no content file', () => {
    // The production STATIC_SLUG_REDIRECTS targets are fixtures here; removing
    // one makes its redirect target dangle.
    const target = path.join(contentDir, 'docs/user-guide/concepts/tools/custom-tools.mdx')
    fs.rmSync(target)

    expect(() => buildStaticRedirects(contentDir)).toThrow(/has no content file/)

    writeDoc(contentDir, 'docs/user-guide/concepts/tools/custom-tools.mdx', 'title: Custom Tools')
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
    ['segments are slugified', 'docs/user-guide/deploy_to_aws_lambda.mdx', undefined, 'docs/user-guide/deploy_to_aws_lambda'],
    ['root index becomes index', 'index.mdx', undefined, 'index'],
    ['frontmatter slug wins', 'docs/user-guide/state.mdx', 'custom/slug', 'custom/slug'],
  ])('%s', (_description, entryPath, slugOverride, expected) => {
    expect(pathToDocsSlug(entryPath, slugOverride)).toBe(expected)
  })
})
