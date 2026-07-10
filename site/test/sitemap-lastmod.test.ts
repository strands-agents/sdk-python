import { describe, it, expect } from 'vitest'
import { urlToContentPaths } from '../src/plugins/sitemap-lastmod'

// The sitemap lastmod lookup matches URLs against keys produced by
// `git log --name-only`, which are repo-root-relative with forward slashes
// (e.g. `site/src/content/...` when the build runs from `site/`). These cases
// lock in that path space so a future change can't silently reintroduce the
// prefix/separator mismatch this plugin exists to avoid.
describe('urlToContentPaths', () => {
  it('builds repo-root-relative candidates with the git prefix', () => {
    expect(urlToContentPaths('/docs/user-guide/quickstart/python/', 'src/content', 'site/')).toEqual([
      'site/src/content/docs/user-guide/quickstart/python.mdx',
      'site/src/content/docs/user-guide/quickstart/python.md',
      'site/src/content/docs/user-guide/quickstart/python/index.mdx',
      'site/src/content/docs/user-guide/quickstart/python/index.md',
    ])
  })

  it('omits the prefix when built from the repo root (empty gitPrefix)', () => {
    expect(urlToContentPaths('/changelog/harness/python-v1.43.0/', 'src/content', '')).toEqual([
      'src/content/changelog/harness/python-v1.43.0.mdx',
      'src/content/changelog/harness/python-v1.43.0.md',
      'src/content/changelog/harness/python-v1.43.0/index.mdx',
      'src/content/changelog/harness/python-v1.43.0/index.md',
    ])
  })

  it('always emits forward slashes regardless of platform', () => {
    // path.posix.join keeps `/` so candidates match git output on Windows too.
    for (const candidate of urlToContentPaths('/docs/a/b/', 'src/content', 'site/')) {
      expect(candidate).not.toContain('\\')
    }
  })
})
