/**
 * Search (Pagefind) doc-type helpers.
 *
 * The site search is powered by Starlight's bundled Pagefind index. Pagefind
 * tags a page with a filter facet via the `data-pagefind-filter` data attribute
 * on the indexed page body. Every indexed page is tagged with a `Type` facet,
 * which the search UI surfaces as a doc-type toggle. The Python and TypeScript
 * API references are separate facet values so users can target one language.
 */

/** Filter key surfaced in the search UI (rendered as the facet's label). */
export const DOC_TYPE_FILTER_KEY = 'Type'

/**
 * Doc-type facet values. This is the single source of truth for the vocabulary:
 * the same strings label the search filter toggle AND the per-result badge, so
 * the two never drift.
 */
export const DOC_TYPE_USER = 'User Docs'
export const DOC_TYPE_API_PYTHON = 'Python API'
export const DOC_TYPE_API_TYPESCRIPT = 'TypeScript API'
export const DOC_TYPE_BLOG = 'Blog'

export type DocType =
  | typeof DOC_TYPE_USER
  | typeof DOC_TYPE_API_PYTHON
  | typeof DOC_TYPE_API_TYPESCRIPT
  | typeof DOC_TYPE_BLOG

/**
 * Whether a content id points at an auto-generated API reference page (either
 * language). Mirrors the id checks in dynamic-sidebar.ts and route-middleware.ts.
 */
export function isApiDocId(id: string): boolean {
  return id.startsWith('docs/api/python') || id.startsWith('docs/api/typescript')
}

/** Whether a content id points at a blog post. */
export function isBlogId(id: string): boolean {
  return id.startsWith('blog/')
}

/**
 * Classify a page into its search doc type. Blog posts render through
 * StarlightPage, whose id is derived from the URL (e.g. `blog/my-post`);
 * docs use their content-collection id (e.g. `docs/user-guide/...`). The two
 * API references are split by language.
 */
export function getDocType(id: string): DocType {
  if (id.startsWith('docs/api/python')) return DOC_TYPE_API_PYTHON
  if (id.startsWith('docs/api/typescript')) return DOC_TYPE_API_TYPESCRIPT
  if (isBlogId(id)) return DOC_TYPE_BLOG
  return DOC_TYPE_USER
}

/** The `data-pagefind-filter` attribute value for a page's Type facet. */
export function getDocTypeFilter(id: string): string {
  return `${DOC_TYPE_FILTER_KEY}:${getDocType(id)}`
}

/**
 * Classify a result by its rendered URL path (e.g. `/docs/api/python/...`),
 * for badging search results client-side. Unlike getDocType (which keys off the
 * content-collection id), this must tolerate the deploy base path, so it looks
 * for the `docs/api/...` / `blog/` segments anywhere in the path rather than at
 * the start. Returns null when the path matches no known type. The returned
 * DocType value is shown verbatim as the result's badge label.
 */
export function getDocTypeFromUrl(url: string): DocType | null {
  // Strip origin + normalize to a leading-slashed pathname.
  let path = url
  try {
    path = new URL(url, 'https://x').pathname
  } catch {
    /* already a path */
  }
  if (/\/docs\/api\/python(\/|$)/.test(path)) return DOC_TYPE_API_PYTHON
  if (/\/docs\/api\/typescript(\/|$)/.test(path)) return DOC_TYPE_API_TYPESCRIPT
  if (/\/blog\//.test(path)) return DOC_TYPE_BLOG
  if (/\/docs\//.test(path)) return DOC_TYPE_USER
  return null
}
