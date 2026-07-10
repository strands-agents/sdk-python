import { describe, it, expect } from 'vitest'
import {
  DOC_TYPE_USER,
  DOC_TYPE_API_PYTHON,
  DOC_TYPE_API_TYPESCRIPT,
  DOC_TYPE_BLOG,
  isApiDocId,
  isBlogId,
  getDocType,
  getDocTypeFilter,
  getDocTypeFromUrl,
} from '../src/util/search'

describe('search doc-type classification', () => {
  describe('isApiDocId', () => {
    it('is true for python and typescript API content ids', () => {
      expect(isApiDocId('docs/api/python/strands.agent.agent')).toBe(true)
      expect(isApiDocId('docs/api/typescript/Agent')).toBe(true)
    })

    it('is false for guides, blog, and unrelated ids', () => {
      expect(isApiDocId('docs/user-guide/quickstart/overview')).toBe(false)
      expect(isApiDocId('blog/my-post')).toBe(false)
      expect(isApiDocId('docs/community/plugins/x')).toBe(false)
    })
  })

  describe('isBlogId', () => {
    it('is true only for blog/ ids', () => {
      expect(isBlogId('blog/my-post')).toBe(true)
      expect(isBlogId('docs/user-guide/x')).toBe(false)
      expect(isBlogId('docs/api/python/x')).toBe(false)
    })
  })

  describe('getDocType', () => {
    it('splits API reference by language', () => {
      expect(getDocType('docs/api/python/strands.agent.agent')).toBe(DOC_TYPE_API_PYTHON)
      expect(getDocType('docs/api/typescript/Agent')).toBe(DOC_TYPE_API_TYPESCRIPT)
    })

    it('classifies blog posts', () => {
      expect(getDocType('blog/interleaved-thinking')).toBe(DOC_TYPE_BLOG)
    })

    it('defaults everything else to User Docs', () => {
      expect(getDocType('docs/user-guide/quickstart/overview')).toBe(DOC_TYPE_USER)
      expect(getDocType('docs/community/plugins/x')).toBe(DOC_TYPE_USER)
      expect(getDocType('docs/examples/y')).toBe(DOC_TYPE_USER)
    })
  })

  describe('getDocTypeFilter', () => {
    it('prefixes the doc type with the Type filter key', () => {
      expect(getDocTypeFilter('docs/api/python/strands.agent.agent')).toBe('Type:Python API')
      expect(getDocTypeFilter('docs/user-guide/x')).toBe('Type:User Docs')
      expect(getDocTypeFilter('blog/my-post')).toBe('Type:Blog')
    })
  })

  describe('getDocTypeFromUrl', () => {
    it('classifies base-relative URL paths', () => {
      expect(getDocTypeFromUrl('/docs/api/python/strands.agent.agent/')).toBe(DOC_TYPE_API_PYTHON)
      expect(getDocTypeFromUrl('/docs/api/typescript/Agent/')).toBe(DOC_TYPE_API_TYPESCRIPT)
      expect(getDocTypeFromUrl('/blog/my-post/')).toBe(DOC_TYPE_BLOG)
      expect(getDocTypeFromUrl('/docs/user-guide/quickstart/overview/')).toBe(DOC_TYPE_USER)
    })

    it('tolerates a deploy base path prefix (segments not at the start)', () => {
      // Unlike getDocType (id-based, matches at the start), URL classification
      // must find the segment anywhere in the path — e.g. a site served at /sub/.
      expect(getDocTypeFromUrl('/sub/docs/api/python/x/')).toBe(DOC_TYPE_API_PYTHON)
      expect(getDocTypeFromUrl('/sub/blog/my-post/')).toBe(DOC_TYPE_BLOG)
    })

    it('accepts absolute URLs', () => {
      expect(getDocTypeFromUrl('https://strandsagents.com/docs/api/typescript/Agent/')).toBe(
        DOC_TYPE_API_TYPESCRIPT
      )
    })

    it('returns null when no known type matches', () => {
      expect(getDocTypeFromUrl('/changelog/')).toBeNull()
      expect(getDocTypeFromUrl('/')).toBeNull()
      expect(getDocTypeFromUrl('https://example.com/')).toBeNull()
    })

    it('does not misclassify api/blog segments outside the docs tree', () => {
      // "/docs/api/python" must win over the generic "/docs/" fallback.
      expect(getDocTypeFromUrl('/docs/api/python/')).toBe(DOC_TYPE_API_PYTHON)
      // A guide page that merely mentions api in its slug is still User Docs.
      expect(getDocTypeFromUrl('/docs/user-guide/api-reference-intro/')).toBe(DOC_TYPE_USER)
    })
  })
})
