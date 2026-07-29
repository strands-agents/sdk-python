import { getCollection } from 'astro:content'

// Built once per build, not once per page: Head.astro runs for every route
// (~850 pages) and only needs membership checks against the docs ids.
let cached: Promise<Set<string>> | undefined

/** The set of docs content-collection ids, cached across pages within a build. */
export function getDocsIds(): Promise<Set<string>> {
  cached ??= getCollection('docs').then((docs) => new Set(docs.map((doc) => doc.id)))
  return cached
}
