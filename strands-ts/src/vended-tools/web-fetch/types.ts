import type { z } from 'zod'
import type { webFetchInputSchema } from './web-fetch.js'

/**
 * Input parameters for the web fetch tool.
 *
 * Derived from the Zod schema in `web-fetch.ts` so the two cannot drift.
 */
export type WebFetchInput = z.infer<typeof webFetchInputSchema>

/**
 * Output from the web fetch tool.
 */
export interface WebFetchOutput {
  /** Final URL after any redirects. */
  url: string
  /** HTTP status code of the final response. */
  status: number
  /** Content-Type header of the final response, or an empty string. */
  contentType: string
  /** Extracted document title, or an empty string if none was found. */
  title: string
  /**
   * Extracted, cleaned markdown suitable for a model to read. For non-HTML
   * responses (JSON, plain text), the decoded body is returned verbatim.
   */
  markdown: string
}
