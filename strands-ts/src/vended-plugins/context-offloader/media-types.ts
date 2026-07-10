/**
 * Canonical media-type mapping for offloaded document content blocks.
 *
 * At offload time the plugin knows each document block's `format`, which is
 * projected into a single canonical MIME string. The storage backends persist
 * that string (metadata sidecar / S3 `ContentType`), and it drives three
 * consumers: the on-disk file extension (shell-friendly artifacts), the
 * searchability gate, and reconstruction of the original block on full
 * retrieval.
 *
 * The canonical `format <-> MIME` direction is delegated to the SDK's own
 * {@link toMimeType} / {@link toMediaFormat} contract in `mime.ts`, so no table
 * is duplicated here. This module adds only the offloader-specific pieces:
 * backward-compatible reconstruction of the fabricated `application/{format}`
 * artifacts written by earlier releases, the text-searchable subset of those
 * legacy types, and file extensions.
 *
 * Documents round-trip to the original block for every format except `txt`,
 * which intentionally retrieves as plain text (content-identical, and
 * additionally pattern-searchable).
 *
 * @internal
 */

import { DOCUMENT_FORMATS, toMediaFormat, toMimeType, type DocumentFormat, type MediaFormat } from '../../mime.js'

/**
 * Fabricated `application/{format}` MIME types written by earlier releases,
 * mapped back to their document format so already-stored artifacts still
 * reconstruct into document blocks.
 */
const LEGACY_DOCUMENT_MIME_TO_FORMAT: Record<string, DocumentFormat> = Object.fromEntries(
  DOCUMENT_FORMATS.map((format): [string, DocumentFormat] => [`application/${format}`, format])
)

/**
 * Legacy `application/{format}` types whose bytes are text and are therefore
 * pattern-searchable (`csv`, `txt`, `md`, `html`).
 *
 * @internal
 */
export const LEGACY_TEXT_DOCUMENT_TYPES: ReadonlySet<string> = new Set(
  (['csv', 'txt', 'md', 'html'] as const).map((format) => `application/${format}`)
)

/**
 * File extensions for stored content types. Keeps offloaded artifacts
 * shell-friendly so agents can inspect them with `grep`/`cat` instead of
 * re-injecting content into context.
 */
const MIME_TO_EXTENSION: Record<string, string> = {
  'text/plain': '.txt',
  'text/csv': '.csv',
  'text/html': '.html',
  'text/markdown': '.md',
  'application/pdf': '.pdf',
  'application/msword': '.doc',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
  'application/vnd.ms-excel': '.xls',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
  'application/json': '.json',
  'application/xml': '.xml',
  'application/octet-stream': '.bin',
}

function isDocumentFormat(format: string): format is DocumentFormat {
  return (DOCUMENT_FORMATS as readonly string[]).includes(format)
}

/**
 * Return the canonical MIME type to store a document block under.
 *
 * Formats outside {@link DocumentFormat} (reachable only with untyped runtime
 * data) fall back to `application/octet-stream`, which retrieval decodes as
 * text rather than fabricating a document block the model would reject.
 *
 * @param format - The document block's `format` value.
 * @returns The canonical MIME type for the format.
 * @internal
 */
export function mimeForDocumentFormat(format: string): string {
  if (!isDocumentFormat(format)) return 'application/octet-stream'
  return toMimeType(format) ?? 'application/octet-stream'
}

/**
 * Return the original document format for a stored MIME type, if any.
 *
 * Resolves both the canonical `mime.ts` types and the legacy
 * `application/{format}` aliases. `text/plain` is intentionally excluded: it is
 * shared with plain text blocks and retrieves as text.
 *
 * @param contentType - The stored MIME type.
 * @returns The document format to reconstruct, or `undefined` if the type does
 *   not correspond to a document block.
 * @internal
 */
export function documentFormatForMime(contentType: string): DocumentFormat | undefined {
  if (contentType === 'text/plain') return undefined
  const canonical: MediaFormat | undefined = toMediaFormat(contentType)
  if (canonical !== undefined && isDocumentFormat(canonical)) return canonical
  return LEGACY_DOCUMENT_MIME_TO_FORMAT[contentType]
}

/**
 * Return a file extension (including the leading dot) for a stored content type.
 *
 * @param contentType - The stored MIME type.
 * @returns A file extension including the leading dot.
 * @internal
 */
export function extensionForContentType(contentType: string): string {
  return MIME_TO_EXTENSION[contentType] ?? `.${contentType.split('/').pop()}`
}
