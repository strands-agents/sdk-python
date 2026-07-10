import { describe, it, expect } from 'vitest'
import {
  documentFormatForMime,
  extensionForContentType,
  LEGACY_TEXT_DOCUMENT_TYPES,
  mimeForDocumentFormat,
} from '../media-types.js'
import { DOCUMENT_FORMATS, toMimeType } from '../../../mime.js'

describe('media-types', () => {
  describe('mimeForDocumentFormat', () => {
    it('maps text-based document formats to canonical text/* IANA types', () => {
      expect(mimeForDocumentFormat('csv')).toBe('text/csv')
      expect(mimeForDocumentFormat('html')).toBe('text/html')
      expect(mimeForDocumentFormat('md')).toBe('text/markdown')
      expect(mimeForDocumentFormat('txt')).toBe('text/plain')
    })

    it('maps binary document formats to their canonical application/* types', () => {
      expect(mimeForDocumentFormat('pdf')).toBe('application/pdf')
      expect(mimeForDocumentFormat('docx')).toBe(
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
      )
      expect(mimeForDocumentFormat('xlsx')).toBe('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    })

    it('falls back to application/octet-stream for formats outside the contract', () => {
      // Reachable only with untyped runtime data. octet-stream decodes as text on
      // retrieval rather than fabricating a document.format the model rejects.
      expect(mimeForDocumentFormat('bogus')).toBe('application/octet-stream')
    })
  })

  describe('documentFormatForMime', () => {
    it('reconstructs the format from canonical MIME types', () => {
      expect(documentFormatForMime('text/csv')).toBe('csv')
      expect(documentFormatForMime('application/pdf')).toBe('pdf')
      expect(documentFormatForMime('application/vnd.openxmlformats-officedocument.wordprocessingml.document')).toBe(
        'docx'
      )
    })

    it('excludes text/plain so txt retrieves as text rather than a document block', () => {
      expect(documentFormatForMime('text/plain')).toBeUndefined()
    })

    it('returns undefined for the octet-stream fallback and non-document types', () => {
      expect(documentFormatForMime('application/octet-stream')).toBeUndefined()
      expect(documentFormatForMime('image/png')).toBeUndefined()
    })

    it('reconstructs legacy application/{format} artifacts written by earlier releases', () => {
      expect(documentFormatForMime('application/csv')).toBe('csv')
      expect(documentFormatForMime('application/docx')).toBe('docx')
      expect(documentFormatForMime('application/txt')).toBe('txt')
    })
  })

  describe('format <-> MIME round-trip', () => {
    it('is the identity for every document format except txt', () => {
      // Verifies the mapping in isolation. On the plugin's retrieval path,
      // application/json is intercepted and parsed as JSON before reaching
      // documentFormatForMime, so a json document does not round-trip as a
      // document block there; txt maps to text/plain (shared with plain text
      // blocks) and intentionally retrieves as text.
      for (const format of DOCUMENT_FORMATS) {
        const mime = mimeForDocumentFormat(format)
        if (format === 'txt') {
          expect(documentFormatForMime(mime)).toBeUndefined()
        } else {
          expect(documentFormatForMime(mime)).toBe(format)
        }
      }
    })

    it('has a canonical MIME type for every DocumentFormat', () => {
      // Guards against a format being added to the union without a mapping, which
      // would silently degrade that document to octet-stream/text on retrieval.
      for (const format of DOCUMENT_FORMATS) {
        expect(toMimeType(format)).toBeDefined()
      }
    })
  })

  describe('LEGACY_TEXT_DOCUMENT_TYPES', () => {
    it('marks legacy text-based document artifacts as searchable', () => {
      expect(LEGACY_TEXT_DOCUMENT_TYPES.has('application/csv')).toBe(true)
      expect(LEGACY_TEXT_DOCUMENT_TYPES.has('application/txt')).toBe(true)
      expect(LEGACY_TEXT_DOCUMENT_TYPES.has('application/md')).toBe(true)
      expect(LEGACY_TEXT_DOCUMENT_TYPES.has('application/html')).toBe(true)
      expect(LEGACY_TEXT_DOCUMENT_TYPES.has('application/pdf')).toBe(false)
    })
  })

  describe('extensionForContentType', () => {
    it('returns shell-friendly extensions for canonical types', () => {
      expect(extensionForContentType('text/csv')).toBe('.csv')
      expect(extensionForContentType('text/markdown')).toBe('.md')
      expect(extensionForContentType('application/vnd.openxmlformats-officedocument.wordprocessingml.document')).toBe(
        '.docx'
      )
      expect(extensionForContentType('application/octet-stream')).toBe('.bin')
    })

    it('falls back to the MIME subtype for unmapped types', () => {
      expect(extensionForContentType('image/png')).toBe('.png')
      expect(extensionForContentType('application/foo')).toBe('.foo')
    })
  })
})
