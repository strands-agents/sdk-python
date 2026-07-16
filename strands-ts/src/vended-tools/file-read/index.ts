/**
 * Read-only file tool.
 *
 * A thin shim over `fileEditor`'s `view` command with a narrower two-parameter
 * surface (`path` and `view_range`). All validation is delegated to
 * `fileEditor`.
 */

export { fileRead, makeFileRead, DEFAULT_FILE_READ_DESCRIPTION } from './file-read.js'
export type { MakeFileReadOptions, FileReadInput } from './file-read.js'
