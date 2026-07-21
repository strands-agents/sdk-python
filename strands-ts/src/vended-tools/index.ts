/**
 * Barrel export for all vended tools.
 *
 * Provides a single import path for consumers who want all built-in tools:
 * ```typescript
 * import { bash, fileEditor, httpRequest, notebook } from '@strands-agents/sdk/vended-tools'
 * ```
 *
 * Note: This module requires a Node.js environment because the `bash` tool
 * imports `child_process`. For browser-compatible usage, import individual
 * tools via their subpath exports (e.g., `@strands-agents/sdk/vended-tools/notebook`).
 *
 * The `a2a-client` tool is deliberately *not* re-exported here because it
 * transitively imports `@a2a-js/sdk`, which is an optional peer dependency.
 * Consumers must import it via the subpath:
 * `@strands-agents/sdk/vended-tools/a2a-client`.
 */

export * from './bash/index.js'
export * from './file-editor/index.js'
export * from './http-request/index.js'
export * from './notebook/index.js'
