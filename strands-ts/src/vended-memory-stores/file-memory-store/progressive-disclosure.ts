/**
 * Progressive disclosure for {@link FileMemoryStore}: the listing injector and the read tool.
 *
 * Injecting every file's path and description — rather than their content — lets the model judge what
 * is relevant and pull only that, splitting retrieval into a cheap recurring part and an on-demand one.
 */

import type { JSONValue } from '../../types/json.js'
import type { Plugin } from '../../plugins/plugin.js'
import type { Tool } from '../../tools/tool.js'
import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import { ContextInjector } from '../../vended-plugins/context-injector/plugin.js'
import { escapeXmlAttr, escapeXmlText } from '../../injection/xml.js'

/**
 * Builds the plugin that injects the file listing. Skips an empty store, so a fresh one costs no tokens.
 * Injects every turn, not just user turns: an autonomous turn that just read one file still needs the
 * listing to find the next. Fields are flattened and XML-escaped, since stored content is a
 * prompt-injection surface.
 *
 * The store caps how many files it reads per turn, so a large listing is truncated to a stable prefix
 * and the instruction reports how many were omitted.
 *
 * @param storeName - The store's name, which makes the plugin name unique per store
 * @param getListing - Supplies the listing and total file count, once per injected turn; `files.length`
 *   is below `total` when the store capped it
 * @returns A {@link ContextInjector} that injects the listing
 *
 * @internal
 */
export function createProgressiveDisclosureInjector(
  storeName: string,
  getListing: () => Promise<{ files: { path: string; description: string }[]; total: number }>
): Plugin {
  return new ContextInjector({
    name: `strands:file-memory-progressive-disclosure:${storeName}`,
    trigger: 'everyTurn',
    renderContent: async (): Promise<string | undefined> => {
      const { files, total } = await getListing()
      if (files.length === 0) return undefined

      const truncated = total > files.length
      const truncationNote = truncated
        ? ` Only the first ${files.length} of ${total} memory files are shown; the rest are not in this listing.`
        : ''
      const instruction = `You have these memory files from previous conversations. Read any whose description looks relevant to the current request with ${readToolName(storeName)} before answering — the descriptions below are summaries, not the content.${truncationNote}`
      const lines = files.map(
        (file) => `<file path="${escapeXmlAttr(flatten(file.path))}">${escapeXmlText(flatten(file.description))}</file>`
      )
      return `<memory-files>\n${instruction}\n\n${lines.join('\n')}\n</memory-files>`
    },
  })
}

/**
 * Builds the tool that reads one file by path. Named after the store (`agent-memory` yields
 * `read_agent_memory_file`), since tool names must be unique agent-wide. The store supplies `readFile`,
 * so this holds only the tool's shape and stays free of the store's path and frontmatter rules; a
 * `readFile` error reaches the model as the tool error, so it can correct a bad path.
 *
 * @param storeName - The store's name, which names the tool
 * @param readFile - Reads one file's body by path, applying the store's own path rules
 * @returns The read tool, ready to register
 *
 * @internal
 */
export function createReadTool(storeName: string, readFile: (path: string) => Promise<string>): Tool {
  return tool({
    name: readToolName(storeName),
    description:
      'Read one memory file in full, by its exact path. Use when the memory file listing shows a path whose description looks relevant to the current task — the listing gives you only a one-line summary, this gives you the content.',
    inputSchema: z.object({
      path: z
        .string()
        .describe('Exact path of the file to read, as shown in the memory file listing (e.g. "facts/testing.md").'),
    }),
    callback: async (input) => ({ content: await readFile(input.path) }) as JSONValue,
  })
}

/** Collapses whitespace runs so one file always renders as exactly one line. */
function flatten(value: string): string {
  return value.replace(/\s+/g, ' ').trim()
}

/** Derives the read tool's name from the store's name. */
function readToolName(storeName: string): string {
  return `read_${storeName.replace(/[^a-zA-Z0-9]+/g, '_').toLowerCase()}_file`
}
