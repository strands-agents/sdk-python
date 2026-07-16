import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import { Sandbox } from '../../sandbox/base.js'
import { makeFileEditor } from '../file-editor/file-editor.js'

/**
 * Description shown to the model for the file-read tool.
 */
export const DEFAULT_FILE_READ_DESCRIPTION =
  'Read-only filesystem tool. View a file (with an optional line range) or list a directory. Paths must be absolute. For creating or editing files, use `fileEditor`.'

/**
 * Zod schema for the read-only file tool.
 *
 * Intentionally narrower than `fileEditor`'s schema: only `path` and
 * `view_range` are exposed. There is no `command` enum, no `file_text`,
 * no `old_str`/`new_str`, no `insert_line` — a read-only agent cannot
 * ask this tool to write.
 *
 * This schema is the single source of truth for the tool's input shape;
 * `FileReadInput` below is derived from it via `z.infer` so the two cannot
 * drift.
 */
const fileReadInputSchema = z
  .object({
    path: z.string().describe('Absolute path to a file or directory.'),
    view_range: z
      .tuple([z.number(), z.number()])
      .optional()
      .describe(
        'Optional line range [start, end] to view. 1-indexed; end can be -1 for end-of-file. Not allowed for directories.'
      ),
  })
  .strict()

/**
 * Input parameters for the read-only file tool.
 *
 * Derived from `fileReadInputSchema` so any schema change flows through
 * automatically.
 */
export type FileReadInput = z.infer<typeof fileReadInputSchema>

export interface MakeFileReadOptions {
  name?: string
  description?: string
}

/**
 * Create a sandbox-routed, read-only file tool.
 *
 * A thin shim over {@link makeFileEditor}'s `view` command with a narrower
 * input schema. All validation (absolute path, `..` traversal, size limit,
 * `view_range` bounds, sandbox probing) is delegated to `fileEditor`; this
 * tool intentionally adds no new logic and no new checks.
 *
 * If a sandbox is passed, it is bound at creation time. Otherwise, the
 * underlying `fileEditor` reads the sandbox from `context.agent.sandbox` at
 * call time.
 *
 * @example
 * ```typescript
 * import { fileRead } from '@strands-agents/sdk/vended-tools/file-read'
 * import { Agent } from '@strands-agents/sdk'
 *
 * const agent = new Agent({
 *   model: new BedrockModel({ region: 'us-east-1' }),
 *   tools: [fileRead],
 * })
 *
 * await agent.invoke('Read /tmp/config.json')
 * ```
 */
export function makeFileRead(options?: MakeFileReadOptions): ReturnType<typeof tool>
export function makeFileRead(sandbox: Sandbox | undefined, options?: MakeFileReadOptions): ReturnType<typeof tool>
export function makeFileRead(
  sandboxOrOptions?: Sandbox | MakeFileReadOptions,
  maybeOptions?: MakeFileReadOptions
): ReturnType<typeof tool> {
  const boundSandbox = sandboxOrOptions instanceof Sandbox ? sandboxOrOptions : undefined
  const options = sandboxOrOptions instanceof Sandbox || maybeOptions ? (maybeOptions ?? {}) : (sandboxOrOptions ?? {})

  const editor = boundSandbox ? makeFileEditor(boundSandbox) : makeFileEditor()

  return tool({
    name: options.name ?? 'fileRead',
    description: options.description ?? DEFAULT_FILE_READ_DESCRIPTION,
    inputSchema: fileReadInputSchema,
    callback: async (input, context) => {
      if (!context) throw new Error('Tool context is required for fileRead operations')
      return editor.invoke({ command: 'view', path: input.path, view_range: input.view_range }, context)
    },
  })
}

/**
 * Default read-only file tool. Reads the sandbox from the agent's context at
 * call time.
 */
export const fileRead = makeFileRead()
