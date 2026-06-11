/**
 * Sandbox-bound bash tool factory.
 *
 * Separated from bash.ts to avoid pulling Node dependencies (child_process, Buffer)
 * into sandbox implementations that import this.
 */

import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import { SandboxTimeoutError } from '../../sandbox/errors.js'
import { Sandbox } from '../../sandbox/base.js'
import type { BashOutput } from './types.js'
import { BashTimeoutError, BashSessionError, SANDBOX_BASH_DESCRIPTION } from './types.js'

const sandboxBashInputSchema = z.object({
  command: z.string().describe('The bash command to execute.'),
  timeout: z.number().positive().optional().describe('Timeout in seconds (default: 120).'),
})

export interface MakeBashOptions {
  name?: string
  description?: string
  inputSchema?: z.ZodType
}

/**
 * Create a sandbox bash tool. If a sandbox is passed, it's bound at creation time.
 * Otherwise, the tool reads from `context.agent.sandbox` at call time.
 * Used by sandbox implementations in `getTools()` and by users who want a customized bash tool.
 */
export function makeBash(options?: MakeBashOptions): ReturnType<typeof tool>
export function makeBash(sandbox: Sandbox | undefined, options?: MakeBashOptions): ReturnType<typeof tool>
export function makeBash(
  sandboxOrOptions?: Sandbox | MakeBashOptions,
  maybeOptions?: MakeBashOptions
): ReturnType<typeof tool> {
  const boundSandbox = sandboxOrOptions instanceof Sandbox ? sandboxOrOptions : undefined
  const options = sandboxOrOptions instanceof Sandbox || maybeOptions ? (maybeOptions ?? {}) : (sandboxOrOptions ?? {})

  return tool({
    name: options.name ?? 'bash',
    description: options.description ?? SANDBOX_BASH_DESCRIPTION,
    inputSchema: (options.inputSchema ?? sandboxBashInputSchema) as typeof sandboxBashInputSchema,
    callback: async (input, context) => {
      if (!context) {
        throw new Error('Tool context is required for bash operations')
      }

      const sandbox = boundSandbox ?? context.agent.sandbox
      try {
        const result = await sandbox.execute(input.command, { timeout: input.timeout ?? 120 })
        return { output: result.stdout, error: result.stderr } as BashOutput
      } catch (err) {
        if (err instanceof SandboxTimeoutError) throw new BashTimeoutError(err.message)
        throw new BashSessionError((err as Error).message)
      }
    },
  })
}
