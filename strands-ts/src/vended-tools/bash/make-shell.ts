/**
 * Sandbox-bound shell tool factory.
 *
 * Separated from bash.ts to avoid pulling Node dependencies (child_process, Buffer)
 * into sandbox implementations that import this.
 *
 * The command runs in whichever shell the sandbox provides -- `sh` for the Docker
 * and local environments, the remote login shell over SSH -- so it must not rely
 * on shell-specific syntax.
 */

import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import { SandboxTimeoutError } from '../../sandbox/errors.js'
import { Sandbox } from '../../sandbox/base.js'
import type { BashOutput } from './types.js'
import { ShellTimeoutError, ShellExecutionError, SANDBOX_SHELL_DESCRIPTION } from './types.js'

const sandboxShellInputSchema = z.object({
  command: z.string().describe('The shell command to execute.'),
  timeout: z.number().positive().optional().describe('Timeout in seconds (default: 120).'),
})

export interface MakeShellOptions {
  name?: string
  description?: string
  inputSchema?: z.ZodType
}

/**
 * Create a sandbox shell tool. If a sandbox is passed, it's bound at creation time.
 * Otherwise, the tool reads from `context.agent.sandbox` at call time.
 * Used by sandbox implementations in `getTools()` and by users who want a customized shell tool.
 */
export function makeShell(options?: MakeShellOptions): ReturnType<typeof tool>
export function makeShell(sandbox: Sandbox | undefined, options?: MakeShellOptions): ReturnType<typeof tool>
export function makeShell(
  sandboxOrOptions?: Sandbox | MakeShellOptions,
  maybeOptions?: MakeShellOptions
): ReturnType<typeof tool> {
  const boundSandbox = sandboxOrOptions instanceof Sandbox ? sandboxOrOptions : undefined
  const options = sandboxOrOptions instanceof Sandbox || maybeOptions ? (maybeOptions ?? {}) : (sandboxOrOptions ?? {})

  return tool({
    name: options.name ?? 'shell',
    description: options.description ?? SANDBOX_SHELL_DESCRIPTION,
    inputSchema: (options.inputSchema ?? sandboxShellInputSchema) as typeof sandboxShellInputSchema,
    callback: async (input, context) => {
      if (!context) {
        throw new Error('Tool context is required for shell operations')
      }

      const sandbox = boundSandbox ?? context.agent.sandbox
      try {
        const result = await sandbox.execute(input.command, { timeout: input.timeout ?? 120 })
        return { output: result.stdout, error: result.stderr } as BashOutput
      } catch (err) {
        // Shell* extends Bash* so pre-rename catch clauses keep matching.
        if (err instanceof SandboxTimeoutError) throw new ShellTimeoutError(err.message)
        throw new ShellExecutionError((err as Error).message)
      }
    },
  })
}

/**
 * @deprecated makeBash is deprecated and will be removed in v2.0.0. Use makeShell instead. The tool
 * routes commands through the sandbox, which runs sh or the remote login shell rather than bash
 * specifically.
 */
export const makeBash = makeShell

/**
 * @deprecated MakeBashOptions is deprecated and will be removed in v2.0.0. Use MakeShellOptions instead.
 */
export type MakeBashOptions = MakeShellOptions
