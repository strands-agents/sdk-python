import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import { Buffer } from 'buffer'
import type { CodeInterpreterOutput } from './types.js'
import type { OutputFile } from '../../sandbox/types.js'

const codeInterpreterInputSchema = z.object({
  code: z.string().min(1).describe('The source code to execute.'),
  language: z.string().min(1).describe('Name of the interpreter binary (e.g., `python3`, `node`, `ruby`).'),
  timeout: z.number().positive().optional().describe('Maximum execution time in seconds. Defaults to 120 seconds.'),
})

/**
 * Code interpreter tool for executing source code in the agent's sandbox.
 *
 * The code is piped to the named interpreter binary's stdin. When the agent
 * has a sandbox configured, code runs inside that environment (Docker, SSH,
 * etc.); otherwise it runs on the host with no isolation.
 *
 * @example
 * ```typescript
 * import { codeInterpreter } from '@strands-agents/sdk/vended-tools/code-interpreter'
 * import { Agent } from '@strands-agents/sdk'
 *
 * const agent = new Agent({
 *   model: new BedrockModel({ region: 'us-east-1' }),
 *   tools: [codeInterpreter],
 * })
 *
 * await agent.invoke('Compute the sum of squares from 1 to 100 in Python')
 * ```
 */
export const codeInterpreter = tool({
  name: 'codeInterpreter',
  description:
    'Executes code with the specified interpreter. Each call runs in a fresh interpreter process. ' +
    'Variables and imports do not persist across calls.',
  inputSchema: codeInterpreterInputSchema,
  callback: async (input, context) => {
    if (!context) throw new Error('Tool context is required for codeInterpreter operations')
    const sandbox = context.agent.sandbox
    const result = await sandbox.executeCode(input.code, input.language, {
      timeout: input.timeout ?? 120,
    })
    return {
      stdout: result.stdout,
      stderr: result.stderr,
      exitCode: result.exitCode,
      outputFiles: result.outputFiles.map((f: OutputFile) => ({
        name: f.name,
        content: Buffer.from(f.content).toString('base64'),
        mimeType: f.mimeType,
      })),
    } satisfies CodeInterpreterOutput
  },
})
