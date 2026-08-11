import { describe, it, expect } from 'vitest'
import { makeBash, makeShell, ShellTimeoutError, ShellExecutionError } from '../index.js'
import { BashTimeoutError, BashSessionError, type BashOutput } from '../../bash/types.js'
import type { ToolContext } from '../../../index.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import { TestSandbox } from '../../../__fixtures__/test-sandbox.node.js'
import { mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'

describe.skipIf(process.platform === 'win32')('makeShell', () => {
  const createSandboxShell = (): { sandboxShell: ReturnType<typeof makeShell>; context: ToolContext } => {
    const workDir = mkdtempSync(join(tmpdir(), 'shell-sandbox-test-'))
    const sandbox = new TestSandbox(workDir)
    const sandboxShell = makeShell(sandbox)
    const agent = createMockAgent()
    const context: ToolContext = {
      toolUse: { name: 'shell', toolUseId: 'test-id', input: {} },
      agent,
      invocationState: {},
      interrupt: () => {
        throw new Error('interrupt not available in mock context')
      },
    }
    return { sandboxShell, context }
  }

  it('executes command via sandbox', async () => {
    const { sandboxShell, context } = createSandboxShell()
    const result = await sandboxShell.invoke({ command: 'echo "hello sandbox"' }, context)

    expect((result as BashOutput).output).toContain('hello sandbox')
    expect((result as BashOutput).error).toBe('')
  })

  it('captures stderr via sandbox', async () => {
    const { sandboxShell, context } = createSandboxShell()
    const result = await sandboxShell.invoke({ command: 'echo "oops" >&2' }, context)

    expect((result as BashOutput).error).toContain('oops')
  })

  it('does not persist state between calls (stateless)', async () => {
    const { sandboxShell, context } = createSandboxShell()
    await sandboxShell.invoke({ command: 'export MY_VAR=hello' }, context)
    const result = await sandboxShell.invoke({ command: 'echo "${MY_VAR:-empty}"' }, context)

    expect((result as BashOutput).output.trim()).toBe('empty')
  })

  it('respects timeout', async () => {
    const { sandboxShell, context } = createSandboxShell()
    await expect(sandboxShell.invoke({ command: 'sleep 10', timeout: 0.1 }, context)).rejects.toThrow()
  })

  it('throws ShellTimeoutError on timeout', async () => {
    const { sandboxShell, context } = createSandboxShell()
    await expect(sandboxShell.invoke({ command: 'sleep 10', timeout: 0.1 }, context)).rejects.toThrow(ShellTimeoutError)
  })

  it('timeout error still matches the pre-rename BashTimeoutError', async () => {
    const { sandboxShell, context } = createSandboxShell()
    await expect(sandboxShell.invoke({ command: 'sleep 10', timeout: 0.1 }, context)).rejects.toThrow(BashTimeoutError)
  })

  it('ShellExecutionError still matches the pre-rename BashSessionError', () => {
    expect(new ShellExecutionError('boom')).toBeInstanceOf(BashSessionError)
  })
})

describe('deprecated makeBash alias', () => {
  // Consumers key registries, hooks, and defaults lists on the runtime name, so an
  // alias that returned a tool named 'shell' would still break them (see awsarron/stan#6).
  it('keeps the pre-rename tool name', () => {
    expect(makeBash().name).toBe('bash')
  })

  it('matches makeShell apart from the name', () => {
    expect(makeBash().toolSpec).toEqual({ ...makeShell().toolSpec, name: 'bash' })
  })

  it('an explicit name still wins', () => {
    expect(makeBash({ name: 'sandbox_bash' }).name).toBe('sandbox_bash')
  })
})
