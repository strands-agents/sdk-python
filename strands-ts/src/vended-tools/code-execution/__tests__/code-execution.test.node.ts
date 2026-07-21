import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { codeExecution, makeCodeExecution } from '../index.js'
import {
  CODE_EXECUTION_DESCRIPTION,
  CodeSizeExceededError,
  DEFAULT_MAX_CODE_BYTES,
  SandboxNotConfiguredError,
  TRUNCATION_MARKER,
  type CodeExecutionOutput,
} from '../types.js'
import { NotASandboxLocalEnvironment } from '../../../sandbox/not-a-sandbox-local-environment.js'
import { PosixShellSandbox } from '../../../sandbox/posix-shell.js'
import type { ExecuteOptions } from '../../../sandbox/base.js'
import type { ExecutionResult, StreamChunk } from '../../../sandbox/types.js'
import { buildShellEnvPrefix } from '../../../sandbox/posix-shell.js'
import { shellQuote } from '../../../sandbox/constants.js'
import { streamProcess } from '../../../sandbox/stream-process.js'
import type { ToolContext } from '../../../index.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import type { Sandbox } from '../../../sandbox/base.js'

/**
 * A concrete PosixShellSandbox subclass used as an "isolating sandbox" stand-in.
 * The tool treats NotASandboxLocalEnvironment as "no isolation" and refuses to
 * execute; using a different subclass lets the happy-path tests run without
 * provisioning a real Docker/SSH backend.
 *
 * Executes `node`/`python3` in a temp working directory via `sh -c`, mirroring
 * the shape of the `TestSandbox` fixture used by the sandbox tests.
 */
class TestPosixSandbox extends PosixShellSandbox {
  constructor(readonly workingDir: string) {
    super()
  }

  async *executeStreaming(
    command: string,
    options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult, void, undefined> {
    const cwd = options?.cwd ?? this.workingDir
    const envPrefix = buildShellEnvPrefix(options?.env)
    const fullCommand = `cd ${shellQuote(cwd)} && ${envPrefix}${command}`
    yield* streamProcess('sh', ['-c', fullCommand], { timeout: options?.timeout, signal: options?.signal })
  }
}

// Skip on Windows — the sandbox stand-in requires POSIX shell.
describe.skipIf(process.platform === 'win32')('codeExecution tool', () => {
  let workDir: string
  let sandbox: TestPosixSandbox

  const createContext = (sandboxOverride?: Sandbox): ToolContext => {
    const agent = createMockAgent(
      sandboxOverride
        ? { extra: { sandbox: sandboxOverride } as unknown as Partial<import('../../../agent/agent.js').Agent> }
        : undefined
    )
    return {
      toolUse: { name: 'code_execution', toolUseId: 'test-id', input: {} },
      agent,
      invocationState: {},
      interrupt: () => {
        throw new Error('interrupt not available in mock context')
      },
    }
  }

  beforeEach(() => {
    workDir = mkdtempSync(join(tmpdir(), 'code-exec-test-'))
    sandbox = new TestPosixSandbox(workDir)
  })

  afterEach(() => {
    rmSync(workDir, { recursive: true, force: true })
  })

  describe('security: refuses when no isolating sandbox', () => {
    it('refuses when agent sandbox is the host default', async () => {
      // createMockAgent's sandbox getter returns the registered default,
      // which is NotASandboxLocalEnvironment (see __fixtures__/register-node-defaults.ts).
      await expect(codeExecution.invoke({ code: "console.log('nope')" }, createContext())).rejects.toThrow(
        SandboxNotConfiguredError
      )
    })

    it('refuses even when NotASandboxLocalEnvironment is bound at creation', async () => {
      const t = makeCodeExecution(new NotASandboxLocalEnvironment())
      await expect(t.invoke({ code: "console.log('nope')" }, createContext())).rejects.toThrow(
        SandboxNotConfiguredError
      )
    })
  })

  describe('security: input caps', () => {
    it('rejects oversized code before touching the sandbox', async () => {
      const t = makeCodeExecution(sandbox, { maxCodeBytes: 100 })
      const oversized = 'x'.repeat(200)
      await expect(t.invoke({ code: oversized }, createContext())).rejects.toThrow(CodeSizeExceededError)
    })

    it('enforces the default cap', async () => {
      const t = makeCodeExecution(sandbox)
      const code = 'a'.repeat(DEFAULT_MAX_CODE_BYTES + 1)
      await expect(t.invoke({ code }, createContext())).rejects.toThrow(CodeSizeExceededError)
    })

    it('truncates stdout over the output limit and appends the marker', async () => {
      const t = makeCodeExecution(sandbox, { maxOutputBytes: 32 })
      const result = (await t.invoke({ code: "console.log('x'.repeat(200))" }, createContext())) as CodeExecutionOutput
      expect(result.stdout.endsWith(TRUNCATION_MARKER)).toBe(true)
      expect(new TextEncoder().encode(result.stdout).byteLength).toBeLessThanOrEqual(
        32 + new TextEncoder().encode(TRUNCATION_MARKER).byteLength
      )
    })

    it('factory rejects nonpositive caps', () => {
      expect(() => makeCodeExecution({ maxCodeBytes: 0 })).toThrow(/maxCodeBytes/)
      expect(() => makeCodeExecution({ maxOutputBytes: -1 })).toThrow(/maxOutputBytes/)
      expect(() => makeCodeExecution({ defaultTimeout: 0 })).toThrow(/defaultTimeout/)
    })

    it('factory rejects non-finite caps and timeouts', () => {
      // `NaN <= 0` is false, so a bare comparison would silently disable the cap.
      expect(() => makeCodeExecution({ maxCodeBytes: Number.NaN })).toThrow(/finite/)
      expect(() => makeCodeExecution({ maxCodeBytes: Number.POSITIVE_INFINITY })).toThrow(/finite/)
      expect(() => makeCodeExecution({ maxOutputBytes: Number.NaN })).toThrow(/finite/)
      expect(() => makeCodeExecution({ defaultTimeout: Number.NaN })).toThrow(/finite/)
      expect(() => makeCodeExecution({ defaultTimeout: Number.POSITIVE_INFINITY })).toThrow(/finite/)
    })

    it('input schema rejects a nonpositive timeout', async () => {
      const t = makeCodeExecution(sandbox)
      // Zod's `.positive()` rejects zero and negative.
      await expect(t.invoke({ code: 'console.log(1)', timeout: 0 }, createContext())).rejects.toThrow()
      await expect(t.invoke({ code: 'console.log(1)', timeout: -1 }, createContext())).rejects.toThrow()
    })

    it('input schema rejects non-finite timeouts', async () => {
      const t = makeCodeExecution(sandbox)
      await expect(t.invoke({ code: 'console.log(1)', timeout: Number.NaN }, createContext())).rejects.toThrow()
      await expect(
        t.invoke({ code: 'console.log(1)', timeout: Number.POSITIVE_INFINITY }, createContext())
      ).rejects.toThrow()
    })
  })

  describe('happy path', () => {
    it('returns stdout for a trivial program', async () => {
      const t = makeCodeExecution(sandbox)
      const result = (await t.invoke({ code: 'console.log(2 + 2)' }, createContext())) as CodeExecutionOutput
      expect(result.stdout.trim()).toBe('4')
      expect(result.stderr).toBe('')
      expect(result.exitCode).toBe(0)
      expect(typeof result.elapsedMs).toBe('number')
      expect(result.elapsedMs).toBeGreaterThanOrEqual(0)
    })

    it('returns stderr from the interpreter', async () => {
      const t = makeCodeExecution(sandbox)
      const result = (await t.invoke({ code: "process.stderr.write('oops')" }, createContext())) as CodeExecutionOutput
      expect(result.stderr).toContain('oops')
    })

    it('returns a nonzero exit code when the code fails', async () => {
      const t = makeCodeExecution(sandbox)
      const result = (await t.invoke({ code: 'process.exit(3)' }, createContext())) as CodeExecutionOutput
      expect(result.exitCode).toBe(3)
    })

    it('surfaces a syntax error as a nonzero exit with stderr', async () => {
      const t = makeCodeExecution(sandbox)
      const result = (await t.invoke({ code: 'this is not valid js' }, createContext())) as CodeExecutionOutput
      expect(result.exitCode).not.toBe(0)
      expect(result.stderr.length).toBeGreaterThan(0)
    })

    it('unbound instance reads the sandbox from the agent context', async () => {
      // The default `codeExecution` reads from context.agent.sandbox at call time.
      const result = (await codeExecution.invoke(
        { code: "console.log('via-context')" },
        createContext(sandbox)
      )) as CodeExecutionOutput
      expect(result.stdout.trim()).toBe('via-context')
    })

    it('a bound-sandbox tool works without a tool context', async () => {
      // Mirrors the sibling http-request tool, which supports direct .invoke
      // without a context.
      const t = makeCodeExecution(sandbox)
      const result = (await t.invoke({ code: "console.log('no-ctx')" })) as CodeExecutionOutput
      expect(result.stdout.trim()).toBe('no-ctx')
    })
  })

  describe('cancellation', () => {
    it('surfaces timeout as AbortError on a runaway program', async () => {
      const t = makeCodeExecution(sandbox)
      // Loop forever; timeout is 200ms.
      const error = await t
        .invoke({ code: 'while (true) {}', timeout: 0.2 }, createContext())
        .catch((cause: unknown) => cause)
      expect(error).toBeInstanceOf(Error)
      expect((error as Error).name).toBe('AbortError')
      expect((error as Error).message).toMatch(/timed out/)
    })

    it('surfaces agent cancellation as AbortError', async () => {
      const controller = new AbortController()
      const t = makeCodeExecution(sandbox)
      const context = createContext()
      Object.defineProperty(context.agent, 'cancelSignal', { value: controller.signal, configurable: true })

      const invocation = t.invoke({ code: 'while (true) {}', timeout: 5 }, context)
      globalThis.setTimeout(() => controller.abort(), 50)
      const error = await invocation.catch((cause: unknown) => cause)
      expect(error).toBeInstanceOf(Error)
      expect((error as Error).name).toBe('AbortError')
    })
  })

  describe('tool metadata', () => {
    it('has the default name', () => {
      expect(codeExecution.name).toBe('code_execution')
    })

    it('accepts a custom name', () => {
      expect(makeCodeExecution({ name: 'sandbox_code' }).name).toBe('sandbox_code')
    })

    it('has the default description', () => {
      expect(makeCodeExecution().description).toBe(CODE_EXECUTION_DESCRIPTION)
    })

    it('accepts a custom description', () => {
      expect(makeCodeExecution({ description: 'custom desc' }).description).toBe('custom desc')
    })
  })
})
