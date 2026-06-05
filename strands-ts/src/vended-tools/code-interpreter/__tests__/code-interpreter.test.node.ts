import { describe, it, expect } from 'vitest'
import { codeInterpreter } from '../index.js'
import type { ToolContext } from '../../../index.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import { TestSandbox } from '../../../__fixtures__/test-sandbox.node.js'
import { mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'

describe.skipIf(process.platform === 'win32')('codeInterpreter tool', () => {
  const createContext = (): ToolContext => {
    const workDir = mkdtempSync(join(tmpdir(), 'code-interp-test-'))
    const sandbox = new TestSandbox(workDir)
    const agent = createMockAgent({ extra: { sandbox } as any })
    return {
      toolUse: { name: 'codeInterpreter', toolUseId: 'test-id', input: {} },
      agent,
      invocationState: {},
      interrupt: () => {
        throw new Error('interrupt not available in mock context')
      },
    }
  }

  describe('code execution', () => {
    it('executes python code and returns stdout', async () => {
      const context = createContext()
      const result = (await codeInterpreter.invoke(
        { code: 'print("hello from python")', language: 'python3' },
        context
      )) as any

      expect(result.stdout).toContain('hello from python')
      expect(result.exitCode).toBe(0)
    })

    it('executes node code and returns stdout', async () => {
      const context = createContext()
      const result = (await codeInterpreter.invoke(
        { code: 'console.log("hello from node")', language: 'node' },
        context
      )) as any

      expect(result.stdout).toContain('hello from node')
      expect(result.exitCode).toBe(0)
    })

    it('captures stderr', async () => {
      const context = createContext()
      const result = (await codeInterpreter.invoke(
        { code: 'import sys; sys.stderr.write("oops")', language: 'python3' },
        context
      )) as any

      expect(result.stderr).toContain('oops')
    })

    it('returns non-zero exit code on error', async () => {
      const context = createContext()
      const result = (await codeInterpreter.invoke(
        { code: 'import sys; sys.exit(42)', language: 'python3' },
        context
      )) as any

      expect(result.exitCode).toBe(42)
    })

    it('returns empty outputFiles by default', async () => {
      const context = createContext()
      const result = (await codeInterpreter.invoke({ code: 'print("hi")', language: 'python3' }, context)) as any

      expect(result.outputFiles).toEqual([])
    })
  })

  describe('input validation', () => {
    it('rejects empty code', async () => {
      const context = createContext()
      await expect(codeInterpreter.invoke({ code: '', language: 'python3' }, context)).rejects.toThrow()
    })

    it('rejects empty language', async () => {
      const context = createContext()
      await expect(codeInterpreter.invoke({ code: 'print("hi")', language: '' }, context)).rejects.toThrow()
    })

    it('rejects invalid language characters', async () => {
      const context = createContext()
      await expect(codeInterpreter.invoke({ code: 'print("hi")', language: '../bin/evil' }, context)).rejects.toThrow(
        'invalid characters'
      )
    })
  })

  describe('timeout', () => {
    it('respects timeout option', async () => {
      const context = createContext()
      await expect(
        codeInterpreter.invoke({ code: 'import time; time.sleep(10)', language: 'python3', timeout: 0.2 }, context)
      ).rejects.toThrow()
    })

    it('defaults to 120s when no timeout is provided, and forwards an explicit value', async () => {
      const seenTimeouts: Array<number | undefined> = []
      const recordingSandbox = {
        executeCode: async (_code: string, _language: string, options?: { timeout?: number }) => {
          seenTimeouts.push(options?.timeout)
          return { type: 'executionResult', stdout: '', stderr: '', exitCode: 0, outputFiles: [] }
        },
      }
      const agent = createMockAgent({ extra: { sandbox: recordingSandbox } as any })
      const context: ToolContext = {
        toolUse: { name: 'codeInterpreter', toolUseId: 'test-id', input: {} },
        agent,
        invocationState: {},
        interrupt: () => {
          throw new Error('interrupt not available in mock context')
        },
      }

      await codeInterpreter.invoke({ code: 'print("hi")', language: 'python3' }, context)
      await codeInterpreter.invoke({ code: 'print("hi")', language: 'python3', timeout: 5 }, context)

      expect(seenTimeouts).toEqual([120, 5])
    })
  })

  describe('context handling', () => {
    it('throws when context is undefined', async () => {
      await expect(codeInterpreter.invoke({ code: 'print("hi")', language: 'python3' })).rejects.toThrow(
        'Tool context is required'
      )
    })
  })

  describe('tool properties', () => {
    it('has correct name', () => {
      expect(codeInterpreter.name).toBe('codeInterpreter')
    })

    it('has description', () => {
      expect(codeInterpreter.description).toBeDefined()
      expect(codeInterpreter.description.length).toBeGreaterThan(0)
    })
  })
})
