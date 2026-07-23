import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { fileRead, makeFileRead, DEFAULT_FILE_READ_DESCRIPTION } from '../file-read.js'
import type { ToolContext } from '../../../index.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import { promises as fs } from 'fs'
import * as path from 'path'
import { tmpdir } from 'os'

describe('fileRead tool', () => {
  let testDir: string
  let context: ToolContext

  const createFreshContext = (): ToolContext => {
    const agent = createMockAgent()
    return {
      toolUse: { name: 'fileRead', toolUseId: 'test-id', input: {} },
      agent,
      invocationState: {},
      interrupt: () => {
        throw new Error('interrupt not available in mock context')
      },
    }
  }

  const createTestFile = async (filename: string, content: string): Promise<string> => {
    const filePath = path.join(testDir, filename)
    const dir = path.dirname(filePath)
    await fs.mkdir(dir, { recursive: true })
    await fs.writeFile(filePath, content, 'utf-8')
    return filePath
  }

  beforeEach(async () => {
    testDir = path.join(tmpdir(), `file-read-test-${Date.now()}-${Math.random().toString(36).slice(2)}`)
    await fs.mkdir(testDir, { recursive: true })
    context = createFreshContext()
  })

  afterEach(async () => {
    try {
      await fs.rm(testDir, { recursive: true, force: true })
    } catch {
      // ignore cleanup errors
    }
  })

  describe('happy path', () => {
    it('reads a file with line numbers', async () => {
      const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3')
      const result = (await fileRead.invoke({ path: filePath }, context)) as string
      expect(result).toContain("Here's the result of running `cat -n`")
      expect(result).toContain('     1  Line 1')
      expect(result).toContain('     3  Line 3')
    })

    it('honours view_range', async () => {
      const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5')
      const result = (await fileRead.invoke({ path: filePath, view_range: [2, 4] }, context)) as string
      expect(result).toContain('     2  Line 2')
      expect(result).toContain('     4  Line 4')
      expect(result).not.toContain('     1  ')
      expect(result).not.toContain('     5  ')
    })

    it('supports -1 for end-of-file', async () => {
      const filePath = await createTestFile('test.txt', 'Line 1\nLine 2\nLine 3')
      const result = (await fileRead.invoke({ path: filePath, view_range: [2, -1] }, context)) as string
      expect(result).toContain('     2  Line 2')
      expect(result).toContain('     3  Line 3')
      expect(result).not.toContain('     1  ')
    })

    it('lists a directory', async () => {
      const dirPath = path.join(testDir, 'sub')
      await fs.mkdir(dirPath, { recursive: true })
      await fs.writeFile(path.join(dirPath, 'a.txt'), 'a', 'utf-8')
      await fs.writeFile(path.join(dirPath, 'b.txt'), 'b', 'utf-8')
      const result = (await fileRead.invoke({ path: dirPath }, context)) as string
      expect(result).toContain('a.txt')
      expect(result).toContain('b.txt')
    })
  })

  describe('security delegation', () => {
    // The shim adds no validation logic — file_editor owns the security surface
    // (absolute path, `..` traversal, size cap, view_range bounds, directory
    // checks). We prove delegation with a single traversal probe; the exhaustive
    // security suite lives with file_editor.
    it('surfaces file_editor validation through the shim', async () => {
      await expect(fileRead.invoke({ path: '/tmp/../etc/passwd' }, context)).rejects.toThrow('path traversal')
    })
  })

  describe('tool metadata', () => {
    it('has the default name and description', () => {
      expect(fileRead.name).toBe('fileRead')
      expect(fileRead.description).toBe(DEFAULT_FILE_READ_DESCRIPTION)
    })

    it('allows overriding name and description', () => {
      const custom = makeFileRead({ name: 'reader', description: 'custom read tool' })
      expect(custom.name).toBe('reader')
      expect(custom.description).toBe('custom read tool')
    })

    it('exposes a strictly read-only schema (no write parameters)', () => {
      const schema = fileRead.toolSpec.inputSchema as { properties: Record<string, unknown> } | undefined
      expect(schema).toBeDefined()
      const props = schema!.properties
      expect(Object.keys(props).sort()).toEqual(['path', 'view_range'])
      for (const banned of ['command', 'file_text', 'old_str', 'new_str', 'insert_line']) {
        expect(props).not.toHaveProperty(banned)
      }
    })

    it('rejects unknown schema fields at invocation (write params cannot leak in)', async () => {
      const filePath = await createTestFile('test.txt', 'ok')
      // The schema is `.strict()`: any key outside `{ path, view_range }` is a
      // hard validation error, not a silent strip. This gives the model clear
      // feedback if it tries to route a write through the read tool, and the
      // shim's hard-coded `command: 'view'` means even without .strict() the
      // smuggle would be harmless.
      const smuggled = { path: filePath, command: 'create', file_text: 'pwned' } as unknown as {
        path: string
        view_range?: [number, number]
      }
      await expect(fileRead.invoke(smuggled, context)).rejects.toThrow()
      // The smuggled write did not happen: file content is unchanged.
      expect(await fs.readFile(filePath, 'utf-8')).toBe('ok')
    })
  })
})
