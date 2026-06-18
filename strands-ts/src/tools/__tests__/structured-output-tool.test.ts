import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { StructuredOutputTool, STRUCTURED_OUTPUT_TOOL_NAME } from '../structured-output-tool.js'
import { JsonBlock, TextBlock, ToolResultBlock } from '../../types/messages.js'
import { createMockContext } from '../../__fixtures__/tool-helpers.js'
import type { JSONValue } from '../../types/json.js'

/** Helper to run the tool and return the final ToolResultBlock. */
async function runTool(tool: StructuredOutputTool, input: JSONValue): Promise<ToolResultBlock> {
  const context = createMockContext({ name: STRUCTURED_OUTPUT_TOOL_NAME, toolUseId: 'tool-1', input })
  const result = await tool.stream(context).next()
  return result.value as ToolResultBlock
}

describe('StructuredOutputTool', () => {
  describe('constructor', () => {
    it('builds tool spec from schema', () => {
      const tool = new StructuredOutputTool(z.object({ name: z.string() }).describe('A person schema'))

      expect(tool.name).toBe(STRUCTURED_OUTPUT_TOOL_NAME)
      expect(tool.toolSpec.name).toBe(STRUCTURED_OUTPUT_TOOL_NAME)
      expect(tool.toolSpec.inputSchema).toBeDefined()
      expect(tool.description).toContain('MUST only be invoked')
      expect(tool.description).toContain('A person schema')
    })

    it('uses base description when schema has no description', () => {
      const tool = new StructuredOutputTool(z.object({ name: z.string() }))

      expect(tool.description).toContain('MUST only be invoked')
      expect(tool.description).not.toContain('<description>')
    })
  })

  describe('stream', () => {
    it('returns success with validated JSON for valid input', async () => {
      const tool = new StructuredOutputTool(z.object({ name: z.string(), age: z.number() }))
      const result = await runTool(tool, { name: 'John', age: 30 })

      expect(result).toStrictEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new JsonBlock({ json: { name: 'John', age: 30 } })],
        })
      )
    })

    it('returns error with ZodError for invalid input', async () => {
      const tool = new StructuredOutputTool(z.object({ name: z.string(), age: z.number() }))
      const result = await runTool(tool, { name: 'John', age: 'invalid' })

      expect(result.status).toBe('error')
      expect(result.error).toBeInstanceOf(z.ZodError)
      expect((result.content[0] as TextBlock).text).toContain('age')
    })

    it('includes validation details for multiple fields', async () => {
      const tool = new StructuredOutputTool(z.object({ name: z.string(), age: z.number(), email: z.string().email() }))
      const result = await runTool(tool, { name: 123, age: 'invalid', email: 'not-email' })

      expect(result.status).toBe('error')
      const errorText = (result.content[0] as TextBlock).text
      expect(errorText).toContain('name')
      expect(errorText).toContain('age')
      expect(errorText).toContain('email')
    })

    it('validates nested objects', async () => {
      const tool = new StructuredOutputTool(z.object({ user: z.object({ name: z.string(), age: z.number() }) }))
      const result = await runTool(tool, { user: { name: 'John', age: 30 } })

      expect(result.status).toBe('success')
      expect((result.content[0] as JsonBlock).json).toEqual({ user: { name: 'John', age: 30 } })
    })

    it('validates arrays', async () => {
      const tool = new StructuredOutputTool(z.object({ items: z.array(z.string()) }))
      const result = await runTool(tool, { items: ['a', 'b', 'c'] })

      expect(result.status).toBe('success')
      expect((result.content[0] as JsonBlock).json).toEqual({ items: ['a', 'b', 'c'] })
    })

    it('handles optional fields', async () => {
      const tool = new StructuredOutputTool(z.object({ name: z.string(), age: z.number().optional() }))
      const result = await runTool(tool, { name: 'John' })

      expect(result.status).toBe('success')
      expect((result.content[0] as JsonBlock).json).toEqual({ name: 'John' })
    })

    it('returns error result for non-ZodError exceptions', async () => {
      const tool = new StructuredOutputTool(z.object({ value: z.string() }))
      vi.spyOn(tool['_schema'], 'parse').mockImplementation(() => {
        throw new Error('unexpected parse error')
      })
      const result = await runTool(tool, { value: 'valid' })

      expect(result.status).toBe('error')
      expect(result.error).toBeInstanceOf(Error)
      expect((result.content[0] as TextBlock).text).toContain('unexpected parse error')
    })
  })
})
