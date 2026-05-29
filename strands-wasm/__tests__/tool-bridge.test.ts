import { describe, it, expect, vi, beforeEach } from 'vitest'
import { createTools } from '../entry'
import { callTool } from 'strands:agent/tool-provider'

const emptyToolContext = { toolUse: { toolUseId: '' } } as any

describe('createTools', () => {
  describe('spec handling', () => {
    it('returns undefined for undefined specs', () => {
      expect(createTools(undefined)).toStrictEqual(undefined)
    })

    it('returns undefined for empty array', () => {
      expect(createTools([])).toStrictEqual(undefined)
    })

    it('creates FunctionTool with correct properties', () => {
      const specs = [
        {
          name: 'calculator',
          description: 'Does math',
          inputSchema: '{"type":"object","properties":{"expression":{"type":"string"}}}',
        },
      ]
      const tools = createTools(specs)
      expect(tools).toHaveLength(1)
      expect(tools![0].name).toBe('calculator')
      expect(tools![0].description).toBe('Does math')
    })

    it('parses inputSchema from JSON string', () => {
      const tools = createTools([{ name: 'x', description: 'y', inputSchema: '{"type":"object"}' }])
      expect(tools![0].toolSpec.inputSchema).toStrictEqual({ type: 'object' })
    })
  })

  describe('callback behavior', () => {
    const makeTools = (name = 'calc') => createTools([{ name, description: 'math', inputSchema: '{"type":"object"}' }])!

    beforeEach(() => {
      vi.mocked(callTool).mockReset()
    })

    it('calls callTool with correct args', async () => {
      const tools = makeTools()
      vi.mocked(callTool).mockReturnValue('{"result": 42}')
      const toolContext = { toolUse: { toolUseId: 'tu-123' } }
      await tools[0].invoke({ expression: '1+1' }, toolContext)
      expect(callTool).toHaveBeenCalledWith({
        name: 'calc',
        input: '{"expression":"1+1"}',
        toolUseId: 'tu-123',
      })
    })

    it('strips {status, content} wrapper from host result', async () => {
      const tools = makeTools()
      vi.mocked(callTool).mockReturnValue(JSON.stringify({ status: 'success', content: [{ text: 'hello' }] }))
      const result = await tools[0].invoke({}, emptyToolContext)
      expect(result).toStrictEqual([{ text: 'hello' }])
    })

    it('handles WIT Result ok variant', async () => {
      const tools = makeTools()
      vi.mocked(callTool).mockReturnValue({
        tag: 'ok',
        val: JSON.stringify({ status: 'success', content: [{ text: 'ok' }] }),
      })
      const result = await tools[0].invoke({}, emptyToolContext)
      expect(result).toStrictEqual([{ text: 'ok' }])
    })

    it('throws on WIT Result err variant', async () => {
      const tools = makeTools()
      vi.mocked(callTool).mockReturnValue({ tag: 'err', val: 'tool failed' })
      await expect(tools[0].invoke({}, emptyToolContext)).rejects.toThrow('tool failed')
    })

    it('propagates host exceptions', async () => {
      const tools = makeTools()
      vi.mocked(callTool).mockImplementation(() => {
        throw new Error('host crashed')
      })
      await expect(tools[0].invoke({}, emptyToolContext)).rejects.toThrow('host crashed')
    })

    it('uses empty string for toolUseId when context is missing', async () => {
      const tools = makeTools()
      vi.mocked(callTool).mockReturnValue('{"value": 1}')
      await tools[0].invoke({ x: 1 }, emptyToolContext)
      expect(callTool).toHaveBeenCalledWith({
        name: 'calc',
        input: '{"x":1}',
        toolUseId: '',
      })
    })

    it('returns parsed result directly when not a {status, content} wrapper', async () => {
      const tools = makeTools()
      vi.mocked(callTool).mockReturnValue('{"custom": "data"}')
      const result = await tools[0].invoke({}, emptyToolContext)
      expect(result).toStrictEqual({ custom: 'data' })
    })
  })
})
