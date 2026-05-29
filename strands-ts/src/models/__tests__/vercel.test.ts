import { describe, it, expect, vi, beforeEach } from 'vitest'
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
} from '@ai-sdk/provider'
import { APICallError } from '@ai-sdk/provider'
import { VercelModel } from '../vercel.js'
import { ContextWindowOverflowError, ModelError, ModelThrottledError } from '../../errors.js'
import { logger } from '../../logging/logger.js'
import { collectIterator } from '../../__fixtures__/model-test-helpers.js'
import { Message, TextBlock, ToolUseBlock, ToolResultBlock, ReasoningBlock, JsonBlock } from '../../types/messages.js'
import { DocumentBlock, ImageBlock, VideoBlock } from '../../types/media.js'
import type { ToolSpec } from '../../tools/types.js'

/**
 * Creates a mock LanguageModelV3 that streams the given parts.
 */
function createMockModel(parts: LanguageModelV3StreamPart[]): LanguageModelV3 {
  return {
    specificationVersion: 'v3',
    provider: 'test',
    modelId: 'test-model',
    supportedUrls: {},
    doGenerate: vi.fn(),
    doStream: vi.fn(
      async (): Promise<LanguageModelV3StreamResult> => ({
        stream: new ReadableStream({
          start(controller) {
            for (const part of parts) {
              controller.enqueue(part)
            }
            controller.close()
          },
        }),
      })
    ),
  }
}

/** Standard usage object for finish events */
const testUsage = {
  inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 5, noCache: undefined, text: 5, reasoning: undefined },
}

/** Standard finish reason */
const stopFinish = { unified: 'stop' as const, raw: 'stop' }

/** Minimal stream parts that produce a valid (empty) response */
const minimalParts: LanguageModelV3StreamPart[] = [
  { type: 'stream-start', warnings: [] },
  { type: 'finish', usage: testUsage, finishReason: stopFinish },
]

/**
 * Creates a model backed by a mock that streams the given parts,
 * collects events, and returns the mock's doStream call args for inspection.
 */
function setupCaptureTest(
  parts: LanguageModelV3StreamPart[] = minimalParts,
  config?: Parameters<typeof VercelModel.prototype.updateConfig>[0]
): {
  model: VercelModel
  mock: LanguageModelV3
  callArgs: () => LanguageModelV3CallOptions
  collect: (messages: Message[], options?: Parameters<VercelModel['stream']>[1]) => ReturnType<typeof collectIterator>
} {
  const mock = createMockModel(parts)
  const model = new VercelModel({ provider: mock, ...config })
  return {
    model,
    mock,
    callArgs: () => (mock.doStream as ReturnType<typeof vi.fn>).mock.calls[0]![0] as LanguageModelV3CallOptions,
    collect: (messages, options) => collectIterator(model.stream(messages, options)),
  }
}

describe('VercelModel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('constructor and config', () => {
    it('uses model.modelId as default and allows override', () => {
      const mock = createMockModel([])
      expect(new VercelModel({ provider: mock }).getConfig().modelId).toBe('test-model')
      expect(new VercelModel({ provider: mock, modelId: 'custom-id' }).getConfig().modelId).toBe('custom-id')
    })

    it('passes through all config fields', () => {
      const mock = createMockModel([])
      const model = new VercelModel({
        provider: mock,
        maxTokens: 100,
        temperature: 0.5,
        topP: 0.9,
        topK: 40,
        presencePenalty: 0.5,
        frequencyPenalty: 0.3,
        stopSequences: ['END'],
        seed: 42,
      })
      expect(model.getConfig()).toStrictEqual({
        modelId: 'test-model',
        maxTokens: 100,
        temperature: 0.5,
        topP: 0.9,
        topK: 40,
        presencePenalty: 0.5,
        frequencyPenalty: 0.3,
        stopSequences: ['END'],
        seed: 42,
      })
    })

    it('updateConfig merges config and getConfig returns a copy', () => {
      const mock = createMockModel([])
      const model = new VercelModel({ provider: mock })
      model.updateConfig({ modelId: 'updated', maxTokens: 200 })
      const config1 = model.getConfig()
      const config2 = model.getConfig()
      expect(config1).toStrictEqual({ modelId: 'updated', maxTokens: 200 })
      expect(config1).not.toBe(config2)
    })
  })

  describe('stream', () => {
    describe('text streaming', () => {
      it('emits correct events for simple text response', async () => {
        const { model } = setupCaptureTest([
          { type: 'stream-start', warnings: [] },
          { type: 'text-start', id: 't1' },
          { type: 'text-delta', id: 't1', delta: 'Hello' },
          { type: 'text-delta', id: 't1', delta: ' world' },
          { type: 'text-end', id: 't1' },
          { type: 'finish', usage: testUsage, finishReason: stopFinish },
        ])

        const events = await collectIterator(model.stream([]))

        expect(events[0]).toMatchObject({ type: 'modelMessageStartEvent', role: 'assistant' })
        expect(events[1]).toMatchObject({ type: 'modelContentBlockStartEvent' })
        expect(events[2]).toMatchObject({
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'textDelta', text: 'Hello' },
        })
        expect(events[3]).toMatchObject({
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'textDelta', text: ' world' },
        })
        expect(events[4]).toMatchObject({ type: 'modelContentBlockStopEvent' })
        expect(events[5]).toMatchObject({ type: 'modelMetadataEvent' })
        expect(events[6]).toMatchObject({ type: 'modelMessageStopEvent', stopReason: 'endTurn' })
      })
    })

    describe('reasoning streaming', () => {
      it('emits reasoning content delta events', async () => {
        const { model } = setupCaptureTest([
          { type: 'stream-start', warnings: [] },
          { type: 'reasoning-start', id: 'r1' },
          { type: 'reasoning-delta', id: 'r1', delta: 'Let me think...' },
          { type: 'reasoning-end', id: 'r1' },
          { type: 'text-start', id: 't1' },
          { type: 'text-delta', id: 't1', delta: 'Answer' },
          { type: 'text-end', id: 't1' },
          { type: 'finish', usage: testUsage, finishReason: stopFinish },
        ])

        const events = await collectIterator(model.stream([]))

        const reasoningDelta = events.find(
          (e) => e.type === 'modelContentBlockDeltaEvent' && e.delta.type === 'reasoningContentDelta'
        )
        expect(reasoningDelta).toMatchObject({
          delta: { type: 'reasoningContentDelta', text: 'Let me think...' },
        })
      })
    })

    describe('tool call streaming', () => {
      it('synthesizes start/delta/stop from complete tool-call part', async () => {
        const { model } = setupCaptureTest([
          { type: 'stream-start', warnings: [] },
          { type: 'tool-call', toolCallId: 'call_1', toolName: 'calculator', input: '{"expr":"2+2"}' },
          { type: 'finish', usage: testUsage, finishReason: { unified: 'tool-calls', raw: 'tool_calls' } },
        ])

        const events = await collectIterator(model.stream([]))

        expect(events[1]).toMatchObject({
          type: 'modelContentBlockStartEvent',
          start: { type: 'toolUseStart', name: 'calculator', toolUseId: 'call_1' },
        })
        expect(events[2]).toMatchObject({
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'toolUseInputDelta', input: '{"expr":"2+2"}' },
        })
        expect(events[3]).toMatchObject({ type: 'modelContentBlockStopEvent' })
        expect(events[5]).toMatchObject({ type: 'modelMessageStopEvent', stopReason: 'toolUse' })
      })

      it('normalizes object tool-call input to JSON string', async () => {
        const { model } = setupCaptureTest([
          { type: 'stream-start', warnings: [] },
          {
            type: 'tool-call',
            toolCallId: 'call_1',
            toolName: 'calculator',
            input: { expr: '2+2' } as unknown as string,
          },
          { type: 'finish', usage: testUsage, finishReason: { unified: 'tool-calls', raw: 'tool_calls' } },
        ])

        const events = await collectIterator(model.stream([]))

        expect(events[2]).toMatchObject({
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'toolUseInputDelta', input: '{"expr":"2+2"}' },
        })
      })

      it('skips duplicate tool-call when incremental tool-input events were already emitted', async () => {
        const { model } = setupCaptureTest([
          { type: 'stream-start', warnings: [] },
          { type: 'tool-input-start', id: 'call_1', toolName: 'calculator' },
          { type: 'tool-input-delta', id: 'call_1', delta: '{"expr":"2+2"}' },
          { type: 'tool-input-end', id: 'call_1' },
          { type: 'tool-call', toolCallId: 'call_1', toolName: 'calculator', input: '{"expr":"2+2"}' },
          { type: 'finish', usage: testUsage, finishReason: { unified: 'tool-calls', raw: 'tool_calls' } },
        ])

        const events = await collectIterator(model.stream([]))

        const toolStarts = events.filter(
          (e) => e.type === 'modelContentBlockStartEvent' && e.start?.type === 'toolUseStart'
        )
        expect(toolStarts).toHaveLength(1)
      })

      it('emits tool use start/delta/stop events', async () => {
        const { model } = setupCaptureTest([
          { type: 'stream-start', warnings: [] },
          { type: 'tool-input-start', id: 'call_1', toolName: 'calculator' },
          { type: 'tool-input-delta', id: 'call_1', delta: '{"expr' },
          { type: 'tool-input-delta', id: 'call_1', delta: '":"2+2"}' },
          { type: 'tool-input-end', id: 'call_1' },
          { type: 'finish', usage: testUsage, finishReason: { unified: 'tool-calls', raw: 'tool_calls' } },
        ])

        const events = await collectIterator(model.stream([]))

        expect(events[1]).toMatchObject({
          type: 'modelContentBlockStartEvent',
          start: { type: 'toolUseStart', name: 'calculator', toolUseId: 'call_1' },
        })
        expect(events[2]).toMatchObject({
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'toolUseInputDelta', input: '{"expr' },
        })
        expect(events[3]).toMatchObject({
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'toolUseInputDelta', input: '":"2+2"}' },
        })
        expect(events[4]).toMatchObject({ type: 'modelContentBlockStopEvent' })
        expect(events[6]).toMatchObject({ type: 'modelMessageStopEvent', stopReason: 'toolUse' })
      })
    })

    describe('finish reasons', () => {
      it.each([
        ['stop', 'endTurn'],
        ['length', 'maxTokens'],
        ['content-filter', 'contentFiltered'],
        ['tool-calls', 'toolUse'],
        ['other', 'endTurn'],
      ] as const)('maps Language Model "%s" to Strands "%s"', async (unified, expected) => {
        const { model } = setupCaptureTest([
          { type: 'stream-start', warnings: [] },
          { type: 'finish', usage: testUsage, finishReason: { unified, raw: unified } },
        ])

        const events = await collectIterator(model.stream([]))
        const stopEvent = events.find((e) => e.type === 'modelMessageStopEvent')
        expect(stopEvent?.stopReason).toBe(expected)
      })

      it('throws ModelError for error finish reason', async () => {
        const { model } = setupCaptureTest([
          { type: 'stream-start', warnings: [] },
          { type: 'finish', usage: testUsage, finishReason: { unified: 'error', raw: 'internal_error' } },
        ])

        await expect(collectIterator(model.stream([]))).rejects.toThrow(ModelError)
      })
    })

    describe('usage mapping', () => {
      it('maps usage with cache tokens', async () => {
        const { model } = setupCaptureTest([
          { type: 'stream-start', warnings: [] },
          {
            type: 'finish',
            usage: {
              inputTokens: { total: 100, noCache: 80, cacheRead: 15, cacheWrite: 5 },
              outputTokens: { total: 50, text: 40, reasoning: 10 },
            },
            finishReason: stopFinish,
          },
        ])

        const events = await collectIterator(model.stream([]))
        const metaEvent = events.find((e) => e.type === 'modelMetadataEvent')

        expect(metaEvent?.usage).toEqual({
          inputTokens: 100,
          outputTokens: 50,
          totalTokens: 150,
          cacheReadInputTokens: 15,
          cacheWriteInputTokens: 5,
        })
      })

      it('handles undefined token counts', async () => {
        const { model } = setupCaptureTest([
          { type: 'stream-start', warnings: [] },
          {
            type: 'finish',
            usage: {
              inputTokens: { total: undefined, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
              outputTokens: { total: undefined, text: undefined, reasoning: undefined },
            },
            finishReason: stopFinish,
          },
        ])

        const events = await collectIterator(model.stream([]))
        const metaEvent = events.find((e) => e.type === 'modelMetadataEvent')

        expect(metaEvent?.usage).toEqual({
          inputTokens: 0,
          outputTokens: 0,
          totalTokens: 0,
        })
      })
    })

    describe('error handling', () => {
      it('throws ModelError on stream error part', async () => {
        const { model } = setupCaptureTest([
          { type: 'stream-start', warnings: [] },
          { type: 'error', error: new Error('rate limit exceeded') },
        ])

        await expect(collectIterator(model.stream([]))).rejects.toThrow(ModelError)
      })

      it('throws ModelError when doStream fails with generic error', async () => {
        const { mock, model } = setupCaptureTest()
        ;(mock.doStream as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('connection failed'))

        await expect(collectIterator(model.stream([]))).rejects.toThrow(
          'Language model stream error: connection failed'
        )
      })

      it('throws ModelThrottledError for APICallError with status 429', async () => {
        const { mock, model } = setupCaptureTest()
        ;(mock.doStream as ReturnType<typeof vi.fn>).mockRejectedValue(
          new APICallError({
            message: 'Too many requests',
            url: 'https://api.example.com',
            requestBodyValues: {},
            statusCode: 429,
          })
        )

        await expect(collectIterator(model.stream([]))).rejects.toThrow(ModelThrottledError)
      })

      it('throws ContextWindowOverflowError for APICallError with context overflow in responseBody', async () => {
        const { mock, model } = setupCaptureTest()
        ;(mock.doStream as ReturnType<typeof vi.fn>).mockRejectedValue(
          new APICallError({
            message: 'Bad request',
            url: 'https://api.example.com',
            requestBodyValues: {},
            statusCode: 400,
            responseBody: 'Input is too long for requested model',
          })
        )

        await expect(collectIterator(model.stream([]))).rejects.toThrow(ContextWindowOverflowError)
      })

      it('throws ContextWindowOverflowError for non-APICallError with context overflow message', async () => {
        const { mock, model } = setupCaptureTest()
        ;(mock.doStream as ReturnType<typeof vi.fn>).mockRejectedValue(
          new Error('context_length_exceeded: maximum context length is 128000')
        )

        await expect(collectIterator(model.stream([]))).rejects.toThrow(ContextWindowOverflowError)
      })

      it('classifies errors thrown during reader.read()', async () => {
        const mock = createMockModel([])
        ;(mock.doStream as ReturnType<typeof vi.fn>).mockResolvedValue({
          stream: new ReadableStream({
            start(controller) {
              controller.enqueue({ type: 'stream-start', warnings: [] })
              controller.error(
                new APICallError({
                  message: 'Too many requests',
                  url: 'https://api.example.com',
                  requestBodyValues: {},
                  statusCode: 429,
                })
              )
            },
          }),
        })
        const model = new VercelModel({ provider: mock })

        await expect(collectIterator(model.stream([]))).rejects.toThrow(ModelThrottledError)
      })
    })

    describe('call options forwarding', () => {
      it('forwards config to doStream', async () => {
        const { collect, callArgs } = setupCaptureTest(minimalParts, {
          maxTokens: 100,
          temperature: 0.7,
          topP: 0.95,
          topK: 40,
          presencePenalty: 0.5,
          frequencyPenalty: 0.3,
          stopSequences: ['END'],
          seed: 42,
        })
        await collect([])

        expect(callArgs()).toMatchObject({
          maxOutputTokens: 100,
          temperature: 0.7,
          topP: 0.95,
          topK: 40,
          presencePenalty: 0.5,
          frequencyPenalty: 0.3,
          stopSequences: ['END'],
          seed: 42,
        })
      })

      it('omits undefined config values', async () => {
        const { collect, callArgs } = setupCaptureTest()
        await collect([])

        const args = callArgs()
        for (const key of [
          'maxOutputTokens',
          'temperature',
          'topP',
          'topK',
          'presencePenalty',
          'frequencyPenalty',
          'stopSequences',
          'seed',
        ]) {
          expect(args).not.toHaveProperty(key)
        }
      })
    })

    it('logs response-metadata at debug level', async () => {
      const debugSpy = vi.spyOn(logger, 'debug').mockImplementation(() => {})
      const { model } = setupCaptureTest([
        { type: 'stream-start', warnings: [] },
        { type: 'text-start', id: 't1' },
        { type: 'text-delta', id: 't1', delta: 'Hi' },
        { type: 'text-end', id: 't1' },
        { type: 'response-metadata', id: 'resp1', timestamp: new Date() } as any,
        { type: 'finish', usage: testUsage, finishReason: stopFinish },
      ])

      const events = await collectIterator(model.stream([]))
      expect(events.map((e) => e.type)).not.toContain('response-metadata')
      expect(debugSpy).toHaveBeenCalled()
      debugSpy.mockRestore()
    })
  })

  describe('message formatting', () => {
    describe('system prompt', () => {
      it('formats string system prompt', async () => {
        const { collect, callArgs } = setupCaptureTest()
        await collect([], { systemPrompt: 'You are helpful.' })

        expect(callArgs().prompt[0]).toEqual({ role: 'system', content: 'You are helpful.' })
      })

      it('formats system prompt content blocks', async () => {
        const { collect, callArgs } = setupCaptureTest()
        await collect([], { systemPrompt: [{ text: 'Part 1' }, { text: 'Part 2' }] as any })

        expect(callArgs().prompt[0]).toEqual({ role: 'system', content: 'Part 1Part 2' })
      })

      it('ignores cache points in system prompt', async () => {
        const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
        const { collect, callArgs } = setupCaptureTest()
        await collect([], {
          systemPrompt: [
            { type: 'textBlock', text: 'Hello' },
            { type: 'cachePointBlock', cacheType: 'default' },
          ] as any,
        })

        expect(callArgs().prompt[0]).toEqual({ role: 'system', content: 'Hello' })
        expect(warnSpy).toHaveBeenCalled()
        warnSpy.mockRestore()
      })

      it('ignores guard content in system prompt', async () => {
        const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
        const { collect, callArgs } = setupCaptureTest()
        await collect([], {
          systemPrompt: [
            { type: 'textBlock', text: 'Hello' },
            { type: 'guardContentBlock', guardContent: {} },
          ] as any,
        })

        expect(callArgs().prompt[0]).toEqual({ role: 'system', content: 'Hello' })
        expect(warnSpy).toHaveBeenCalled()
        warnSpy.mockRestore()
      })
    })

    describe('user messages', () => {
      it('formats user text message', async () => {
        const { collect, callArgs } = setupCaptureTest()
        await collect([new Message({ role: 'user', content: [new TextBlock('Hello')] })])

        const userMsg = callArgs().prompt[0] as any
        expect(userMsg.role).toBe('user')
        expect(userMsg.content[0]).toEqual({ type: 'text', text: 'Hello' })
      })

      it('formats image blocks with bytes and URL sources', async () => {
        const { collect, callArgs } = setupCaptureTest()
        await collect([
          new Message({
            role: 'user',
            content: [
              new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1, 2, 3]) } }),
              new ImageBlock({ format: 'png', source: { url: 'https://example.com/image.png' } }),
            ],
          }),
        ])

        const userMsg = callArgs().prompt[0] as any
        expect(userMsg.content[0]).toMatchObject({ type: 'file', mediaType: 'image/png' })
        expect(userMsg.content[0].data).toBeInstanceOf(Uint8Array)
        expect(userMsg.content[1]).toMatchObject({ type: 'file', mediaType: 'image/png' })
        expect(userMsg.content[1].data).toBeInstanceOf(URL)
        expect(userMsg.content[1].data.href).toBe('https://example.com/image.png')
      })

      it('formats document content block source as text parts', async () => {
        const { collect, callArgs } = setupCaptureTest()
        await collect([
          new Message({
            role: 'user',
            content: [
              new DocumentBlock({
                format: 'txt',
                name: 'doc',
                source: { content: [{ text: 'paragraph 1' }, { text: 'paragraph 2' }] },
              }),
            ],
          }),
        ])

        const userMsg = callArgs().prompt[0] as any
        expect(userMsg.content).toHaveLength(2)
        expect(userMsg.content[0]).toEqual({ type: 'text', text: 'paragraph 1' })
        expect(userMsg.content[1]).toEqual({ type: 'text', text: 'paragraph 2' })
      })

      it('formats video bytes in user messages', async () => {
        const { collect, callArgs } = setupCaptureTest()
        await collect([
          new Message({
            role: 'user',
            content: [new VideoBlock({ format: 'mp4', source: { bytes: new Uint8Array([1, 2]) } })],
          }),
        ])

        const userMsg = callArgs().prompt[0] as any
        expect(userMsg.content[0]).toMatchObject({ type: 'file', mediaType: 'video/mp4' })
      })

      it.each([
        {
          name: 'image S3 source',
          block: new ImageBlock({
            format: 'png',
            source: { location: { type: 's3', uri: 's3://bucket/key', bucketOwner: '' } },
          }),
        },
        {
          name: 'video S3 source',
          block: new VideoBlock({
            format: 'mp4',
            source: { location: { type: 's3', uri: 's3://bucket/video', bucketOwner: '' } },
          }),
        },
      ])('skips unsupported $name', async ({ block }) => {
        const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
        const { collect, callArgs } = setupCaptureTest()
        await collect([new Message({ role: 'user', content: [block] })])

        expect(callArgs().prompt).toHaveLength(0)
        expect(warnSpy).toHaveBeenCalled()
        warnSpy.mockRestore()
      })
    })

    describe('assistant messages', () => {
      it('formats text and tool use blocks', async () => {
        const { collect, callArgs } = setupCaptureTest()
        await collect([
          new Message({
            role: 'assistant',
            content: [
              new TextBlock('Let me calculate'),
              new ToolUseBlock({ name: 'calc', toolUseId: 'tu1', input: { x: 1 } }),
            ],
          }),
        ])

        const prompt = callArgs().prompt
        expect(prompt).toHaveLength(1)
        const assistantMsg = prompt[0] as any
        expect(assistantMsg.role).toBe('assistant')
        expect(assistantMsg.content).toHaveLength(2)
        expect(assistantMsg.content[0]).toEqual({ type: 'text', text: 'Let me calculate' })
        expect(assistantMsg.content[1].type).toBe('tool-call')
        expect(assistantMsg.content[1].toolCallId).toBe('tu1')
      })

      it('formats reasoning blocks', async () => {
        const { collect, callArgs } = setupCaptureTest()
        await collect([
          new Message({
            role: 'assistant',
            content: [new ReasoningBlock({ text: 'thinking...' })],
          }),
        ])

        const assistantMsg = callArgs().prompt[0] as any
        expect(assistantMsg.content[0]).toEqual({ type: 'reasoning', text: 'thinking...' })
      })

      it('warns and skips tool results in assistant messages', async () => {
        const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
        const { collect, callArgs } = setupCaptureTest()
        await collect([
          new Message({
            role: 'assistant',
            content: [
              new ToolUseBlock({ name: 'calc', toolUseId: 'tu1', input: {} }),
              new ToolResultBlock({ toolUseId: 'tu1', status: 'success', content: [new TextBlock('42')] }),
            ],
          }),
        ])

        const prompt = callArgs().prompt
        expect(prompt).toHaveLength(1)
        const assistantMsg = prompt[0] as any
        expect(assistantMsg.content).toHaveLength(1)
        expect(assistantMsg.content[0].type).toBe('tool-call')
        expect(warnSpy).toHaveBeenCalled()
        warnSpy.mockRestore()
      })

      it('handles assistant message with no tool results', async () => {
        const { collect, callArgs } = setupCaptureTest()
        await collect([new Message({ role: 'assistant', content: [new TextBlock('Just text')] })])

        const prompt = callArgs().prompt
        expect(prompt).toHaveLength(1)
        expect((prompt[0] as any).role).toBe('assistant')
      })
    })
    describe('tool result output formatting', () => {
      function toolResultMessages(
        content: ToolResultBlock['content'],
        status: 'success' | 'error' = 'success'
      ): Message[] {
        return [
          new Message({
            role: 'assistant',
            content: [new ToolUseBlock({ name: 'tool', toolUseId: 'tu1', input: {} })],
          }),
          new Message({
            role: 'user',
            content: [new ToolResultBlock({ toolUseId: 'tu1', status, content })],
          }),
        ]
      }

      async function getToolOutput(content: ToolResultBlock['content'], status?: 'success' | 'error'): Promise<any> {
        const { collect, callArgs } = setupCaptureTest()
        await collect(toolResultMessages(content, status))
        return (callArgs().prompt.find((m: any) => m.role === 'tool') as any).content[0].output
      }

      it('formats error status with text and fallback', async () => {
        expect(await getToolOutput([new TextBlock('boom')], 'error')).toStrictEqual({
          type: 'error-text',
          value: 'boom',
        })
        expect(await getToolOutput([], 'error')).toStrictEqual({
          type: 'error-text',
          value: 'Tool execution failed',
        })
      })

      it.each([
        { name: 'text', content: [new TextBlock('result')], expected: [{ type: 'text', text: 'result' }] },
        {
          name: 'json',
          content: [new JsonBlock({ json: { k: 'v' } })],
          expected: [{ type: 'text', text: '{"k":"v"}' }],
        },
        {
          name: 'image URL',
          content: [new ImageBlock({ format: 'png', source: { url: 'https://example.com/img.png' } })],
          expected: [{ type: 'text', text: 'https://example.com/img.png' }],
        },
        {
          name: 'document text',
          content: [new DocumentBlock({ format: 'txt', name: 'd', source: { text: 'doc' } })],
          expected: [{ type: 'text', text: 'doc' }],
        },
        {
          name: 'document content blocks',
          content: [
            new DocumentBlock({ format: 'txt', name: 'd', source: { content: [{ text: 'p1' }, { text: 'p2' }] } }),
          ],
          expected: [
            { type: 'text', text: 'p1' },
            { type: 'text', text: 'p2' },
          ],
        },
      ])('formats $name content as text', async ({ content, expected }) => {
        expect(await getToolOutput(content)).toStrictEqual({ type: 'content', value: expected })
      })

      it.each([
        {
          name: 'image bytes',
          content: new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1]) } }),
          mediaType: 'image/png',
        },
        {
          name: 'document bytes',
          content: new DocumentBlock({ format: 'pdf', name: 'd', source: { bytes: new Uint8Array([1]) } }),
          mediaType: 'application/pdf',
        },
        {
          name: 'video bytes',
          content: new VideoBlock({ format: 'mp4', source: { bytes: new Uint8Array([1]) } }),
          mediaType: 'video/mp4',
        },
      ])('formats $name as file-data', async ({ content, mediaType }) => {
        const output = await getToolOutput([content])
        expect(output.value[0]).toMatchObject({ type: 'file-data', mediaType })
      })

      it.each([
        {
          name: 'image S3',
          block: new ImageBlock({
            format: 'png',
            source: { location: { type: 's3', uri: 's3://b/k', bucketOwner: '' } },
          }),
        },
        {
          name: 'document S3',
          block: new DocumentBlock({
            format: 'pdf',
            name: 'd',
            source: { location: { type: 's3', uri: 's3://b/k', bucketOwner: '' } },
          } as any),
        },
        {
          name: 'video S3',
          block: new VideoBlock({
            format: 'mp4',
            source: { location: { type: 's3', uri: 's3://b/k', bucketOwner: '' } },
          }),
        },
      ])('warns on unsupported $name source', async ({ block }) => {
        const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
        await getToolOutput([block])
        expect(warnSpy).toHaveBeenCalled()
        warnSpy.mockRestore()
      })
    })
  })

  describe('tool formatting', () => {
    it('formats tool specs', async () => {
      const tools: ToolSpec[] = [
        {
          name: 'calculator',
          description: 'Does math',
          inputSchema: { type: 'object', properties: { expr: { type: 'string' } }, required: ['expr'] },
        },
      ]

      const { collect, callArgs } = setupCaptureTest()
      await collect([], { toolSpecs: tools })

      expect(callArgs().tools![0]).toMatchObject({
        type: 'function',
        name: 'calculator',
        description: 'Does math',
      })
    })

    it('handles tool spec with no inputSchema', async () => {
      const tools: ToolSpec[] = [{ name: 'noop', description: 'Does nothing' }]

      const { collect, callArgs } = setupCaptureTest()
      await collect([], { toolSpecs: tools })

      const tool = callArgs().tools![0]!
      expect(tool.type).toBe('function')
      if (tool.type === 'function') {
        expect(tool.inputSchema).toEqual({ type: 'object', properties: {} })
      }
    })

    it.each([
      { name: 'auto', input: { auto: {} }, expected: { type: 'auto' } },
      { name: 'any -> required', input: { any: {} }, expected: { type: 'required' } },
      { name: 'specific tool', input: { tool: { name: 'calc' } }, expected: { type: 'tool', toolName: 'calc' } },
    ])('maps toolChoice $name', async ({ input, expected }) => {
      const { collect, callArgs } = setupCaptureTest()
      await collect([], { toolChoice: input })

      expect(callArgs().toolChoice).toEqual(expected)
    })

    it('omits tools when not provided', async () => {
      const { collect, callArgs } = setupCaptureTest()
      await collect([])

      const args = callArgs()
      expect(args).not.toHaveProperty('tools')
      expect(args).not.toHaveProperty('toolChoice')
    })
  })
})
