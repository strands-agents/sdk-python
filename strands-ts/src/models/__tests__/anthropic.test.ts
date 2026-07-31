import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import Anthropic from '@anthropic-ai/sdk'
import { isNode } from '../../__fixtures__/environment.js'
import { AnthropicModel } from '../anthropic.js'
import { ContextWindowOverflowError, ModelThrottledError } from '../../errors.js'
import { collectIterator } from '../../__fixtures__/model-test-helpers.js'
import {
  Message,
  TextBlock,
  CachePointBlock,
  GuardContentBlock,
  ToolResultBlock,
  JsonBlock,
  ReasoningBlock,
} from '../../types/messages.js'
import { ImageBlock, DocumentBlock, VideoBlock } from '../../types/media.js'
import { warnOnce } from '../../logging/warn-once.js'

/**
 * Helper to create a mock Anthropic client with streaming support
 */
function createMockClient(streamGenerator: () => AsyncGenerator<unknown>): Anthropic {
  return {
    messages: {
      stream: vi.fn(() => streamGenerator()),
      countTokens: vi.fn(),
    },
  } as unknown as Anthropic
}

// Mock the Anthropic SDK
vi.mock('@anthropic-ai/sdk', () => {
  const mockConstructor = vi.fn(function () {
    return {
      messages: {
        stream: vi.fn(),
        countTokens: vi.fn(),
      },
    }
  })
  return {
    default: mockConstructor,
  }
})

vi.mock('../../logging/warn-once.js', () => ({
  warnOnce: vi.fn(),
}))

describe('AnthropicModel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    if (isNode) {
      vi.stubEnv('ANTHROPIC_API_KEY', 'sk-ant-test-env')
    }
  })

  afterEach(() => {
    vi.clearAllMocks()
    if (isNode) {
      vi.unstubAllEnvs()
    }
  })

  describe('constructor', () => {
    it('creates an instance with default configuration', () => {
      const provider = new AnthropicModel({ apiKey: 'sk-ant-test' })
      const config = provider.getConfig()
      expect(config.modelId).toBe('claude-sonnet-4-6')
      expect(config.maxTokens).toBe(64_000)
    })

    it('uses provided model ID', () => {
      const customModelId = 'claude-3-opus-20240229'
      const provider = new AnthropicModel({ modelId: customModelId, apiKey: 'sk-ant-test' })
      expect(provider.getConfig().modelId).toBe(customModelId)
    })

    it('uses API key from constructor parameter', () => {
      const apiKey = 'sk-explicit'
      new AnthropicModel({ apiKey })
      expect(Anthropic).toHaveBeenCalledWith(
        expect.objectContaining({
          apiKey,
        })
      )
    })

    if (isNode) {
      it('uses API key from environment variable', () => {
        vi.stubEnv('ANTHROPIC_API_KEY', 'sk-from-env')
        new AnthropicModel()
        expect(Anthropic).toHaveBeenCalled()
      })

      it('throws error when no API key is available', () => {
        vi.stubEnv('ANTHROPIC_API_KEY', '')
        expect(() => new AnthropicModel()).toThrow('Anthropic API key is required')
      })
    }

    it('uses provided client instance', () => {
      const mockClient = {} as Anthropic
      const provider = new AnthropicModel({ client: mockClient })
      expect(Anthropic).not.toHaveBeenCalled()
      expect(provider).toBeDefined()
    })

    it('warns when maxTokens is not explicitly set', () => {
      new AnthropicModel({ apiKey: 'sk-ant-test' })
      expect(warnOnce).toHaveBeenCalledWith(
        expect.objectContaining({ warn: expect.any(Function) }),
        expect.stringContaining('using default maxTokens')
      )
    })

    it('does not warn when maxTokens is explicitly set', () => {
      new AnthropicModel({ apiKey: 'sk-ant-test', maxTokens: 4096 })
      expect(warnOnce).not.toHaveBeenCalledWith(
        expect.objectContaining({ warn: expect.any(Function) }),
        expect.stringContaining('using default maxTokens')
      )
    })

    it('warns when modelId is not explicitly set', () => {
      new AnthropicModel({ apiKey: 'sk-ant-test' })
      expect(warnOnce).toHaveBeenCalledWith(
        expect.objectContaining({ warn: expect.any(Function) }),
        expect.stringContaining('using default modelId')
      )
    })

    it('does not warn when modelId is explicitly set', () => {
      new AnthropicModel({ apiKey: 'sk-ant-test', modelId: 'claude-3-opus-20240229' })
      expect(warnOnce).not.toHaveBeenCalledWith(
        expect.objectContaining({ warn: expect.any(Function) }),
        expect.stringContaining('using default modelId')
      )
    })

    it('auto-populates contextWindowLimit from model ID lookup', () => {
      const provider = new AnthropicModel({ apiKey: 'sk-test', modelId: 'claude-sonnet-4-20250514' })
      expect(provider.getConfig()).toStrictEqual({
        modelId: 'claude-sonnet-4-20250514',
        maxTokens: 64_000,
        contextWindowLimit: 1_000_000,
      })
    })

    it('auto-populates contextWindowLimit for default model ID', () => {
      const provider = new AnthropicModel({ apiKey: 'sk-test' })
      expect(provider.getConfig()).toStrictEqual({
        modelId: 'claude-sonnet-4-6',
        maxTokens: 64_000,
        contextWindowLimit: 1_000_000,
      })
    })

    it('does not override explicit contextWindowLimit', () => {
      const provider = new AnthropicModel({
        apiKey: 'sk-test',
        modelId: 'claude-sonnet-4-20250514',
        contextWindowLimit: 100_000,
      })
      expect(provider.getConfig()).toStrictEqual({
        modelId: 'claude-sonnet-4-20250514',
        maxTokens: 64_000,
        contextWindowLimit: 100_000,
      })
    })

    it('leaves contextWindowLimit undefined for unknown model IDs', () => {
      const provider = new AnthropicModel({ apiKey: 'sk-test', modelId: 'unknown-model' })
      expect(provider.getConfig()).toStrictEqual({
        modelId: 'unknown-model',
        maxTokens: 64_000,
      })
    })
  })

  describe('updateConfig', () => {
    it('merges new config with existing config', () => {
      const provider = new AnthropicModel({ apiKey: 'sk-test', temperature: 0.5 })
      provider.updateConfig({ temperature: 0.8, maxTokens: 8192 })
      expect(provider.getConfig()).toMatchObject({
        temperature: 0.8,
        maxTokens: 8192,
      })
    })

    it('re-resolves contextWindowLimit when modelId changes and it was auto-resolved', () => {
      const provider = new AnthropicModel({ apiKey: 'sk-test' })
      expect(provider.getConfig().contextWindowLimit).toBe(1_000_000) // claude-sonnet-4-6 default

      provider.updateConfig({ modelId: 'claude-sonnet-4-20250514' })
      expect(provider.getConfig().contextWindowLimit).toBe(1_000_000) // claude-sonnet-4-20250514 value
    })

    it('preserves explicit contextWindowLimit when modelId changes', () => {
      const provider = new AnthropicModel({ apiKey: 'sk-test', contextWindowLimit: 50_000 })
      expect(provider.getConfig().contextWindowLimit).toBe(50_000)

      provider.updateConfig({ modelId: 'claude-sonnet-4-20250514' })
      expect(provider.getConfig().contextWindowLimit).toBe(50_000) // preserved
    })
  })

  describe('stream event handling', () => {
    it('yields correct event sequence for simple text response', async () => {
      const mockClient = createMockClient(async function* () {
        yield { type: 'message_start', message: { role: 'assistant', usage: { input_tokens: 10 } } }
        yield { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }
        yield { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello' } }
        yield { type: 'content_block_stop', index: 0 }
        yield { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 5 } }
        yield { type: 'message_stop' }
      })

      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      const events = await collectIterator(provider.stream(messages))

      expect(events).toHaveLength(6)
      expect(events[0]).toEqual({ type: 'modelMessageStartEvent', role: 'assistant' })
      expect(events[1]).toEqual({ type: 'modelContentBlockStartEvent' })
      expect(events[2]).toEqual({
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'textDelta', text: 'Hello' },
      })
      expect(events[3]).toEqual({ type: 'modelContentBlockStopEvent' })
      expect(events[4]).toEqual({
        type: 'modelMetadataEvent',
        usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
      })
      expect(events[5]).toEqual({ type: 'modelMessageStopEvent', stopReason: 'endTurn' })
    })

    it('handles tool use events', async () => {
      const mockClient = createMockClient(async function* () {
        yield { type: 'message_start', message: { role: 'assistant', usage: { input_tokens: 10 } } }
        yield {
          type: 'content_block_start',
          index: 0,
          content_block: { type: 'tool_use', id: 'tool_1', name: 'calc' },
        }
        yield { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"a"' } }
        yield { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: ':1}' } }
        yield { type: 'content_block_stop', index: 0 }
        yield { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 10 } }
        yield { type: 'message_stop' }
      })

      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      const events = await collectIterator(provider.stream(messages))

      expect(events).toContainEqual({
        type: 'modelContentBlockStartEvent',
        start: { type: 'toolUseStart', name: 'calc', toolUseId: 'tool_1' },
      })
      expect(events).toContainEqual({
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'toolUseInputDelta', input: '{"a"' },
      })
      expect(events).toContainEqual({
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'toolUseInputDelta', input: ':1}' },
      })
      expect(events).toContainEqual({ type: 'modelMessageStopEvent', stopReason: 'toolUse' })
    })

    it.each([
      ['pause_turn', 'pauseTurn'],
      ['refusal', 'refusal'],
    ])('maps anthropic stop reason "%s" to "%s"', async (anthropicReason, expected) => {
      const mockClient = createMockClient(async function* () {
        yield { type: 'message_start', message: { role: 'assistant', usage: { input_tokens: 1 } } }
        yield { type: 'message_delta', delta: { stop_reason: anthropicReason }, usage: { output_tokens: 1 } }
        yield { type: 'message_stop' }
      })

      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      const events = await collectIterator(provider.stream(messages))

      expect(events).toContainEqual({ type: 'modelMessageStopEvent', stopReason: expected })
    })

    it('handles thinking/reasoning events', async () => {
      const mockClient = createMockClient(async function* () {
        yield { type: 'message_start', message: { role: 'assistant', usage: { input_tokens: 10 } } }
        // Thinking block
        yield { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '' } }
        yield { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Hmm...' } }
        yield { type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'sig_123' } }
        yield { type: 'content_block_stop', index: 0 }
        // Text block
        yield { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '' } }
        yield { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Answer' } }
        yield { type: 'content_block_stop', index: 1 }

        yield { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 20 } }
        yield { type: 'message_stop' }
      })

      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      const events = await collectIterator(provider.stream(messages))

      // Check for thinking deltas
      expect(events).toContainEqual({
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'reasoningContentDelta', text: 'Hmm...' },
      })
      expect(events).toContainEqual({
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'reasoningContentDelta', signature: 'sig_123' },
      })
    })

    it('handles redacted thinking events', async () => {
      const mockClient = createMockClient(async function* () {
        yield { type: 'message_start', message: { role: 'assistant', usage: { input_tokens: 10 } } }
        yield {
          type: 'content_block_start',
          index: 0,
          content_block: { type: 'redacted_thinking', data: 'data' },
        }
        yield { type: 'content_block_stop', index: 0 }
        yield { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 5 } }
        yield { type: 'message_stop' }
      })

      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      const events = await collectIterator(provider.stream(messages))

      expect(events).toContainEqual({
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'reasoningContentDelta', redactedContent: 'data' },
      })
    })

    it('handles text payload directly in content_block_start (optimization)', async () => {
      const mockClient = createMockClient(async function* () {
        yield { type: 'message_start', message: { role: 'assistant', usage: { input_tokens: 10 } } }
        yield { type: 'content_block_start', index: 0, content_block: { type: 'text', text: 'Full text' } }
        yield { type: 'content_block_stop', index: 0 }
        yield { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 5 } }
        yield { type: 'message_stop' }
      })

      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      const events = await collectIterator(provider.stream(messages))

      expect(events).toContainEqual({
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'textDelta', text: 'Full text' },
      })
    })

    it('handles error during stream', async () => {
      const mockClient = createMockClient(async function* () {
        yield { type: 'ping' } // Satisfy linter require-yield
        throw new Error('API Error')
      })
      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await expect(collectIterator(provider.stream(messages))).rejects.toThrow('API Error')
    })

    it('maps overload error to ContextWindowOverflowError', async () => {
      const mockClient = createMockClient(async function* () {
        yield { type: 'ping' } // Satisfy linter require-yield
        throw new Error('prompt is too long')
      })
      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await expect(collectIterator(provider.stream(messages))).rejects.toThrow(ContextWindowOverflowError)
    })

    it.each([
      'input is too long',
      'input length exceeds context window',
      'input and output tokens exceed your context limit',
    ])('maps overflow phrase %p to ContextWindowOverflowError', async (phrase) => {
      const mockClient = createMockClient(async function* () {
        yield { type: 'ping' }
        throw new Error(phrase)
      })
      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await expect(collectIterator(provider.stream(messages))).rejects.toThrow(ContextWindowOverflowError)
    })

    it('matches overflow phrases case-insensitively', async () => {
      const mockClient = createMockClient(async function* () {
        yield { type: 'ping' }
        throw new Error('PROMPT IS TOO LONG: 200000 tokens')
      })
      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await expect(collectIterator(provider.stream(messages))).rejects.toThrow(ContextWindowOverflowError)
    })

    it('maps HTTP 429 error to ModelThrottledError', async () => {
      const rateLimitError = Object.assign(new Error('Rate limit exceeded'), { status: 429 })
      // eslint-disable-next-line require-yield
      const mockClient = createMockClient(async function* () {
        throw rateLimitError
      })
      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

      await expect(collectIterator(provider.stream(messages))).rejects.toThrow(ModelThrottledError)
      await expect(collectIterator(provider.stream(messages))).rejects.toThrow('Rate limit exceeded')
    })
  })

  describe('request formatting', () => {
    // Helper to capture request arguments
    const setupCapture = () => {
      const captured: { request: any; options: any } = { request: null, options: null }
      const mockClient = {
        messages: {
          stream: vi.fn((req, opts) => {
            captured.request = req
            captured.options = opts
            return (async function* () {})()
          }),
        },
      } as any
      return { captured, mockClient }
    }

    it('formats basic request correctly', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({
        modelId: 'claude-3-opus',
        maxTokens: 1000,
        temperature: 0.7,
        client: mockClient,
      })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hello')] })]

      await collectIterator(provider.stream(messages))

      expect(captured.request).toEqual({
        model: 'claude-3-opus',
        max_tokens: 1000,
        temperature: 0.7,
        messages: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
        stream: true,
      })
    })

    it('never sends the durable message id to the provider', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ modelId: 'claude-3-opus', maxTokens: 1000, client: mockClient })
      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('Hello')],
          trackingId: 'durable-1',
          metadata: { usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } },
        }),
      ]

      await collectIterator(provider.stream(messages))

      // The request the provider receives carries only role and content — never trackingId or metadata.
      expect(captured.request.messages).toEqual([{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }])
      expect('trackingId' in captured.request.messages[0]).toBe(false)
    })

    it('formats tools correctly', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]
      const toolSpecs = [
        {
          name: 'calc',
          description: 'calculate',
          inputSchema: { type: 'object' as const, properties: {} },
        },
      ]

      await collectIterator(provider.stream(messages, { toolSpecs, toolChoice: { auto: {} } }))

      expect(captured.request.tools).toHaveLength(1)
      expect(captured.request.tools[0]).toEqual({
        name: 'calc',
        description: 'calculate',
        input_schema: { type: 'object', properties: {} },
      })
      expect(captured.request.tool_choice).toEqual({ type: 'auto' })
    })

    it('formats a signature-only reasoning block', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'assistant',
          content: [new ReasoningBlock({ signature: 'sig-abc' }), new TextBlock('answer')],
        }),
      ]

      await collectIterator(provider.stream(messages))

      expect(captured.request.messages[0].content).toEqual([
        { type: 'thinking', thinking: '', signature: 'sig-abc' },
        { type: 'text', text: 'answer' },
      ])
    })

    describe('Prompt Caching (Lookahead logic)', () => {
      it('attaches cache control to message content block followed by cache point', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const messages = [
          new Message({
            role: 'user',
            content: [
              new TextBlock('Cached content'),
              // Use 'default' here; provider converts it to 'ephemeral' for Anthropic
              new CachePointBlock({ cacheType: 'default' }),
              new TextBlock('Non-cached content'),
            ],
          }),
        ]

        await collectIterator(provider.stream(messages))

        const content = captured.request.messages[0].content
        expect(content).toHaveLength(2) // 3 blocks reduced to 2 (cache point merged)
        expect(content[0]).toEqual({
          type: 'text',
          text: 'Cached content',
          cache_control: { type: 'ephemeral' },
        })
        expect(content[1]).toEqual({
          type: 'text',
          text: 'Non-cached content',
        })
      })

      it('formats system prompt string without cache', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        await collectIterator(provider.stream(messages, { systemPrompt: 'System instruction' }))

        expect(captured.request.system).toBe('System instruction')
      })

      it('formats system prompt array with cache points', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]
        const systemPrompt = [
          new TextBlock('Heavy context'),
          new CachePointBlock({ cacheType: 'default' }),
          new TextBlock('Light context'),
        ]

        await collectIterator(provider.stream(messages, { systemPrompt }))

        expect(Array.isArray(captured.request.system)).toBe(true)
        const system = captured.request.system
        expect(system).toHaveLength(2)
        expect(system[0]).toEqual({
          type: 'text',
          text: 'Heavy context',
          cache_control: { type: 'ephemeral' },
        })
        expect(system[1]).toEqual({
          type: 'text',
          text: 'Light context',
        })
      })
    })

    describe('Media blocks', () => {
      it('formats images correctly', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const imageBytes = new Uint8Array([72, 101, 108, 108, 111]) // "Hello"
        const messages = [
          new Message({
            role: 'user',
            content: [
              new ImageBlock({
                format: 'png',
                source: { bytes: imageBytes },
              }),
            ],
          }),
        ]

        await collectIterator(provider.stream(messages))

        const content = captured.request.messages[0].content[0]
        expect(content.type).toBe('image')
        expect(content.source.media_type).toBe('image/png')
        // Base64 of "Hello" is "SGVsbG8="
        expect(content.source.data).toBe('SGVsbG8=')
      })

      it('formats PDFs correctly', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const pdfBytes = new Uint8Array([1, 2, 3])
        const messages = [
          new Message({
            role: 'user',
            content: [
              new DocumentBlock({
                name: 'doc.pdf',
                format: 'pdf',
                source: { bytes: pdfBytes },
              }),
            ],
          }),
        ]

        await collectIterator(provider.stream(messages))

        const content = captured.request.messages[0].content[0]
        expect(content.type).toBe('document')
        expect(content.source.media_type).toBe('application/pdf')
        expect(content.title).toBe('doc.pdf')
      })

      it('logs warning for unsupported GuardContentBlock in user message', async () => {
        const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {}) // Spy on console.warn (via logger)
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const messages = [
          new Message({
            role: 'user',
            content: [
              new GuardContentBlock({
                text: { text: 'guard', qualifiers: ['query'] },
              }),
            ],
          }),
        ]

        await collectIterator(provider.stream(messages))

        // Should result in empty content if blocked
        expect(captured.request.messages[0].content).toHaveLength(0)
        warnSpy.mockRestore()
      })
    })

    describe('Tool Results', () => {
      it('formats simple text tool result', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const messages = [
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 't1',
                status: 'success',
                content: [new TextBlock('42')],
              }),
            ],
          }),
        ]

        await collectIterator(provider.stream(messages))

        const content = captured.request.messages[0].content[0]
        expect(content.type).toBe('tool_result')
        expect(content.tool_use_id).toBe('t1')
        expect(content.content).toBe('42') // Simplified to string
        expect(content.is_error).toBe(false)
      })

      it('formats mixed tool result (json/image)', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const messages = [
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 't1',
                status: 'error',
                content: [new JsonBlock({ json: { error: 'failed' } }), new TextBlock('Details here')],
              }),
            ],
          }),
        ]

        await collectIterator(provider.stream(messages))

        const content = captured.request.messages[0].content[0]
        expect(content.type).toBe('tool_result')
        expect(content.is_error).toBe(true)
        expect(Array.isArray(content.content)).toBe(true)
        // JSON is stringified in Anthropic tool result content
        expect(content.content[0]).toEqual({ type: 'text', text: '{"error":"failed"}' })
        expect(content.content[1]).toEqual({ type: 'text', text: 'Details here' })
      })

      it('formats image block inside tool result via recursive formatting', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const imageBytes = new Uint8Array([72, 101, 108, 108, 111])
        const messages = [
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 't1',
                status: 'success',
                content: [
                  new TextBlock('Here is the screenshot'),
                  new ImageBlock({ format: 'png', source: { bytes: imageBytes } }),
                ],
              }),
            ],
          }),
        ]

        await collectIterator(provider.stream(messages))

        const content = captured.request.messages[0].content[0]
        expect(content.type).toBe('tool_result')
        expect(Array.isArray(content.content)).toBe(true)
        expect(content.content[0]).toEqual({ type: 'text', text: 'Here is the screenshot' })
        expect(content.content[1]).toEqual({
          type: 'image',
          source: { type: 'base64', media_type: 'image/png', data: 'SGVsbG8=' },
        })
      })

      it('formats document block inside tool result as text for text formats', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const messages = [
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 't1',
                status: 'success',
                content: [new DocumentBlock({ name: 'data.json', format: 'json', source: { text: '{"key":"val"}' } })],
              }),
            ],
          }),
        ]

        await collectIterator(provider.stream(messages))

        const content = captured.request.messages[0].content[0]
        expect(content.type).toBe('tool_result')
        // Single text item collapses to string
        expect(content.content).toBe('{"key":"val"}')
      })

      it('skips video block inside tool result with warning', async () => {
        const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const messages = [
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 't1',
                status: 'success',
                content: [
                  new TextBlock('result'),
                  new VideoBlock({ format: 'mp4', source: { bytes: new Uint8Array([1]) } }),
                ],
              }),
            ],
          }),
        ]

        await collectIterator(provider.stream(messages))

        const content = captured.request.messages[0].content[0]
        expect(content.type).toBe('tool_result')
        // Video is filtered out, single text collapses to string
        expect(content.content).toBe('result')
        expect(warnSpy).toHaveBeenCalled()
        warnSpy.mockRestore()
      })
    })

    describe('Beta headers', () => {
      it('does not pass per-request options when betas is unset', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        await collectIterator(provider.stream(messages))

        expect(captured.options).toBeUndefined()
      })

      it('forwards configured betas as a per-request anthropic-beta header', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({
          client: mockClient,
          betas: ['interleaved-thinking-2025-05-14', 'mcp-client-2025-11-20'],
        })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        await collectIterator(provider.stream(messages))

        expect(captured.options).toEqual({
          headers: { 'anthropic-beta': 'interleaved-thinking-2025-05-14,mcp-client-2025-11-20' },
        })
      })

      it('reflects updateConfig({ betas }) on the next request', async () => {
        const { captured, mockClient } = setupCapture()
        const provider = new AnthropicModel({ client: mockClient })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hi')] })]

        await collectIterator(provider.stream(messages))
        expect(captured.options).toBeUndefined()

        provider.updateConfig({ betas: ['interleaved-thinking-2025-05-14'] })
        await collectIterator(provider.stream(messages))

        expect(captured.options).toEqual({
          headers: { 'anthropic-beta': 'interleaved-thinking-2025-05-14' },
        })
      })
    })
  })

  describe('countTokens', () => {
    const messages: Message[] = [new Message({ role: 'user', content: [new TextBlock('hello')] })]
    const toolSpecs = [
      { name: 'test_tool', description: 'A test tool', inputSchema: { type: 'object' as const, properties: {} } },
    ]

    function createCountTokensClient(mockCountTokens: ReturnType<typeof vi.fn>): Anthropic {
      return {
        messages: {
          stream: vi.fn(),
          countTokens: mockCountTokens,
        },
      } as unknown as Anthropic
    }

    it('should use heuristic by default when useNativeTokenCount is not set', async () => {
      const mockCountTokens = vi.fn()
      const client = createCountTokensClient(mockCountTokens)
      const model = new AnthropicModel({ client, modelId: 'claude-sonnet-4-6' })

      const result = await model.countTokens(messages)

      expect(mockCountTokens).not.toHaveBeenCalled()
      expect(result).toBe(2) // heuristic: Math.ceil('hello'.length / 4)
    })

    it('should return native token count on success', async () => {
      const mockCountTokens = vi.fn(async () => ({ input_tokens: 42 }))
      const client = createCountTokensClient(mockCountTokens)
      const model = new AnthropicModel({ client, modelId: 'claude-sonnet-4-6', useNativeTokenCount: true })

      const result = await model.countTokens(messages)

      expect(result).toBe(42)
      expect(mockCountTokens).toHaveBeenCalledOnce()
    })

    it('should include system prompt in request', async () => {
      const mockCountTokens = vi.fn(async () => ({ input_tokens: 55 }))
      const client = createCountTokensClient(mockCountTokens)
      const model = new AnthropicModel({ client, modelId: 'claude-sonnet-4-6', useNativeTokenCount: true })

      const result = await model.countTokens(messages, { systemPrompt: 'Be helpful.' })

      expect(result).toBe(55)
      expect(mockCountTokens).toHaveBeenCalledWith({
        model: 'claude-sonnet-4-6',
        messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
        system: 'Be helpful.',
      })
    })

    it('should include tool specs in request', async () => {
      const mockCountTokens = vi.fn(async () => ({ input_tokens: 100 }))
      const client = createCountTokensClient(mockCountTokens)
      const model = new AnthropicModel({ client, modelId: 'claude-sonnet-4-6', useNativeTokenCount: true })

      const result = await model.countTokens(messages, { toolSpecs })

      expect(result).toBe(100)
      expect(mockCountTokens).toHaveBeenCalledWith({
        model: 'claude-sonnet-4-6',
        messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
        tools: [{ name: 'test_tool', description: 'A test tool', input_schema: { type: 'object', properties: {} } }],
      })
    })

    it('should strip max_tokens from request', async () => {
      const mockCountTokens = vi.fn(async () => ({ input_tokens: 10 }))
      const client = createCountTokensClient(mockCountTokens)
      const model = new AnthropicModel({ client, modelId: 'claude-sonnet-4-6', useNativeTokenCount: true })

      await model.countTokens(messages)

      expect(mockCountTokens).toHaveBeenCalledWith({
        model: 'claude-sonnet-4-6',
        messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
      })
    })

    it('should fall back to estimation on API error', async () => {
      const mockCountTokens = vi.fn(async () => {
        throw new Error('Unsupported')
      })
      const client = createCountTokensClient(mockCountTokens)
      const model = new AnthropicModel({ client, modelId: 'claude-sonnet-4-6', useNativeTokenCount: true })

      const result = await model.countTokens(messages)

      expect(typeof result).toBe('number')
      expect(result).toBeGreaterThanOrEqual(0)
    })

    it('should fall back to estimation on generic exception', async () => {
      const mockCountTokens = vi.fn(async () => {
        throw new Error('Connection failed')
      })
      const client = createCountTokensClient(mockCountTokens)
      const model = new AnthropicModel({ client, modelId: 'claude-sonnet-4-6', useNativeTokenCount: true })

      const result = await model.countTokens(messages)

      expect(typeof result).toBe('number')
      expect(result).toBeGreaterThanOrEqual(0)
    })

    it('should skip native API and use heuristic when useNativeTokenCount is false', async () => {
      const mockCountTokens = vi.fn()
      const client = createCountTokensClient(mockCountTokens)
      const model = new AnthropicModel({ client, modelId: 'claude-sonnet-4-6', useNativeTokenCount: false })

      const result = await model.countTokens(messages)

      expect(mockCountTokens).not.toHaveBeenCalled()
      expect(result).toBe(2) // heuristic: Math.ceil('hello'.length / 4)
    })
  })

  /**
   * Prompt caching via `cacheConfig` / `cacheTools`.
   *
   * Anthropic accepts at most 4 cache breakpoints per request and `ephemeral` is the only cache type.
   * @see https://docs.claude.com/en/docs/build-with-claude/prompt-caching
   */
  describe('prompt caching', () => {
    const MAX_BREAKPOINTS = 4

    const setupCapture = (): { captured: { request: any }; mockClient: Anthropic } => {
      const captured: { request: any } = { request: null }
      const mockClient = {
        messages: {
          stream: vi.fn((req) => {
            captured.request = req
            return (async function* () {})()
          }),
        },
      } as any
      return { captured, mockClient }
    }

    /** Every cache_control in a formatted request, across tools, system and messages. */
    const breakpoints = (request: any): unknown[] => {
      const found: unknown[] = []
      for (const tool of request.tools ?? []) {
        if (tool.cache_control) found.push(['tools', tool.name, tool.cache_control])
      }
      if (Array.isArray(request.system)) {
        for (const block of request.system) {
          if (block.cache_control) found.push(['system', block.type, block.cache_control])
        }
      }
      request.messages.forEach((message: any, messageIndex: number) => {
        for (const block of message.content) {
          if (block.cache_control) found.push(['messages', messageIndex, block.cache_control])
        }
      })
      return found
    }

    const toolSpecs = [
      { name: 't1', description: 'tool one', inputSchema: { type: 'object' as const } },
      { name: 't2', description: 'tool two', inputSchema: { type: 'object' as const } },
    ]

    const userMessage = (text: string): Message => new Message({ role: 'user', content: [new TextBlock(text)] })

    it('is off when unset', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient })

      await collectIterator(provider.stream([userMessage('Hi')], { toolSpecs }))

      expect(breakpoints(captured.request)).toEqual([])
    })

    it('adds a breakpoint to the last user message', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheConfig: { strategy: 'auto' } })

      await collectIterator(provider.stream([userMessage('one'), userMessage('two')]))

      expect(breakpoints(captured.request)).toEqual([['messages', 1, { type: 'ephemeral' }]])
    })

    it('treats auto and anthropic strategies the same', async () => {
      // The Anthropic API caches on every active Claude model, so auto has no model-support check to
      // apply and the two strategies produce the same request.
      const auto = setupCapture()
      await collectIterator(
        new AnthropicModel({ client: auto.mockClient, cacheConfig: { strategy: 'auto' } }).stream([userMessage('Hi')], {
          toolSpecs,
        })
      )

      const explicit = setupCapture()
      await collectIterator(
        new AnthropicModel({ client: explicit.mockClient, cacheConfig: { strategy: 'anthropic' } }).stream(
          [userMessage('Hi')],
          { toolSpecs }
        )
      )

      expect(auto.captured.request).toEqual(explicit.captured.request)
    })

    it('carries cacheConfig ttl onto cache_control', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheConfig: { strategy: 'auto', ttl: '1h' } })

      await collectIterator(provider.stream([userMessage('Hi')]))

      expect(breakpoints(captured.request)).toEqual([['messages', 0, { type: 'ephemeral', ttl: '1h' }]])
    })

    it('caches only the last tool definition', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheTools: 'default' })

      await collectIterator(provider.stream([userMessage('Hi')], { toolSpecs }))

      expect(breakpoints(captured.request)).toEqual([['tools', 't2', { type: 'ephemeral' }]])
      expect(captured.request.tools[0].cache_control).toBeUndefined()
    })

    it('carries cacheTools ttl onto cache_control', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheTools: { ttl: '1h' } })

      await collectIterator(provider.stream([userMessage('Hi')], { toolSpecs }))

      expect(breakpoints(captured.request)).toEqual([['tools', 't2', { type: 'ephemeral', ttl: '1h' }]])
    })

    it('normalizes a Bedrock cache point type to ephemeral', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('hi'), new CachePointBlock({ cacheType: 'default' })],
        }),
      ]

      await collectIterator(provider.stream(messages))

      expect(breakpoints(captured.request)).toEqual([['messages', 0, { type: 'ephemeral' }]])
    })

    it('is a no-op when cacheTools is set without tools', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheTools: 'default' })

      await collectIterator(provider.stream([userMessage('Hi')]))

      expect(captured.request.tools).toBeUndefined()
      expect(breakpoints(captured.request)).toEqual([])
    })

    it('applies cacheTools independently of cacheConfig', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheTools: 'default' })

      await collectIterator(provider.stream([userMessage('Hi')], { toolSpecs }))

      expect(breakpoints(captured.request)).toEqual([['tools', 't2', { type: 'ephemeral' }]])
    })

    it('produces two breakpoints when both options are set', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({
        client: mockClient,
        cacheConfig: { strategy: 'auto' },
        cacheTools: 'default',
      })

      await collectIterator(provider.stream([userMessage('Hi')], { toolSpecs }))

      const found = breakpoints(captured.request)
      expect(found).toEqual([
        ['tools', 't2', { type: 'ephemeral' }],
        ['messages', 0, { type: 'ephemeral' }],
      ])
      expect(found.length).toBeLessThanOrEqual(MAX_BREAKPOINTS)
    })

    it('does not accumulate breakpoints across turns', async () => {
      // A cache point per turn would blow the 4-breakpoint limit on the 5th turn. This history already
      // carries one per turn, as it would if each turn appended one; exactly one must survive.
      const messages: Message[] = []
      for (let turn = 0; turn < 25; turn++) {
        messages.push(
          new Message({
            role: 'user',
            content: [new TextBlock(`question ${turn}`), new CachePointBlock({ cacheType: 'default' })],
          })
        )
        messages.push(
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: `t${turn}`,
                status: 'success',
                content: [new TextBlock(`result ${turn}`)],
              }),
              new CachePointBlock({ cacheType: 'default' }),
            ],
          })
        )
      }

      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({
        client: mockClient,
        cacheConfig: { strategy: 'auto' },
        cacheTools: 'default',
      })

      await collectIterator(provider.stream(messages, { toolSpecs }))

      const found = breakpoints(captured.request)
      expect(found.length).toBeLessThanOrEqual(MAX_BREAKPOINTS)
      const messageBreakpoints = found.filter((bp) => (bp as unknown[])[0] === 'messages')
      expect(messageBreakpoints).toEqual([['messages', messages.length - 1, { type: 'ephemeral' }]])
    })

    it('replaces hand-placed message cache points', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheConfig: { strategy: 'auto' } })
      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('one'), new CachePointBlock({ cacheType: 'default' })],
        }),
        new Message({ role: 'assistant', content: [new TextBlock('two')] }),
        userMessage('three'),
      ]

      await collectIterator(provider.stream(messages))

      expect(breakpoints(captured.request)).toEqual([['messages', 2, { type: 'ephemeral' }]])
      // The stripped cache point leaves the text block behind untouched.
      expect(captured.request.messages[0].content).toEqual([{ type: 'text', text: 'one' }])
    })

    it('places the breakpoint after the last cacheable block', async () => {
      // Anthropic rejects cache_control on a reasoning block, so the cache point skips back past it.
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheConfig: { strategy: 'auto' } })
      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('question'), new ReasoningBlock({ text: 'thinking', signature: 'sig' })],
        }),
      ]

      await collectIterator(provider.stream(messages))

      expect(breakpoints(captured.request)).toEqual([['messages', 0, { type: 'ephemeral' }]])
      expect(captured.request.messages[0].content[0]).toEqual({
        type: 'text',
        text: 'question',
        cache_control: { type: 'ephemeral' },
      })
      expect(captured.request.messages[0].content[1].cache_control).toBeUndefined()
    })

    it('skips a history with no cacheable user content', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheConfig: { strategy: 'auto' } })
      const messages = [new Message({ role: 'assistant', content: [new TextBlock('hello')] })]

      await collectIterator(provider.stream(messages))

      expect(breakpoints(captured.request)).toEqual([])
    })

    it('strips hand-placed cache points even when there is nothing to cache', async () => {
      // cacheConfig owns message breakpoints whenever it is set, not only when it found a block to
      // mark. strands-py strips unconditionally, so failing to do so here diverges the two SDKs.
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheConfig: { strategy: 'auto' } })
      const messages = [
        new Message({
          role: 'assistant',
          content: [new TextBlock('hello'), new CachePointBlock({ cacheType: 'default' })],
        }),
      ]

      await collectIterator(provider.stream(messages))

      expect(breakpoints(captured.request)).toEqual([])
    })

    it('honors the ttl on a hand-placed cache point', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('one'), new CachePointBlock({ cacheType: 'default', ttl: '1h' })],
        }),
      ]

      await collectIterator(provider.stream(messages))

      expect(breakpoints(captured.request)).toEqual([['messages', 0, { type: 'ephemeral', ttl: '1h' }]])
    })

    it('honors the ttl on a system prompt cache point', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient })
      const systemPrompt = [new TextBlock('Heavy context'), new CachePointBlock({ cacheType: 'default', ttl: '1h' })]

      await collectIterator(provider.stream([userMessage('Hi')], { systemPrompt }))

      expect(captured.request.system[0].cache_control).toEqual({ type: 'ephemeral', ttl: '1h' })
    })

    it('keeps the breakpoint when the last cacheable block is dropped in translation', async () => {
      // An image with a location source is an accepted cache carrier by block type but is dropped when
      // the request is formatted. Choosing the target before formatting used to leave the request with
      // no cache_control at all, so enabling caching removed the caching that worked without it.
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheConfig: { strategy: 'auto' } })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new TextBlock('long prefix'),
            new ImageBlock({ format: 'png', source: { url: 'https://example.com/a.png' } as never }),
          ],
        }),
      ]

      await collectIterator(provider.stream(messages))

      expect(breakpoints(captured.request)).toEqual([['messages', 0, { type: 'ephemeral' }]])
      expect(captured.request.messages[0].content[0]).toEqual({
        type: 'text',
        text: 'long prefix',
        cache_control: { type: 'ephemeral' },
      })
    })

    it('never places a breakpoint on a reasoning block', async () => {
      // The only formattable block left is a thinking block, which the API rejects a breakpoint on.
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheConfig: { strategy: 'auto' } })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ReasoningBlock({ text: 'thinking', signature: 'sig' }),
            new ImageBlock({ format: 'png', source: { url: 'https://example.com/a.png' } as never }),
          ],
        }),
      ]

      await collectIterator(provider.stream(messages))

      expect(breakpoints(captured.request)).toEqual([])
    })

    it('does not fall back to an earlier turn when the newest one has nothing cacheable', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheConfig: { strategy: 'auto' } })
      const messages = [
        userMessage('turn one'),
        new Message({ role: 'assistant', content: [new TextBlock('reply')] }),
        new Message({ role: 'user', content: [new ReasoningBlock({ text: 'thinking', signature: 'sig' })] }),
      ]

      await collectIterator(provider.stream(messages))

      // Caching a prefix that has stopped growing would pin every later turn to a stale entry.
      expect(breakpoints(captured.request)).toEqual([])
    })

    it('places the breakpoint on the last of several cacheable blocks', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheConfig: { strategy: 'auto' } })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new TextBlock('first'),
            new ToolResultBlock({ toolUseId: 'x', content: [new TextBlock('second')], status: 'success' }),
          ],
        }),
      ]

      await collectIterator(provider.stream(messages))

      expect(breakpoints(captured.request)).toEqual([['messages', 0, { type: 'ephemeral' }]])
      expect(captured.request.messages[0].content[0].cache_control).toBeUndefined()
      expect(captured.request.messages[0].content[1].type).toBe('tool_result')
      expect(captured.request.messages[0].content[1].cache_control).toEqual({ type: 'ephemeral' })
    })

    it('treats an empty cacheTools value as unset', async () => {
      // cacheTools: '' is what an unset environment variable produces. A presence check enabled caching
      // here while Python's truthiness check did not.
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheTools: '' })

      await collectIterator(provider.stream([userMessage('Hi')], { toolSpecs }))

      expect(breakpoints(captured.request)).toEqual([])
    })

    it('disables caching for an unknown strategy', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({
        client: mockClient,
        cacheConfig: { strategy: 'sometimes' as never },
      })

      await collectIterator(provider.stream([userMessage('Hi')]))

      expect(breakpoints(captured.request)).toEqual([])
    })

    it('warns when hand-placed points and cacheTools exceed the breakpoint ceiling', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheTools: 'default' })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new TextBlock('a'),
            new CachePointBlock({ cacheType: 'default' }),
            new TextBlock('b'),
            new CachePointBlock({ cacheType: 'default' }),
            new TextBlock('c'),
            new CachePointBlock({ cacheType: 'default' }),
            new TextBlock('d'),
            new CachePointBlock({ cacheType: 'default' }),
          ],
        }),
      ]

      await collectIterator(provider.stream(messages, { toolSpecs }))

      // The caller's own points are left alone; an opaque 400 is harder to act on than a named cause.
      expect(breakpoints(captured.request)).toHaveLength(MAX_BREAKPOINTS + 1)
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('too many cache breakpoints'))
      warnSpy.mockRestore()
    })

    it("does not mutate the caller's messages", async () => {
      const { mockClient } = setupCapture()
      const provider = new AnthropicModel({
        client: mockClient,
        cacheConfig: { strategy: 'auto' },
        cacheTools: 'default',
      })
      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('hello'), new CachePointBlock({ cacheType: 'default' })],
        }),
      ]
      const before = JSON.stringify(messages)

      await collectIterator(provider.stream(messages, { toolSpecs }))

      expect(JSON.stringify(messages)).toBe(before)
    })

    it('attaches a hand-placed cache point to the nearest block that accepts one', async () => {
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new TextBlock('question'),
            new ReasoningBlock({ text: 'thinking', signature: 'sig' }),
            new CachePointBlock({ cacheType: 'default' }),
          ],
        }),
      ]

      await collectIterator(provider.stream(messages))

      expect(captured.request.messages[0].content[0].cache_control).toEqual({ type: 'ephemeral' })
      expect(captured.request.messages[0].content[1].cache_control).toBeUndefined()
    })

    it('pins the media request body shared with strands-py', async () => {
      // The mirror of this test is test_cross_sdk_media_request_parity in
      // strands-py/tests/strands/models/test_anthropic.py. Update both together or the SDKs will drift.
      // Text-only parity cannot catch a breakpoint lost while translating a media block, which is
      // exactly what regressed here.
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({ client: mockClient, cacheConfig: { strategy: 'auto' } })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new TextBlock('prefix'),
            new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1, 2, 3]) } }),
            // Dropped in translation: the breakpoint must fall back to the image above, not vanish.
            new ImageBlock({ format: 'png', source: { url: 'https://example.com/a.png' } as never }),
          ],
        }),
      ]

      await collectIterator(provider.stream(messages))

      expect(captured.request.messages).toEqual([
        {
          role: 'user',
          content: [
            { type: 'text', text: 'prefix' },
            {
              type: 'image',
              source: { type: 'base64', media_type: 'image/png', data: 'AQID' },
              cache_control: { type: 'ephemeral' },
            },
          ],
        },
      ])
    })

    it('pins the request body shared with strands-py', async () => {
      // The mirror of this test is test_cross_sdk_request_parity in
      // strands-py/tests/strands/models/test_anthropic.py. Update both together or the SDKs will drift.
      const { captured, mockClient } = setupCapture()
      const provider = new AnthropicModel({
        client: mockClient,
        cacheConfig: { strategy: 'auto', ttl: '1h' },
        cacheTools: { ttl: '1h' },
      })
      const messages = [
        userMessage('hello'),
        new Message({ role: 'assistant', content: [new TextBlock('hi')] }),
        userMessage('again'),
      ]

      await collectIterator(provider.stream(messages, { toolSpecs }))

      expect(captured.request.tools).toEqual([
        { name: 't1', description: 'tool one', input_schema: { type: 'object' } },
        {
          name: 't2',
          description: 'tool two',
          input_schema: { type: 'object' },
          cache_control: { type: 'ephemeral', ttl: '1h' },
        },
      ])
      expect(captured.request.messages).toEqual([
        { role: 'user', content: [{ type: 'text', text: 'hello' }] },
        { role: 'assistant', content: [{ type: 'text', text: 'hi' }] },
        {
          role: 'user',
          content: [{ type: 'text', text: 'again', cache_control: { type: 'ephemeral', ttl: '1h' } }],
        },
      ])
    })
  })
})
