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

    it.each([
      'PROMPT IS TOO LONG: request exceeds context window',
      'max_tokens exceeded',
      'input too long',
      'input is too long',
      'input length exceeds context window',
      'input and output tokens exceed your context limit',
    ])('maps context overflow error "%s" to ContextWindowOverflowError', async (message) => {
      const mockClient = createMockClient(async function* () {
        yield { type: 'ping' } // Satisfy linter require-yield
        throw new Error(message)
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
})
