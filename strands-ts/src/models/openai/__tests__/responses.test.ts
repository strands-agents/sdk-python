import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import OpenAI from 'openai'
import { isNode } from '../../../__fixtures__/environment.js'
import { OpenAIModel } from '../index.js'
import { ContextWindowOverflowError, ModelThrottledError } from '../../../errors.js'
import { collectIterator } from '../../../__fixtures__/model-test-helpers.js'
import { Message, TextBlock, ToolUseBlock, ToolResultBlock } from '../../../types/messages.js'
import { ImageBlock, DocumentBlock } from '../../../types/media.js'
import { StateStore } from '../../../state-store.js'
import { logger } from '../../../logging/logger.js'

/**
 * Build a mock OpenAI client whose `responses.create` returns the given async generator.
 * The last request passed to `create` is captured on `capture.request`.
 */
function createMockClient(streamGenerator: () => AsyncGenerator<any>, capture: { request?: any } = {}): OpenAI {
  return {
    responses: {
      create: vi.fn(async (request: any) => {
        capture.request = request
        return streamGenerator()
      }),
    },
  } as any
}

// Mock the OpenAI SDK
vi.mock('openai', () => {
  const mockConstructor = vi.fn(function (this: any) {
    return {}
  })
  return {
    default: mockConstructor,
  }
})

describe("OpenAIModel (api: 'responses')", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.restoreAllMocks()
    if (isNode) {
      vi.stubEnv('OPENAI_API_KEY', 'sk-test-env')
    }
  })

  afterEach(() => {
    vi.clearAllMocks()
    if (isNode) {
      vi.unstubAllEnvs()
    }
  })

  describe('constructor', () => {
    it('uses API key from constructor parameter', () => {
      new OpenAIModel({ api: 'responses', modelId: 'gpt-4o', apiKey: 'sk-explicit' })
      expect(OpenAI).toHaveBeenCalledWith(expect.objectContaining({ apiKey: 'sk-explicit' }))
    })

    if (isNode) {
      it('uses API key from environment variable', () => {
        vi.stubEnv('OPENAI_API_KEY', 'sk-from-env')
        new OpenAIModel({ api: 'responses', modelId: 'gpt-4o' })
        expect(OpenAI).toHaveBeenCalled()
      })
    }

    it('throws error when no API key is available', () => {
      if (isNode) {
        vi.stubEnv('OPENAI_API_KEY', '')
      }
      expect(() => new OpenAIModel({ api: 'responses', modelId: 'gpt-4o' })).toThrow(/OpenAI API key is required/)
    })

    it('uses provided client instance and skips OpenAI constructor', () => {
      vi.clearAllMocks()
      const client = {} as OpenAI
      const model = new OpenAIModel({ api: 'responses', client })
      expect(OpenAI).not.toHaveBeenCalled()
      expect(model).toBeDefined()
    })

    it('does not require API key when client is provided', () => {
      if (isNode) {
        vi.stubEnv('OPENAI_API_KEY', '')
      }
      const client = {} as OpenAI
      expect(() => new OpenAIModel({ api: 'responses', client })).not.toThrow()
    })
  })

  describe('stateful', () => {
    it('defaults to false', () => {
      const model = new OpenAIModel({ api: 'responses', client: {} as OpenAI })
      expect(model.stateful).toBe(false)
    })

    it('returns true when explicitly enabled', () => {
      const model = new OpenAIModel({ api: 'responses', client: {} as OpenAI, stateful: true })
      expect(model.stateful).toBe(true)
    })

    it('is construction-only and cannot be changed via updateConfig', () => {
      const model = new OpenAIModel({ api: 'responses', client: {} as OpenAI, stateful: false })
      const warnSpy = vi.spyOn(logger, 'warn')
      expect(model.stateful).toBe(false)
      model.updateConfig({ stateful: true })
      expect(model.stateful).toBe(false)
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("'stateful' is construction-only"))
      warnSpy.mockRestore()
    })
  })

  describe('updateConfig / getConfig', () => {
    it('merges config without clobbering unspecified fields', () => {
      const model = new OpenAIModel({
        api: 'responses',
        client: {} as OpenAI,
        modelId: 'gpt-4o',
        temperature: 0.5,
        maxTokens: 1024,
      })
      model.updateConfig({ temperature: 0.9 })
      expect(model.getConfig()).toMatchObject({
        modelId: 'gpt-4o',
        temperature: 0.9,
        maxTokens: 1024,
      })
    })
  })

  describe('managed params warning', () => {
    it('warns on construction when params contains provider-managed keys', () => {
      const warnSpy = vi.spyOn(logger, 'warn')
      new OpenAIModel({ api: 'responses', client: {} as OpenAI, params: { model: 'bad', store: false } })
      expect(warnSpy).toHaveBeenCalledTimes(2)
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("'model'"))
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("'store'"))
      warnSpy.mockRestore()
    })

    it('warns on updateConfig when params contains provider-managed keys', () => {
      const model = new OpenAIModel({ api: 'responses', client: {} as OpenAI })
      const warnSpy = vi.spyOn(logger, 'warn')
      model.updateConfig({ params: { stream: true } })
      expect(warnSpy).toHaveBeenCalledTimes(1)
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("'stream'"))
      warnSpy.mockRestore()
    })

    it('does not warn when params contains only non-managed keys', () => {
      const warnSpy = vi.spyOn(logger, 'warn')
      new OpenAIModel({ api: 'responses', client: {} as OpenAI, params: { reasoning: { summary: 'auto' } } })
      expect(warnSpy).not.toHaveBeenCalled()
      warnSpy.mockRestore()
    })
  })

  describe('request formatting', () => {
    const mkUserMessage = () => new Message({ role: 'user', content: [new TextBlock('Hi')] })

    async function runOnce(
      modelOptions: Omit<Extract<ConstructorParameters<typeof OpenAIModel>[0], { api?: 'responses' }>, 'api'> = {},
      messages = [mkUserMessage()],
      streamOptions: Parameters<OpenAIModel['stream']>[1] = undefined
    ): Promise<any> {
      const capture: { request?: any } = {}
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'resp_123' } }
        yield { type: 'response.completed', response: { usage: undefined } }
      }, capture)
      const model = new OpenAIModel({ api: 'responses', client, ...modelOptions })
      await collectIterator(model.stream(messages, streamOptions))
      return capture.request
    }

    it('includes model, input, stream, and store=false by default', async () => {
      const req = await runOnce()
      expect(req.model).toBe('gpt-5.4')
      expect(req.stream).toBe(true)
      expect(req.store).toBe(false)
      expect(Array.isArray(req.input)).toBe(true)
    })

    it('sets store=true when stateful is enabled', async () => {
      const req = await runOnce({ stateful: true })
      expect(req.store).toBe(true)
    })

    it('chains previous_response_id when stateful and modelState has responseId', async () => {
      const modelState = new StateStore({ responseId: 'resp_prev' })
      const req = await runOnce({ stateful: true }, [mkUserMessage()], { modelState })
      expect(req.previous_response_id).toBe('resp_prev')
    })

    it('omits previous_response_id when stateful is disabled, even with responseId in modelState', async () => {
      const modelState = new StateStore({ responseId: 'resp_prev' })
      const req = await runOnce({}, [mkUserMessage()], { modelState })
      expect(req.previous_response_id).toBeUndefined()
    })

    it('maps systemPrompt string to instructions', async () => {
      const req = await runOnce({}, [mkUserMessage()], { systemPrompt: 'Be helpful.' })
      expect(req.instructions).toBe('Be helpful.')
    })

    it('merges toolSpecs with built-in tools from params', async () => {
      const req = await runOnce({ params: { tools: [{ type: 'web_search' }] } }, [mkUserMessage()], {
        toolSpecs: [
          {
            name: 'calc',
            description: 'calculator',
            inputSchema: { type: 'object', properties: {} },
          },
        ],
      })
      expect(req.tools).toEqual([
        { type: 'web_search' },
        {
          type: 'function',
          name: 'calc',
          description: 'calculator',
          parameters: { type: 'object', properties: {} },
          strict: null,
        },
      ])
    })

    it('maps tool_choice variants', async () => {
      const toolSpecs = [{ name: 'calc', description: 'd', inputSchema: {} }]
      const autoReq = await runOnce({}, [mkUserMessage()], { toolSpecs, toolChoice: { auto: {} } })
      expect(autoReq.tool_choice).toBe('auto')

      const anyReq = await runOnce({}, [mkUserMessage()], { toolSpecs, toolChoice: { any: {} } })
      expect(anyReq.tool_choice).toBe('required')

      const toolReq = await runOnce({}, [mkUserMessage()], {
        toolSpecs,
        toolChoice: { tool: { name: 'calc' } },
      })
      expect(toolReq.tool_choice).toEqual({ type: 'function', name: 'calc' })
    })

    it('formats temperature, maxTokens→max_output_tokens, and topP', async () => {
      const req = await runOnce({ temperature: 0.3, maxTokens: 512, topP: 0.8 })
      expect(req.temperature).toBe(0.3)
      expect(req.max_output_tokens).toBe(512)
      expect(req.top_p).toBe(0.8)
    })

    it('passes through extra params fields to the request', async () => {
      const req = await runOnce({ params: { reasoning: { summary: 'auto' } } })
      expect(req.reasoning).toEqual({ summary: 'auto' })
    })

    it('provider-managed fields in params are overridden and cannot take effect', async () => {
      const warnSpy = vi.spyOn(logger, 'warn')
      const req = await runOnce({
        modelId: 'gpt-4o',
        stateful: true,
        params: { model: 'attacker-model', input: 'hijacked', stream: false, store: false },
      })
      expect(req.model).toBe('gpt-4o')
      expect(req.stream).toBe(true)
      expect(req.store).toBe(true)
      expect(Array.isArray(req.input)).toBe(true)
      warnSpy.mockRestore()
    })

    it('emits tool_use and tool_result as separate top-level items', async () => {
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('run it')] }),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ name: 'calc', toolUseId: 'call_1', input: { expr: '2+2' } })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'call_1',
              status: 'success',
              content: [new TextBlock('4')],
            }),
          ],
        }),
      ]
      const req = await runOnce({}, messages)
      const functionCall = req.input.find((i: any) => i.type === 'function_call')
      const functionOutput = req.input.find((i: any) => i.type === 'function_call_output')
      expect(functionCall).toMatchObject({
        type: 'function_call',
        call_id: 'call_1',
        name: 'calc',
        arguments: JSON.stringify({ expr: '2+2' }),
      })
      expect(functionOutput).toMatchObject({
        type: 'function_call_output',
        call_id: 'call_1',
        output: '4',
      })
    })

    it('prefixes errored tool results with [ERROR]', async () => {
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('x')] }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 't1',
              status: 'error',
              content: [new TextBlock('boom')],
            }),
          ],
        }),
      ]
      const req = await runOnce({}, messages)
      const out = req.input.find((i: any) => i.type === 'function_call_output')
      expect(out.output).toBe('[ERROR] boom')
    })

    it('emits an array output with input_image when a tool result carries image bytes', async () => {
      const imageBytes = new Uint8Array([1, 2, 3, 4])
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('fetch')] }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'img_tool',
              status: 'success',
              content: [
                new TextBlock('here is the image'),
                new ImageBlock({ format: 'png', source: { bytes: imageBytes } }),
              ],
            }),
          ],
        }),
      ]
      const req = await runOnce({}, messages)
      const out = req.input.find((i: any) => i.type === 'function_call_output')
      expect(Array.isArray(out.output)).toBe(true)
      expect(out.output).toEqual([
        { type: 'input_text', text: 'here is the image' },
        { type: 'input_image', image_url: expect.stringMatching(/^data:image\/png;base64,/) },
      ])
    })

    it('emits an array output with input_file when a tool result carries a document', async () => {
      const docBytes = new Uint8Array([5, 6, 7, 8])
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('read')] }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'doc_tool',
              status: 'success',
              content: [new DocumentBlock({ name: 'report.pdf', format: 'pdf', source: { bytes: docBytes } })],
            }),
          ],
        }),
      ]
      const req = await runOnce({}, messages)
      const out = req.input.find((i: any) => i.type === 'function_call_output')
      expect(Array.isArray(out.output)).toBe(true)
      expect(out.output).toEqual([
        {
          type: 'input_file',
          file_data: expect.stringMatching(/^data:application\/pdf;base64,/),
          filename: 'report.pdf',
        },
      ])
    })

    it('keeps tool result output as a plain string when only text is present', async () => {
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('ping')] }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'text_tool',
              status: 'success',
              content: [new TextBlock('pong')],
            }),
          ],
        }),
      ]
      const req = await runOnce({}, messages)
      const out = req.input.find((i: any) => i.type === 'function_call_output')
      expect(typeof out.output).toBe('string')
      expect(out.output).toBe('pong')
    })
  })

  describe('stream event mapping', () => {
    it('captures responseId on response.created when stateful', async () => {
      const modelState = new StateStore()
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'resp_abc' } }
        yield { type: 'response.completed', response: {} }
      })
      const model = new OpenAIModel({ api: 'responses', client, stateful: true })
      await collectIterator(
        model.stream([new Message({ role: 'user', content: [new TextBlock('hi')] })], { modelState })
      )
      expect(modelState.get('responseId')).toBe('resp_abc')
    })

    it('does NOT capture responseId when stateful is disabled', async () => {
      const modelState = new StateStore()
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'resp_abc' } }
        yield { type: 'response.completed', response: {} }
      })
      const model = new OpenAIModel({ api: 'responses', client })
      await collectIterator(
        model.stream([new Message({ role: 'user', content: [new TextBlock('hi')] })], { modelState })
      )
      expect(modelState.get('responseId')).toBeUndefined()
    })

    it('emits text deltas inside a content block', async () => {
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'r' } }
        yield { type: 'response.output_text.delta', delta: 'Hello' }
        yield { type: 'response.output_text.delta', delta: ' world' }
        yield { type: 'response.completed', response: {} }
      })
      const model = new OpenAIModel({ api: 'responses', client })
      const events = await collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      const types = events.map((e: any) => e.type)
      expect(types).toEqual([
        'modelMessageStartEvent',
        'modelContentBlockStartEvent',
        'modelContentBlockDeltaEvent',
        'modelContentBlockDeltaEvent',
        'modelContentBlockStopEvent',
        'modelMessageStopEvent',
      ])
      const deltas = events.filter((e: any) => e.type === 'modelContentBlockDeltaEvent').map((e: any) => e.delta)
      expect(deltas).toEqual([
        { type: 'textDelta', text: 'Hello' },
        { type: 'textDelta', text: ' world' },
      ])
    })

    it('switches content blocks between reasoning and text', async () => {
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'r' } }
        yield { type: 'response.reasoning_text.delta', delta: 'thinking...' }
        yield { type: 'response.output_text.delta', delta: 'answer' }
        yield { type: 'response.completed', response: {} }
      })
      const model = new OpenAIModel({ api: 'responses', client })
      const events = await collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      const types = events.map((e: any) => e.type)
      expect(types).toEqual([
        'modelMessageStartEvent',
        'modelContentBlockStartEvent',
        'modelContentBlockDeltaEvent', // reasoning
        'modelContentBlockStopEvent',
        'modelContentBlockStartEvent',
        'modelContentBlockDeltaEvent', // text
        'modelContentBlockStopEvent',
        'modelMessageStopEvent',
      ])
    })

    it('emits tool call triplet after stream close and sets stopReason=toolUse', async () => {
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'r' } }
        yield {
          type: 'response.output_item.added',
          item: { type: 'function_call', id: 'item_1', call_id: 'call_1', name: 'calc' },
        }
        yield {
          type: 'response.function_call_arguments.delta',
          item_id: 'item_1',
          delta: '{"a":',
        }
        yield {
          type: 'response.function_call_arguments.delta',
          item_id: 'item_1',
          delta: '1}',
        }
        yield {
          type: 'response.function_call_arguments.done',
          item_id: 'item_1',
          arguments: '{"a":1}',
        }
        yield { type: 'response.completed', response: {} }
      })
      const model = new OpenAIModel({ api: 'responses', client })
      const events = await collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      const startEvent = events.find(
        (e: any) => e.type === 'modelContentBlockStartEvent' && e.start?.type === 'toolUseStart'
      ) as any
      expect(startEvent?.start).toEqual({
        type: 'toolUseStart',
        name: 'calc',
        toolUseId: 'call_1',
      })
      const deltaEvent = events.find(
        (e: any) => e.type === 'modelContentBlockDeltaEvent' && e.delta?.type === 'toolUseInputDelta'
      ) as any
      expect(deltaEvent?.delta).toEqual({ type: 'toolUseInputDelta', input: '{"a":1}' })
      const stopEvent = events.find((e: any) => e.type === 'modelMessageStopEvent') as any
      expect(stopEvent?.stopReason).toBe('toolUse')
    })

    it('maps response.incomplete with max_output_tokens to stopReason=maxTokens', async () => {
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'r' } }
        yield { type: 'response.output_text.delta', delta: 'partial' }
        yield {
          type: 'response.incomplete',
          response: {
            incomplete_details: { reason: 'max_output_tokens' },
            usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
          },
        }
      })
      const model = new OpenAIModel({ api: 'responses', client })
      const events = await collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      const stop = events.find((e: any) => e.type === 'modelMessageStopEvent') as any
      expect(stop?.stopReason).toBe('maxTokens')
      const metadata = events.find((e: any) => e.type === 'modelMetadataEvent') as any
      expect(metadata?.usage).toEqual({ inputTokens: 10, outputTokens: 5, totalTokens: 15 })
    })

    it('emits URL citation delta from response.output_text.annotation.added', async () => {
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'r' } }
        yield { type: 'response.output_text.delta', delta: 'The answer is here.' }
        yield {
          type: 'response.output_text.annotation.added',
          annotation: {
            type: 'url_citation',
            url: 'https://example.com',
            title: 'Example',
            cited_text: 'here',
          },
        }
        yield { type: 'response.completed', response: {} }
      })
      const model = new OpenAIModel({ api: 'responses', client })
      const events = await collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      const citation = events.find(
        (e: any) => e.type === 'modelContentBlockDeltaEvent' && e.delta?.type === 'citationsDelta'
      ) as any
      expect(citation?.delta.citations[0]).toMatchObject({
        location: { type: 'web', url: 'https://example.com' },
        source: 'https://example.com',
        title: 'Example',
      })
    })

    it('closes the text block before a citation, producing separate blocks when stream ends after citation', async () => {
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'r' } }
        yield { type: 'response.output_text.delta', delta: 'Before citation' }
        yield {
          type: 'response.output_text.annotation.added',
          annotation: {
            type: 'url_citation',
            url: 'https://example.com',
            title: 'Source',
            cited_text: 'cited',
          },
        }
        yield { type: 'response.completed', response: {} }
      })
      const model = new OpenAIModel({ api: 'responses', client })
      const events = await collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      const types = events.map((e: any) => e.type)
      expect(types).toEqual([
        'modelMessageStartEvent',
        // Text block — closed before citation
        'modelContentBlockStartEvent',
        'modelContentBlockDeltaEvent',
        'modelContentBlockStopEvent',
        // Citation block
        'modelContentBlockStartEvent',
        'modelContentBlockDeltaEvent',
        'modelContentBlockStopEvent',
        'modelMessageStopEvent',
      ])
      const deltas = events.filter((e: any) => e.type === 'modelContentBlockDeltaEvent').map((e: any) => e.delta.type)
      expect(deltas).toEqual(['textDelta', 'citationsDelta'])
    })

    it('closes the text block before a citation and opens a new text block after', async () => {
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'r' } }
        yield { type: 'response.output_text.delta', delta: 'Before ' }
        yield {
          type: 'response.output_text.annotation.added',
          annotation: {
            type: 'url_citation',
            url: 'https://example.com',
            title: 'Source',
            cited_text: 'cited',
          },
        }
        yield { type: 'response.output_text.delta', delta: ' after' }
        yield { type: 'response.completed', response: {} }
      })
      const model = new OpenAIModel({ api: 'responses', client })
      const events = await collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      const types = events.map((e: any) => e.type)
      expect(types).toEqual([
        'modelMessageStartEvent',
        // First text block
        'modelContentBlockStartEvent',
        'modelContentBlockDeltaEvent',
        'modelContentBlockStopEvent',
        // Citation block
        'modelContentBlockStartEvent',
        'modelContentBlockDeltaEvent',
        'modelContentBlockStopEvent',
        // New text block after citation
        'modelContentBlockStartEvent',
        'modelContentBlockDeltaEvent',
        'modelContentBlockStopEvent',
        'modelMessageStopEvent',
      ])
      const deltas = events.filter((e: any) => e.type === 'modelContentBlockDeltaEvent').map((e: any) => e.delta.type)
      expect(deltas).toEqual(['textDelta', 'citationsDelta', 'textDelta'])
    })

    it('keeps consecutive citations in the same block without extra stop/start', async () => {
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'r' } }
        yield {
          type: 'response.output_text.annotation.added',
          annotation: { type: 'url_citation', url: 'https://a.com', title: 'A', cited_text: 'a' },
        }
        yield {
          type: 'response.output_text.annotation.added',
          annotation: { type: 'url_citation', url: 'https://b.com', title: 'B', cited_text: 'b' },
        }
        yield { type: 'response.completed', response: {} }
      })
      const model = new OpenAIModel({ api: 'responses', client })
      const events = await collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      const types = events.map((e: any) => e.type)
      expect(types).toEqual([
        'modelMessageStartEvent',
        'modelContentBlockStartEvent',
        'modelContentBlockDeltaEvent',
        'modelContentBlockDeltaEvent',
        'modelContentBlockStopEvent',
        'modelMessageStopEvent',
      ])
    })

    it('handles text → citation → text → citation → text with separate blocks each time', async () => {
      const client = createMockClient(async function* () {
        yield { type: 'response.created', response: { id: 'r' } }
        yield { type: 'response.output_text.delta', delta: 'intro ' }
        yield {
          type: 'response.output_text.annotation.added',
          annotation: { type: 'url_citation', url: 'https://1.com', title: '1', cited_text: 'c1' },
        }
        yield { type: 'response.output_text.delta', delta: 'middle ' }
        yield {
          type: 'response.output_text.annotation.added',
          annotation: { type: 'url_citation', url: 'https://2.com', title: '2', cited_text: 'c2' },
        }
        yield { type: 'response.output_text.delta', delta: 'end' }
        yield { type: 'response.completed', response: {} }
      })
      const model = new OpenAIModel({ api: 'responses', client })
      const events = await collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      const deltaTypes = events
        .filter((e: any) => e.type === 'modelContentBlockDeltaEvent')
        .map((e: any) => e.delta.type)
      expect(deltaTypes).toEqual(['textDelta', 'citationsDelta', 'textDelta', 'citationsDelta', 'textDelta'])
      // 5 content blocks = 5 start + 5 stop events
      const starts = events.filter((e: any) => e.type === 'modelContentBlockStartEvent')
      const stops = events.filter((e: any) => e.type === 'modelContentBlockStopEvent')
      expect(starts).toHaveLength(5)
      expect(stops).toHaveLength(5)
    })
  })

  describe('error mapping', () => {
    it('wraps 429 as ModelThrottledError', async () => {
      const client: any = {
        responses: {
          create: vi.fn(async () => {
            const err: any = new Error('Too many requests')
            err.status = 429
            throw err
          }),
        },
      }
      const model = new OpenAIModel({ api: 'responses', client })
      await expect(
        collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      ).rejects.toBeInstanceOf(ModelThrottledError)
    })

    it('wraps context_length_exceeded as ContextWindowOverflowError', async () => {
      const client: any = {
        responses: {
          create: vi.fn(async () => {
            const err: any = new Error('This model has a maximum context length of 8k.')
            err.code = 'context_length_exceeded'
            throw err
          }),
        },
      }
      const model = new OpenAIModel({ api: 'responses', client })
      await expect(
        collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      ).rejects.toBeInstanceOf(ContextWindowOverflowError)
    })

    it('rethrows unknown errors untouched', async () => {
      const client: any = {
        responses: {
          create: vi.fn(async () => {
            throw new Error('some other failure')
          }),
        },
      }
      const model = new OpenAIModel({ api: 'responses', client })
      await expect(
        collectIterator(model.stream([new Message({ role: 'user', content: [new TextBlock('x')] })]))
      ).rejects.toThrow('some other failure')
    })
  })
})
