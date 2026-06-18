import { describe, it, expect } from 'vitest'
import {
  mapUsage,
  mapMetrics,
  mapStopReasonTag,
  mapStopReason,
  mapEvent,
  mapModelStreamEvent,
  mapContentBlock,
  mapToolStreamEvent,
  parseInput,
  parseStructuredOutputSchema,
  parseSaveLatestStrategy,
} from '../entry'
import type { AgentStreamEvent, ModelStreamEvent, StopReason } from '@strands-agents/sdk'
import { ToolStreamEvent, ToolUseBlock, ToolResultBlock, TextBlock, ReasoningBlock } from '@strands-agents/sdk'

describe('mapUsage', () => {
  it.each([null, undefined])('returns undefined for %s input', (input) => {
    expect(mapUsage(input)).toBeUndefined()
  })

  it('maps all fields correctly', () => {
    expect(mapUsage({ inputTokens: 10, outputTokens: 20, totalTokens: 30 })).toStrictEqual({
      inputTokens: 10,
      outputTokens: 20,
      totalTokens: 30,
      cacheReadInputTokens: undefined,
      cacheWriteInputTokens: undefined,
    })
  })

  it('computes totalTokens when missing', () => {
    expect(mapUsage({ inputTokens: 5, outputTokens: 3 })).toStrictEqual({
      inputTokens: 5,
      outputTokens: 3,
      totalTokens: 8,
      cacheReadInputTokens: undefined,
      cacheWriteInputTokens: undefined,
    })
  })

  it('includes cache fields when present', () => {
    expect(
      mapUsage({
        inputTokens: 10,
        outputTokens: 20,
        totalTokens: 30,
        cacheReadInputTokens: 5,
        cacheWriteInputTokens: 2,
      })
    ).toStrictEqual({
      inputTokens: 10,
      outputTokens: 20,
      totalTokens: 30,
      cacheReadInputTokens: 5,
      cacheWriteInputTokens: 2,
    })
  })
})

describe('mapMetrics', () => {
  it.each([null, undefined])('returns undefined for %s input', (input) => {
    expect(mapMetrics(input)).toBeUndefined()
  })

  it('maps latencyMs', () => {
    expect(mapMetrics({ latencyMs: 150 })).toStrictEqual({ latencyMs: 150 })
  })

  it('defaults latencyMs to 0 when field is absent', () => {
    expect(mapMetrics({})).toStrictEqual({ latencyMs: 0 })
  })

  it('defaults latencyMs to 0 when field is explicitly undefined', () => {
    expect(mapMetrics({ latencyMs: undefined })).toStrictEqual({ latencyMs: 0 })
  })
})

describe('mapStopReasonTag', () => {
  const mappings: [string, string][] = [
    ['endTurn', 'end-turn'],
    ['toolUse', 'tool-use'],
    ['maxTokens', 'max-tokens'],
    ['contentFiltered', 'content-filtered'],
    ['guardrailIntervened', 'guardrail-intervened'],
    ['stopSequence', 'stop-sequence'],
    ['modelContextWindowExceeded', 'model-context-window-exceeded'],
    ['cancelled', 'cancelled'],
  ]

  it.each(mappings)("maps '%s' to '%s'", (input, expected) => {
    expect(mapStopReasonTag(input as StopReason)).toBe(expected)
  })

  it("maps unknown reason to 'error'", () => {
    expect(mapStopReasonTag('unknownReason' as unknown as StopReason)).toBe('error')
  })

  it('covers every WIT StopReason variant except error', () => {
    const witStopReasons = [
      'end-turn',
      'tool-use',
      'max-tokens',
      'error',
      'content-filtered',
      'guardrail-intervened',
      'stop-sequence',
      'model-context-window-exceeded',
      'cancelled',
    ]
    const mappedOutputs = mappings.map(([, wit]) => wit)
    const nonErrorVariants = witStopReasons.filter((r) => r !== 'error')
    expect(mappedOutputs.sort()).toStrictEqual(nonErrorVariants.sort())
  })
})

describe('mapStopReason', () => {
  it('maps reason with no agent result', () => {
    expect(mapStopReason('endTurn')).toStrictEqual({
      reason: 'end-turn',
      usage: undefined,
      metrics: undefined,
      structuredOutput: undefined,
    })
  })

  it('maps reason with usage and metrics', () => {
    expect(
      mapStopReason('toolUse', {
        usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
        metrics: { latencyMs: 100 },
      })
    ).toStrictEqual({
      reason: 'tool-use',
      usage: {
        inputTokens: 1,
        outputTokens: 2,
        totalTokens: 3,
        cacheReadInputTokens: undefined,
        cacheWriteInputTokens: undefined,
      },
      metrics: { latencyMs: 100 },
      structuredOutput: undefined,
    })
  })

  it('serializes structured output as JSON string', () => {
    expect(
      mapStopReason('endTurn', {
        structuredOutput: { name: 'Alice', age: 30 },
      })
    ).toStrictEqual({
      reason: 'end-turn',
      usage: undefined,
      metrics: undefined,
      structuredOutput: '{"name":"Alice","age":30}',
    })
  })

  it('sets structuredOutput to undefined when not present', () => {
    expect(
      mapStopReason('endTurn', { usage: { inputTokens: 5, outputTokens: 10, totalTokens: 15 } })
    ).toStrictEqual({
      reason: 'end-turn',
      usage: {
        inputTokens: 5,
        outputTokens: 10,
        totalTokens: 15,
        cacheReadInputTokens: undefined,
        cacheWriteInputTokens: undefined,
      },
      metrics: undefined,
      structuredOutput: undefined,
    })
  })
})

describe('mapModelStreamEvent', () => {
  it('maps text delta', () => {
    const event: ModelStreamEvent = { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'hello' } }
    expect(mapModelStreamEvent(event)).toStrictEqual({ tag: 'text-delta', val: 'hello' })
  })

  it('returns null for non-text delta', () => {
    const event: ModelStreamEvent = {
      type: 'modelContentBlockDeltaEvent',
      delta: { type: 'toolUseInputDelta', input: '{}' },
    }
    expect(mapModelStreamEvent(event)).toBeNull()
  })

  it('returns null for reasoningContentDelta', () => {
    const event: ModelStreamEvent = {
      type: 'modelContentBlockDeltaEvent',
      delta: { type: 'reasoningContentDelta', text: 'thinking...' },
    }
    expect(mapModelStreamEvent(event)).toBeNull()
  })

  it('maps modelContentBlockStartEvent with toolUseStart', () => {
    const event: ModelStreamEvent = {
      type: 'modelContentBlockStartEvent',
      start: { type: 'toolUseStart', name: 'calc', toolUseId: 'tu-5' },
    }
    expect(mapModelStreamEvent(event)).toStrictEqual({
      tag: 'tool-use',
      val: { name: 'calc', toolUseId: 'tu-5', input: '{}' },
    })
  })

  it('returns null for modelContentBlockStartEvent without start', () => {
    const event: ModelStreamEvent = { type: 'modelContentBlockStartEvent' }
    expect(mapModelStreamEvent(event)).toBeNull()
  })

  it('maps modelMetadataEvent', () => {
    const event: ModelStreamEvent = {
      type: 'modelMetadataEvent',
      usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
      metrics: { latencyMs: 50 },
    }
    expect(mapModelStreamEvent(event)).toStrictEqual({
      tag: 'metadata',
      val: {
        usage: {
          inputTokens: 10,
          outputTokens: 20,
          totalTokens: 30,
          cacheReadInputTokens: undefined,
          cacheWriteInputTokens: undefined,
        },
        metrics: { latencyMs: 50 },
      },
    })
  })

  it('returns null for unrecognized model event', () => {
    const event: ModelStreamEvent = { type: 'modelMessageStartEvent', role: 'assistant' }
    expect(mapModelStreamEvent(event)).toBeNull()
  })

  it('maps modelMetadataEvent without usage or metrics', () => {
    const event: ModelStreamEvent = { type: 'modelMetadataEvent' }
    expect(mapModelStreamEvent(event)).toStrictEqual({
      tag: 'metadata',
      val: { usage: undefined, metrics: undefined },
    })
  })
})

describe('mapContentBlock', () => {
  it('maps toolUseBlock', () => {
    const block = new ToolUseBlock({ name: 'calc', toolUseId: 'tu-1', input: { x: 1 } })
    expect(mapContentBlock(block)).toStrictEqual({
      tag: 'tool-use',
      val: { name: 'calc', toolUseId: 'tu-1', input: '{"x":1}' },
    })
  })

  it('maps toolUseBlock with null input to empty object', () => {
    const block = new ToolUseBlock({ name: 'calc', toolUseId: 'tu-1', input: null })
    expect(mapContentBlock(block)).toStrictEqual({
      tag: 'tool-use',
      val: { name: 'calc', toolUseId: 'tu-1', input: '{}' },
    })
  })

  it('maps toolResultBlock', () => {
    const block = new ToolResultBlock({
      toolUseId: 'tu-1',
      status: 'success',
      content: [new TextBlock('ok')],
    })
    expect(mapContentBlock(block)).toStrictEqual({
      tag: 'tool-result',
      val: { toolUseId: 'tu-1', status: 'success', content: '[{"text":"ok"}]' },
    })
  })

  it('returns null for textBlock', () => {
    const block = new TextBlock('hello')
    expect(mapContentBlock(block)).toBeNull()
  })

  it('returns null for reasoningBlock', () => {
    const block = new ReasoningBlock({ text: '' })
    expect(mapContentBlock(block)).toBeNull()
  })
})

describe('mapToolStreamEvent', () => {
  it('maps event with data', () => {
    const event = new ToolStreamEvent({ data: { value: 42 } })
    expect(mapToolStreamEvent(event)).toStrictEqual({
      tag: 'tool-result',
      val: { toolUseId: '', status: 'success', content: '{"data":{"value":42}}' },
    })
  })

  it('maps event without data', () => {
    const event = new ToolStreamEvent({})
    expect(mapToolStreamEvent(event)).toStrictEqual({
      tag: 'tool-result',
      val: { toolUseId: '', status: 'success', content: '{"data":null}' },
    })
  })

  it('maps event with string data', () => {
    const event = new ToolStreamEvent({ data: 'processing step 1' })
    expect(mapToolStreamEvent(event)).toStrictEqual({
      tag: 'tool-result',
      val: { toolUseId: '', status: 'success', content: '{"data":"processing step 1"}' },
    })
  })
})

describe('mapEvent', () => {
  describe('wrapper events', () => {
    it('unwraps modelStreamUpdateEvent to mapModelStreamEvent', () => {
      const event = {
        type: 'modelStreamUpdateEvent',
        event: { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'wrapped' } },
      } as unknown as AgentStreamEvent
      expect(mapEvent(event)).toStrictEqual({ tag: 'text-delta', val: 'wrapped' })
    })

    it('unwraps contentBlockEvent to mapContentBlock', () => {
      const event = {
        type: 'contentBlockEvent',
        contentBlock: new ToolUseBlock({ name: 'tool1', toolUseId: 'tu-2', input: {} }),
      } as unknown as AgentStreamEvent
      expect(mapEvent(event)).toStrictEqual({
        tag: 'tool-use',
        val: { name: 'tool1', toolUseId: 'tu-2', input: '{}' },
      })
    })

    it('unwraps toolResultEvent to mapContentBlock', () => {
      const event = {
        type: 'toolResultEvent',
        result: new ToolResultBlock({ toolUseId: 'tu-3', status: 'error', content: [] }),
      } as unknown as AgentStreamEvent
      expect(mapEvent(event)).toStrictEqual({
        tag: 'tool-result',
        val: { toolUseId: 'tu-3', status: 'error', content: '[]' },
      })
    })

    it('unwraps toolStreamUpdateEvent to mapToolStreamEvent', () => {
      const event = {
        type: 'toolStreamUpdateEvent',
        event: new ToolStreamEvent({ data: { progress: 50 } }),
      } as unknown as AgentStreamEvent
      expect(mapEvent(event)).toStrictEqual({
        tag: 'tool-result',
        val: { toolUseId: '', status: 'success', content: '{"data":{"progress":50}}' },
      })
    })
  })

  describe('special events', () => {
    it('maps interrupt event', () => {
      const event = { interrupt: { reason: 'user' } }
      expect(mapEvent(event as unknown as AgentStreamEvent)).toStrictEqual({
        tag: 'interrupt',
        val: JSON.stringify(event),
      })
    })

    it('does not treat hook events with interrupt() method as interrupt stream events', () => {
      const event = { type: 'beforeToolCallEvent', interrupt: () => {} }
      expect(mapEvent(event as unknown as AgentStreamEvent)).toBeNull()
    })
  })

  describe('dropped events', () => {
    it.each([
      'beforeInvocationEvent',
      'afterInvocationEvent',
      'beforeModelCallEvent',
      'afterModelCallEvent',
      'beforeToolCallEvent',
      'afterToolCallEvent',
      'messageAddedEvent',
      'modelMessageEvent',
      'agentResultEvent',
      'beforeToolsEvent',
      'afterToolsEvent',
    ])('returns null for %s', (type) => {
      const event = { type } as unknown as AgentStreamEvent
      expect(mapEvent(event)).toBeNull()
    })
  })
})

describe('parseInput', () => {
  it('returns parsed array for JSON array input', () => {
    expect(parseInput('[{"type":"text","text":"hi"}]')).toStrictEqual([{ type: 'text', text: 'hi' }])
  })

  it('returns string for plain text', () => {
    expect(parseInput('hello world')).toBe('hello world')
  })

  it('returns original string for JSON object (non-array)', () => {
    expect(parseInput('{"key":"value"}')).toBe('{"key":"value"}')
  })

  it('returns empty string for empty input', () => {
    expect(parseInput('')).toBe('')
  })

  it('returns original string for malformed JSON', () => {
    expect(parseInput('{bad json')).toBe('{bad json')
  })
})

describe('parseSaveLatestStrategy', () => {
  it.each(['message', 'invocation', 'trigger'] as const)("accepts valid strategy '%s'", (strategy) => {
    expect(parseSaveLatestStrategy(strategy)).toBe(strategy)
  })

  it('returns undefined for unknown strategy', () => {
    expect(parseSaveLatestStrategy('unknown')).toBeUndefined()
  })

  it('returns undefined for undefined input', () => {
    expect(parseSaveLatestStrategy(undefined)).toBeUndefined()
  })

  it('returns undefined for empty string', () => {
    expect(parseSaveLatestStrategy('')).toBeUndefined()
  })
})

describe('parseStructuredOutputSchema', () => {
  it('returns undefined for undefined input', () => {
    expect(parseStructuredOutputSchema(undefined)).toBeUndefined()
  })

  it('returns undefined for empty string', () => {
    expect(parseStructuredOutputSchema('')).toBeUndefined()
  })

  it('parses a valid JSON schema into a Zod schema', () => {
    const schema = parseStructuredOutputSchema(
      JSON.stringify({ type: 'object', properties: { name: { type: 'string' } }, required: ['name'] })
    )
    expect(schema).toBeDefined()
    expect(schema!.parse({ name: 'Alice' })).toStrictEqual({ name: 'Alice' })
  })

  it('throws on invalid JSON', () => {
    expect(() => parseStructuredOutputSchema('not valid json')).toThrow('Invalid structured output schema')
  })

  it('throws on invalid schema', () => {
    expect(() => parseStructuredOutputSchema(JSON.stringify({ type: 'invalid_type_xyz' }))).toThrow(
      'Invalid structured output schema'
    )
  })
})
