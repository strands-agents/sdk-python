import { describe, it, expect } from 'vitest'
import { isModelStreamEvent } from '../streaming.js'
import type { ModelStreamEvent } from '../streaming.js'

describe('isModelStreamEvent', () => {
  it('returns true for modelMessageStartEvent', () => {
    const event: ModelStreamEvent = { type: 'modelMessageStartEvent', role: 'assistant' }
    expect(isModelStreamEvent(event)).toBe(true)
  })

  it('returns true for modelContentBlockStartEvent', () => {
    const event: ModelStreamEvent = { type: 'modelContentBlockStartEvent' }
    expect(isModelStreamEvent(event)).toBe(true)
  })

  it('returns true for modelContentBlockDeltaEvent', () => {
    const event: ModelStreamEvent = {
      type: 'modelContentBlockDeltaEvent',
      delta: { type: 'textDelta', text: 'hello' },
    }
    expect(isModelStreamEvent(event)).toBe(true)
  })

  it('returns true for modelContentBlockStopEvent', () => {
    const event: ModelStreamEvent = { type: 'modelContentBlockStopEvent' }
    expect(isModelStreamEvent(event)).toBe(true)
  })

  it('returns true for modelMessageStopEvent', () => {
    const event: ModelStreamEvent = { type: 'modelMessageStopEvent', stopReason: 'endTurn' }
    expect(isModelStreamEvent(event)).toBe(true)
  })

  it('returns true for modelMetadataEvent', () => {
    const event: ModelStreamEvent = {
      type: 'modelMetadataEvent',
      usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
    }
    expect(isModelStreamEvent(event)).toBe(true)
  })

  it('returns true for modelRedactionEvent', () => {
    const event: ModelStreamEvent = {
      type: 'modelRedactionEvent',
      inputRedaction: { replaceContent: '[User input redacted.]' },
    }
    expect(isModelStreamEvent(event)).toBe(true)
  })

  it('returns false for unknown event types', () => {
    const event = { type: 'unknownEvent' }
    expect(isModelStreamEvent(event)).toBe(false)
  })

  it('returns false for content block types', () => {
    const event = { type: 'textBlock', text: 'hello' }
    expect(isModelStreamEvent(event)).toBe(false)
  })
})
