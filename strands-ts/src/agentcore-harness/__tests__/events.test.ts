import { describe, expect, it } from 'vitest'
import { AgentCoreHarnessStreamUpdateEvent, AgentCoreHarnessResultEvent } from '../events.js'
import { AgentResult } from '../../types/agent.js'
import { Message, TextBlock } from '../../types/messages.js'
import { AgentMetrics } from '../../telemetry/meter.js'
import { anyTrackingId } from '../../__fixtures__/message-helpers.js'
import type { AgentCoreHarnessEventData } from '../events.js'

describe('AgentCoreHarnessStreamUpdateEvent', () => {
  const chunk = { messageStart: { role: 'assistant' } } as AgentCoreHarnessEventData

  it('creates instance with correct properties', () => {
    const event = new AgentCoreHarnessStreamUpdateEvent(chunk)

    expect(event.type).toBe('agentCoreHarnessStreamUpdateEvent')
    expect(event.event).toBe(chunk)
  })

  describe('toJSON', () => {
    const event = new AgentCoreHarnessStreamUpdateEvent(chunk)

    it('serializes', () => {
      expect(JSON.parse(JSON.stringify(event))).toStrictEqual({
        type: 'agentCoreHarnessStreamUpdateEvent',
        event: { messageStart: { role: 'assistant' } },
      })
    })

    it('only excludes expected fields', () => {
      const json = event.toJSON()
      expect(Object.keys(event).filter((key) => !(key in json))).toStrictEqual([])
    })
  })
})

describe('AgentCoreHarnessResultEvent', () => {
  const result = new AgentResult({
    stopReason: 'endTurn',
    lastMessage: new Message({ role: 'assistant', content: [new TextBlock('Done')] }),
    metrics: new AgentMetrics(),
    invocationState: {},
  })

  it('creates instance with correct properties', () => {
    const event = new AgentCoreHarnessResultEvent({ result })

    expect(event.type).toBe('agentCoreHarnessResultEvent')
    expect(event.result).toBe(result)
  })

  describe('toJSON', () => {
    const event = new AgentCoreHarnessResultEvent({ result })

    it('serializes', () => {
      expect(JSON.parse(JSON.stringify(event))).toStrictEqual({
        type: 'agentCoreHarnessResultEvent',
        result: {
          type: 'agentResult',
          stopReason: 'endTurn',
          lastMessage: { role: 'assistant', content: [{ text: 'Done' }], trackingId: anyTrackingId },
        },
      })
    })

    it('only excludes expected fields', () => {
      const json = event.toJSON()
      expect(Object.keys(event).filter((key) => !(key in json))).toStrictEqual([])
    })
  })
})
