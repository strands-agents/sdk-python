import { describe, it, expect, vi } from 'vitest'
import { api, LifecycleBridge } from '../entry'
import { Agent } from '@strands-agents/sdk'
import { MockMessageModel } from '$/fixtures/mock-message-model'

const ResponseStream = api.ResponseStream

function createAgent(): Agent {
  return new Agent({ model: new MockMessageModel(), printer: false })
}

const beforeModelCallEvent = {
  tag: 'lifecycle',
  val: { eventType: 'before-model-call', toolUse: undefined, toolResult: undefined },
}

const afterInvocationEvent = {
  tag: 'lifecycle',
  val: { eventType: 'after-invocation', toolUse: undefined, toolResult: undefined },
}

function setupStream(
  genFn: () => AsyncGenerator<any, any, undefined>,
  preQueued?: any[]
): { stream: InstanceType<typeof ResponseStream>; bridge: LifecycleBridge } {
  const agent = createAgent()
  const bridge = new LifecycleBridge()
  if (preQueued) bridge.queue.push(...preQueued)
  vi.spyOn(agent, 'stream').mockReturnValue(genFn())
  return { stream: new ResponseStream(agent, 'test', bridge), bridge }
}

describe('ResponseStreamImpl.readNext', () => {
  describe('mid-stream batch', () => {
    it('returns lifecycle events interleaved with mapped event', async () => {
      const { stream } = setupStream(
        async function* () {
          yield {
            type: 'modelStreamUpdateEvent',
            event: { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'hello' } },
          }
        },
        [beforeModelCallEvent]
      )
      const batch = await stream.readNext()

      expect(batch).toStrictEqual([beforeModelCallEvent, { tag: 'text-delta', val: 'hello' }])
    })

    it('returns empty array when no events to report', async () => {
      const { stream } = setupStream(async function* () {
        yield { type: 'unknownEvent' }
      })
      const batch = await stream.readNext()

      expect(batch).toStrictEqual([])
    })
  })

  describe('final batch', () => {
    it('returns stop event when generator completes with result', async () => {
      const { stream } = setupStream(async function* () {
        return {
          stopReason: 'endTurn',
          metrics: {
            accumulatedUsage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
            accumulatedMetrics: { latencyMs: 100 },
          },
        }
      })
      const batch = await stream.readNext()

      expect(batch).toStrictEqual([
        {
          tag: 'stop',
          val: {
            reason: 'end-turn',
            usage: {
              inputTokens: 1,
              outputTokens: 2,
              totalTokens: 3,
              cacheReadInputTokens: undefined,
              cacheWriteInputTokens: undefined,
            },
            metrics: { latencyMs: 100 },
            structuredOutput: undefined,
          },
        },
      ])
    })

    it('returns lifecycle events when generator completes with no result but has pending lifecycle events', async () => {
      const { stream } = setupStream(
        async function* () {
          return undefined
        },
        [afterInvocationEvent]
      )
      const batch = await stream.readNext()

      expect(batch).toStrictEqual([afterInvocationEvent])
    })

    it('returns undefined when generator completes with no result and no lifecycle events', async () => {
      const { stream } = setupStream(async function* () {
        return undefined
      })
      const batch = await stream.readNext()

      expect(batch).toBeUndefined()
    })
  })

  describe('error batch', () => {
    it('returns lifecycle events with error event when generator throws', async () => {
      const { stream } = setupStream(
        async function* () {
          throw new Error('model failed')
        },
        [beforeModelCallEvent]
      )
      const batch = await stream.readNext()

      expect(batch).toStrictEqual([beforeModelCallEvent, { tag: 'error', val: 'model failed' }])
    })
  })

  describe('done state', () => {
    it('returns undefined after stream is done', async () => {
      const { stream } = setupStream(async function* () {
        return { stopReason: 'endTurn' }
      })
      await stream.readNext()
      const batch = await stream.readNext()

      expect(batch).toBeUndefined()
    })
  })

  describe('cancel', () => {
    it('cancel sets done state', async () => {
      const { stream } = setupStream(async function* () {
        yield {
          type: 'modelStreamUpdateEvent',
          event: { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'hello' } },
        }
        yield {
          type: 'modelStreamUpdateEvent',
          event: { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'world' } },
        }
      })
      stream.cancel()
      const batch = await stream.readNext()

      expect(batch).toBeUndefined()
    })

    it('cancel restores default tools and model', async () => {
      const agent = createAgent()
      const bridge = new LifecycleBridge()
      const defaultTools = [{ name: 'default_tool' }] as any[]
      const clearSpy = vi.spyOn(agent.toolRegistry, 'clear')
      const addSpy = vi.spyOn(agent.toolRegistry, 'add')

      vi.spyOn(agent, 'stream').mockReturnValue(
        (async function* () {
          yield {
            type: 'modelStreamUpdateEvent',
            event: { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'hello' } },
          }
        })()
      )

      const stream = new ResponseStream(agent, 'test', bridge, defaultTools)
      stream.cancel()

      const batch = await stream.readNext()
      expect(batch).toBeUndefined()
      expect(clearSpy).toHaveBeenCalled()
      expect(addSpy).toHaveBeenCalledWith(defaultTools)
    })
  })
})
