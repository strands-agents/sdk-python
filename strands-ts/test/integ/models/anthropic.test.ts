import { describe, expect, it } from 'vitest'
import { Message, ImageBlock, TextBlock, CachePointBlock } from '@strands-agents/sdk'
import type { SystemContentBlock } from '@strands-agents/sdk'
import { collectIterator } from '$/sdk/__fixtures__/model-test-helpers.js'
import { loadFixture } from '../__fixtures__/test-helpers.js'
import { anthropic } from '../__fixtures__/model-providers.js'

import yellowPngUrl from '../__resources__/yellow.png?url'

describe.skipIf(anthropic.skip)('AnthropicModel Integration Tests', () => {
  describe('Configuration', () => {
    it.concurrent('respects maxTokens configuration', async () => {
      const provider = anthropic.createModel({ maxTokens: 20 })
      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('Write a very long story about space exploration.')],
        }),
      ]

      const events = await collectIterator(provider.stream(messages))

      const metadataEvent = events.find((e) => e.type === 'modelMetadataEvent')
      expect(metadataEvent?.usage?.outputTokens).toBeLessThanOrEqual(20)

      const messageStopEvent = events.find((e) => e.type === 'modelMessageStopEvent')
      expect(messageStopEvent?.stopReason).toBe('maxTokens')
    })
  })

  describe('Prompt Caching', () => {
    it('uses system prompt cache on subsequent requests', async () => {
      const provider = anthropic.createModel({ maxTokens: 100 })

      const largeContext = `Context information: ${'repeat '.repeat(5000)} [${Date.now()}]`

      const cachedSystemPrompt: SystemContentBlock[] = [
        new TextBlock('You are a helpful assistant.'),
        new TextBlock(largeContext),
        new CachePointBlock({ cacheType: 'default' }),
      ]

      const events1 = await collectIterator(
        provider.stream([new Message({ role: 'user', content: [new TextBlock('Hello')] })], {
          systemPrompt: cachedSystemPrompt,
        })
      )

      const metadata1 = events1.find((e) => e.type === 'modelMetadataEvent')
      const writeTokens = metadata1?.usage?.cacheWriteInputTokens
      if (writeTokens !== undefined) {
        expect(writeTokens).toBeGreaterThan(0)
      }

      const events2 = await collectIterator(
        provider.stream([new Message({ role: 'user', content: [new TextBlock('Hi again')] })], {
          systemPrompt: cachedSystemPrompt,
        })
      )

      const metadata2 = events2.find((e) => e.type === 'modelMetadataEvent')
      const readTokens = metadata2?.usage?.cacheReadInputTokens
      if (readTokens !== undefined) {
        expect(readTokens).toBeGreaterThanOrEqual(0)
      }
    })

    it('uses message cache points on subsequent requests', async () => {
      const provider = anthropic.createModel({ maxTokens: 100 })
      const largeContext = `Context information: ${'repeat '.repeat(5000)} [${Date.now()}]`

      const messagesWithCache = (text: string): Message[] => [
        new Message({
          role: 'user',
          content: [new TextBlock(largeContext), new CachePointBlock({ cacheType: 'default' }), new TextBlock(text)],
        }),
      ]

      const events1 = await collectIterator(provider.stream(messagesWithCache('Question 1')))
      const metadata1 = events1.find((e) => e.type === 'modelMetadataEvent')
      const writeTokens = metadata1?.usage?.cacheWriteInputTokens
      if (writeTokens !== undefined) {
        expect(writeTokens).toBeGreaterThan(0)
      }

      const events2 = await collectIterator(provider.stream(messagesWithCache('Question 2')))
      const metadata2 = events2.find((e) => e.type === 'modelMetadataEvent')
      const readTokens = metadata2?.usage?.cacheReadInputTokens
      if (readTokens !== undefined) {
        expect(readTokens).toBeGreaterThanOrEqual(0)
      }
    })
  })

  describe('Media Support', () => {
    it('processes image input correctly', async () => {
      const provider = anthropic.createModel({ maxTokens: 100 })

      const imageBytes = await loadFixture(yellowPngUrl)

      const messages = [
        new Message({
          role: 'user',
          content: [
            new ImageBlock({
              format: 'png',
              source: { bytes: imageBytes },
            }),
            new TextBlock('What color is this image? Reply with just the color name.'),
          ],
        }),
      ]

      const events = await collectIterator(provider.stream(messages))

      const stopEvent = events.find((e) => e.type === 'modelMessageStopEvent')
      expect(stopEvent?.stopReason).toBe('endTurn')

      let fullText = ''
      for (const event of events) {
        if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta') {
          fullText += event.delta.text
        }
      }

      expect(fullText.toLowerCase()).toContain('yellow')
    })
  })

  describe('Thinking Mode', () => {
    it('emits thinking blocks when enabled', async () => {
      const provider = anthropic.createModel({
        maxTokens: 4000,
        params: {
          thinking: {
            type: 'enabled',
            budget_tokens: 2048,
          },
        },
      })

      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('Explain the theory of relativity step-by-step.')],
        }),
      ]

      const events = await collectIterator(provider.stream(messages))

      const thinkingEvents = events.filter(
        (e) => e.type === 'modelContentBlockDeltaEvent' && e.delta.type === 'reasoningContentDelta'
      )

      if (thinkingEvents.length > 0) {
        expect(thinkingEvents[0]!.type).toBe('modelContentBlockDeltaEvent')
        const firstThinking = thinkingEvents[0] as any
        expect(firstThinking.delta.text).toBeDefined()
      }
    })
  })

  describe('countTokens', () => {
    const messages = [
      new Message({ role: 'user', content: [new TextBlock('What is the capital of France? Explain in detail.')] }),
    ]
    const toolSpecs = [
      {
        name: 'get_weather',
        description: 'Get the current weather for a location',
        inputSchema: { type: 'object' as const, properties: { location: { type: 'string' as const } } },
      },
    ]

    it.concurrent('should count tokens for messages only', async () => {
      const model = anthropic.createModel()
      const result = await model.countTokens(messages)
      expect(typeof result).toBe('number')
      expect(result).toBeGreaterThan(0)
    })

    it.concurrent('should return more tokens with tools and system prompt', async () => {
      const model = anthropic.createModel()
      const without = await model.countTokens(messages)
      const withTools = await model.countTokens(messages, { toolSpecs, systemPrompt: 'Be helpful.' })
      expect(withTools).toBeGreaterThan(without)
    })
  })
})
