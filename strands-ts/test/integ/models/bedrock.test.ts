import { beforeAll, describe, expect, it, vi } from 'vitest'
import {
  Agent,
  Message,
  NullConversationManager,
  SlidingWindowConversationManager,
  TextBlock,
  FunctionTool,
  CachePointBlock,
  ImageBlock,
} from '@strands-agents/sdk'
import type { SystemContentBlock, ModelRedactionEvent } from '@strands-agents/sdk'

import { collectIterator } from '$/sdk/__fixtures__/model-test-helpers.js'
import { bedrock } from '../__fixtures__/model-providers.js'
import { loadFixture } from '../__fixtures__/test-helpers.js'
import yellowPngUrl from '../__resources__/yellow.png?url'
import {
  BedrockClient,
  CreateGuardrailCommand,
  GetGuardrailCommand,
  ListGuardrailsCommand,
} from '@aws-sdk/client-bedrock'
import { inject } from 'vitest'

describe.skipIf(bedrock.skip)('BedrockModel Integration Tests', () => {
  describe('Streaming', () => {
    describe('Configuration', () => {
      it.concurrent('respects maxTokens configuration', async () => {
        const provider = bedrock.createModel({ maxTokens: 20 })
        const messages = [
          new Message({
            role: 'user',
            content: [new TextBlock('Write a long story about dragons.')],
          }),
        ]

        const events = await collectIterator(provider.stream(messages))

        const metadataEvent = events.find((e) => e.type === 'modelMetadataEvent')
        expect(metadataEvent?.usage?.outputTokens).toBeLessThanOrEqual(20)

        const messageStopEvent = events.find((e) => e.type === 'modelMessageStopEvent')
        expect(messageStopEvent?.stopReason).toBe('maxTokens')
      })

      it.concurrent('uses system prompt cache on subsequent requests', async () => {
        const provider = bedrock.createModel({
          modelId: 'global.anthropic.claude-sonnet-4-5-20250929-v1:0',
          maxTokens: 100,
        })
        const largeContext = `Context information: ${'hello '.repeat(2000)} [test-${Date.now()}-${Math.random()}]`
        const cachedSystemPrompt: SystemContentBlock[] = [
          new TextBlock('You are a helpful assistant.'),
          new TextBlock(largeContext),
          new CachePointBlock({ cacheType: 'default' }),
        ]

        // First request - creates cache
        const events1 = await collectIterator(
          provider.stream([new Message({ role: 'user', content: [new TextBlock('Say hello')] })], {
            systemPrompt: cachedSystemPrompt,
          })
        )
        const metadata1 = events1.find((e) => e.type === 'modelMetadataEvent')
        expect(metadata1?.usage?.cacheWriteInputTokens).toBeGreaterThan(0)

        // Second request - should use cache
        const events2 = await collectIterator(
          provider.stream([new Message({ role: 'user', content: [new TextBlock('Say goodbye')] })], {
            systemPrompt: cachedSystemPrompt,
          })
        )
        const metadata2 = events2.find((e) => e.type === 'modelMetadataEvent')
        expect(metadata2?.usage?.cacheReadInputTokens).toBeGreaterThan(0)
      })

      it.concurrent('uses message cache points on subsequent requests', async () => {
        const provider = bedrock.createModel({
          modelId: 'global.anthropic.claude-sonnet-4-5-20250929-v1:0',
          maxTokens: 100,
        })
        const largeContext = `Context information: ${'hello '.repeat(2000)} [test-${Date.now()}-${Math.random()}]`
        const messagesWithCachePoint = (text: string): Message[] => [
          new Message({
            role: 'user',
            content: [new TextBlock(largeContext), new CachePointBlock({ cacheType: 'default' }), new TextBlock(text)],
          }),
        ]

        // First request - creates cache
        const events1 = await collectIterator(provider.stream(messagesWithCachePoint('Say hello')))
        const metadata1 = events1.find((e) => e.type === 'modelMetadataEvent')
        expect(metadata1?.usage?.cacheWriteInputTokens).toBeGreaterThan(0)

        // Second request - should use cache
        const events2 = await collectIterator(provider.stream(messagesWithCachePoint('Say goodbye')))
        const metadata2 = events2.find((e) => e.type === 'modelMetadataEvent')
        expect(metadata2?.usage?.cacheReadInputTokens).toBeGreaterThan(0)
      })

      it.concurrent('uses cacheConfig to automatically inject cache points in tools and messages', async () => {
        const provider = bedrock.createModel({
          modelId: 'global.anthropic.claude-sonnet-4-5-20250929-v1:0',
          maxTokens: 100,
          cacheConfig: { strategy: 'auto' },
        })
        const largeContext = `Context information: ${'hello '.repeat(2000)} [test-${Date.now()}-${Math.random()}]`

        const toolSpecs = [
          {
            name: 'lookup',
            description: 'Look up information. '.repeat(100),
            inputSchema: { type: 'object' as const, properties: { query: { type: 'string' as const } } },
          },
        ]

        const messages = [new Message({ role: 'user', content: [new TextBlock(largeContext)] })]

        // First request - writes to cache
        const events1 = await collectIterator(provider.stream(messages, { toolSpecs }))
        const metadata1 = events1.find((e) => e.type === 'modelMetadataEvent')
        expect(metadata1?.usage?.cacheWriteInputTokens).toBeGreaterThan(0)

        // Second request - identical content, should read from cache
        const events2 = await collectIterator(provider.stream(messages, { toolSpecs }))
        const metadata2 = events2.find((e) => e.type === 'modelMetadataEvent')
        expect(metadata2?.usage?.cacheReadInputTokens).toBeGreaterThan(0)
      })

      it.concurrent(
        'uses cacheConfig with explicit anthropic strategy for application inference profiles',
        async () => {
          const provider = bedrock.createModel({
            modelId: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            maxTokens: 100,
            cacheConfig: { strategy: 'anthropic' },
          })
          const largeContext = `Context information: ${'hello '.repeat(2000)} [test-${Date.now()}-${Math.random()}]`

          const messages = [new Message({ role: 'user', content: [new TextBlock(largeContext)] })]

          // First request - writes to cache
          const events1 = await collectIterator(provider.stream(messages))
          const metadata1 = events1.find((e) => e.type === 'modelMetadataEvent')
          expect(metadata1?.usage?.cacheWriteInputTokens).toBeGreaterThan(0)

          // Second request - identical content, should read from cache
          const events2 = await collectIterator(provider.stream(messages))
          const metadata2 = events2.find((e) => e.type === 'modelMetadataEvent')
          expect(metadata2?.usage?.cacheReadInputTokens).toBeGreaterThan(0)
        }
      )
    })

    describe('Error Handling', () => {
      it.concurrent('handles invalid model ID gracefully', async () => {
        const provider = bedrock.createModel({ modelId: 'invalid-model-id-that-does-not-exist' })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Hello')] })]
        await expect(collectIterator(provider.stream(messages))).rejects.toThrow()
      })
    })
  })

  describe('Agent with Conversation Manager', () => {
    it('manages conversation history with SlidingWindowConversationManager', async () => {
      const agent = new Agent({
        model: bedrock.createModel({ maxTokens: 100 }),
        conversationManager: new SlidingWindowConversationManager({ windowSize: 4 }),
      })

      // First exchange
      await agent.invoke('Count from 1 to 1.')
      expect(agent.messages).toHaveLength(2) // user + assistant

      // Second exchange
      await agent.invoke('Count from 2 to 2.')
      expect(agent.messages).toHaveLength(4) // 2 user + 2 assistant

      // Third exchange - should trigger sliding window
      await agent.invoke('Count from 3 to 3.')

      // Should maintain window size of 4 messages
      expect(agent.messages).toHaveLength(4)
    }, 30000)

    it('throws ContextWindowOverflowError with NullConversationManager', async () => {
      const agent = new Agent({
        model: bedrock.createModel({ maxTokens: 50 }),
        conversationManager: new NullConversationManager(),
      })

      // Generate a message that would require context management
      const longPrompt = 'Please write a very detailed explanation of ' + 'many topics '.repeat(50)

      // This should throw since NullConversationManager doesn't handle overflow
      await expect(agent.invoke(longPrompt)).rejects.toThrow()
    }, 30000)
  })

  describe('Region Configuration', () => {
    it('uses explicit region when provided', async () => {
      const provider = bedrock.createModel({
        region: 'us-east-1',
        maxTokens: 50,
      })

      // Validate region configuration by checking config.region() directly
      // Making an actual request doesn't guarantee the correct region is being used
      const regionResult = await provider['_client'].config.region()
      expect(regionResult).toBe('us-east-1')
    })

    it('uses region from clientConfig when provided', async () => {
      const provider = bedrock.createModel({
        clientConfig: { region: 'ap-northeast-1' },
        maxTokens: 50,
      })

      // Validate clientConfig region is used
      // Making an actual request doesn't guarantee the correct region is being used
      const regionResult = await provider['_client'].config.region()
      expect(regionResult).toBe('ap-northeast-1')
    })

    it('defaults to us-west-2 when no region provided and AWS SDK does not resolve one', async () => {
      // Use vitest to stub environment variables
      vi.stubEnv('AWS_REGION', undefined)
      vi.stubEnv('AWS_DEFAULT_REGION', undefined)
      // Point config and credential files to null values
      vi.stubEnv('AWS_CONFIG_FILE', '/dev/null')
      vi.stubEnv('AWS_SHARED_CREDENTIALS_FILE', '/dev/null')

      const provider = bedrock.createModel({
        maxTokens: 50,
      })

      // Validate region defaults to us-west-2
      // Making an actual request doesn't guarantee the correct region is being used
      const regionResult = await provider['_client'].config.region()
      expect(regionResult).toBe('us-west-2')

      // ensure that invocation works
      await collectIterator(
        provider.stream([
          Message.fromMessageData({
            role: 'user',
            content: [new TextBlock('say hi')],
          }),
        ])
      )
    })

    it('uses region from clientConfig when provided', async () => {
      const provider = bedrock.createModel({
        clientConfig: { region: 'ap-northeast-1' },
        maxTokens: 50,
      })

      // Validate clientConfig region is used
      // Making an actual request doesn't guarantee the correct region is being used
      const regionResult = await provider['_client'].config.region()
      expect(regionResult).toBe('ap-northeast-1')
    })
  })

  describe('Thinking Mode with Tools', () => {
    it('handles thinking mode with tool use', async () => {
      const bedrockModel = bedrock.createModel({
        modelId: 'global.anthropic.claude-sonnet-4-6',
        additionalRequestFields: {
          thinking: {
            type: 'enabled',
            budget_tokens: 1024,
          },
        },
        maxTokens: 2048,
      })

      const testTool = new FunctionTool({
        name: 'testTool',
        description: 'Test description',
        inputSchema: { type: 'object' },
        callback: (): string => 'result',
      })

      // Create agent with thinking mode and tool
      const agent = new Agent({
        model: bedrockModel,
        tools: [testTool],
        printer: false,
      })

      // Invoke agent with a prompt that triggers tool use
      const result = await agent.invoke('Use the testTool with the message "Hello World"')

      // Verify the agent completed successfully
      expect(result.stopReason).toBe('endTurn')
      expect(result.lastMessage.role).toBe('assistant')
      expect(result.lastMessage.content.length).toBeGreaterThan(0)

      // Verify the tool was used
      const toolUseMessage = agent.messages.find((msg) => msg.content.some((block) => block.type === 'toolUseBlock'))
      expect(toolUseMessage).toBeDefined()

      // Verify the tool result is in the history
      const toolResultMessage = agent.messages.find((msg) =>
        msg.content.some((block) => block.type === 'toolResultBlock')
      )
      expect(toolResultMessage).toBeDefined()
    }, 30000)
  })

  describe('Guardrails', () => {
    const BLOCKED_INPUT = 'BLOCKED_INPUT'
    const BLOCKED_OUTPUT = 'BLOCKED_OUTPUT'
    const GUARDRAIL_NAME = 'test-guardrail-block-cactus'

    let GUARDRAIL_ID: string | undefined

    /**
     * Gets the guardrail ID by name if it exists
     */
    async function getGuardrailId(client: BedrockClient, guardrailName: string): Promise<string | undefined> {
      const response = await client.send(new ListGuardrailsCommand({}))
      const guardrail = response.guardrails?.find((g) => g.name === guardrailName)
      return guardrail?.id
    }

    /**
     * Waits for the guardrail to become active
     */
    async function waitForGuardrailActive(
      client: BedrockClient,
      guardrailId: string,
      maxAttempts = 10,
      delayMs = 5000
    ): Promise<void> {
      for (let i = 0; i < maxAttempts; i++) {
        const response = await client.send(new GetGuardrailCommand({ guardrailIdentifier: guardrailId }))
        const status = response.status

        if (status === 'READY') {
          console.log(`Guardrail ${guardrailId} is now active`)
          return
        }

        console.log(`Waiting for guardrail to become active. Current status: ${status}`)
        await new Promise((resolve) => setTimeout(resolve, delayMs))
      }

      throw new Error(`Guardrail did not become active within ${(maxAttempts * delayMs) / 1000} seconds`)
    }

    /**
     * Creates or retrieves the test guardrail
     */
    async function setupGuardrail(): Promise<string> {
      const credentials = inject('provider-bedrock')?.credentials
      if (!credentials) {
        throw new Error('No Bedrock credentials provided')
      }

      const client = new BedrockClient({ region: 'us-east-1', credentials })

      // Check if guardrail already exists
      let guardrailId = await getGuardrailId(client, GUARDRAIL_NAME)

      if (guardrailId) {
        console.log(`Guardrail ${GUARDRAIL_NAME} already exists with ID: ${guardrailId}`)
      } else {
        console.log(`Creating guardrail ${GUARDRAIL_NAME}`)
        const response = await client.send(
          new CreateGuardrailCommand({
            name: GUARDRAIL_NAME,
            description: 'Testing Guardrail',
            wordPolicyConfig: {
              wordsConfig: [
                {
                  text: 'CACTUS',
                },
              ],
            },
            blockedInputMessaging: BLOCKED_INPUT,
            blockedOutputsMessaging: BLOCKED_OUTPUT,
          })
        )
        guardrailId = response.guardrailId
        if (!guardrailId) {
          throw new Error('Failed to create guardrail: no ID returned')
        }
        console.log(`Created test guardrail with ID: ${guardrailId}`)
        await waitForGuardrailActive(client, guardrailId)
      }

      if (!guardrailId) {
        throw new Error('Failed to get or create guardrail')
      }

      return guardrailId
    }

    beforeAll(async () => {
      GUARDRAIL_ID = await setupGuardrail()
    }, 60000)

    describe('Input Intervention', () => {
      it.each(['enabled', 'enabled_full'] as const)(
        'blocks input and redacts message with trace=%s',
        async (guardrailTrace) => {
          const model = bedrock.createModel({
            region: 'us-east-1',
            guardrailConfig: {
              guardrailIdentifier: GUARDRAIL_ID!,
              guardrailVersion: 'DRAFT',
              trace: guardrailTrace,
              redaction: {
                input: true,
                inputMessage: 'Redacted.',
              },
            },
          })

          const agent = new Agent({
            model,
            systemPrompt: 'You are a helpful assistant.',
            printer: false,
          })

          const response1 = await agent.invoke('CACTUS')
          const response2 = await agent.invoke('Hello!')

          expect(response1.stopReason).toBe('guardrailIntervened')
          expect(response1.toString().trim()).toBe(BLOCKED_INPUT)
          expect(response2.stopReason).not.toBe('guardrailIntervened')
          expect(response2.toString().trim()).not.toBe(BLOCKED_INPUT)
          expect(agent.messages[0]?.content[0]?.type).toBe('textBlock')
          const firstBlock = agent.messages[0]?.content[0]
          if (firstBlock?.type === 'textBlock') {
            expect(firstBlock.text).toBe('Redacted.')
          }
        },
        30000
      )
    })

    describe('Output Intervention', () => {
      it.each(['sync', 'async'] as const)(
        'blocks output without redaction in %s mode',
        async (processingMode) => {
          const model = bedrock.createModel({
            modelId: 'global.anthropic.claude-sonnet-4-5-20250929-v1:0',
            region: 'us-east-1',
            guardrailConfig: {
              guardrailIdentifier: GUARDRAIL_ID!,
              guardrailVersion: 'DRAFT',
              streamProcessingMode: processingMode,
              redaction: {
                output: false,
              },
            },
          })

          const agent = new Agent({
            model,
            systemPrompt: 'When asked to say the word, say CACTUS.',
            printer: false,
          })

          const response1 = await agent.invoke('Say the word.')
          const response2 = await agent.invoke('Hello!')

          expect(response1.stopReason).toBe('guardrailIntervened')

          if (processingMode === 'sync') {
            // In sync mode, we can reliably check the response content
            expect(response1.toString()).toContain(BLOCKED_OUTPUT)
            expect(response2.stopReason).not.toBe('guardrailIntervened')
            expect(response2.toString()).not.toContain(BLOCKED_OUTPUT)
          } else {
            // In async mode, either:
            // - CACTUS was returned and blocked by input guardrail on next turn, or
            // - CACTUS was blocked in response1, allowing normal response2
            const cactusCaughtByInputGuardrail = response2.toString().includes(BLOCKED_INPUT)
            const cactusBlockedAllowsNextResponse =
              !response2.toString().includes(BLOCKED_OUTPUT) && response2.stopReason !== 'guardrailIntervened'
            expect(cactusCaughtByInputGuardrail || cactusBlockedAllowsNextResponse).toBe(true)
          }
        },
        30000
      )

      it.each([
        ['sync', 'enabled'],
        ['sync', 'enabled_full'],
        ['async', 'enabled'],
        ['async', 'enabled_full'],
      ] as const)(
        'blocks output with redaction in %s mode with trace=%s',
        async (processingMode, guardrailTrace) => {
          const REDACT_MESSAGE = 'Redacted.'
          const model = bedrock.createModel({
            region: 'us-east-1',
            guardrailConfig: {
              guardrailIdentifier: GUARDRAIL_ID!,
              guardrailVersion: 'DRAFT',
              streamProcessingMode: processingMode,
              trace: guardrailTrace,
              redaction: {
                output: true,
                outputMessage: REDACT_MESSAGE,
              },
            },
            temperature: 0, // Deterministic responses
          })

          const agent = new Agent({
            model,
            systemPrompt: 'When asked to say the word, say CACTUS. Otherwise, respond normally.',
            printer: false,
          })

          const response1 = await agent.invoke('Say the word.')
          // Use unrelated prompt to avoid model volunteering CACTUS
          const response2 = await agent.invoke('What is 2+2? Reply with only the number.')

          expect(response1.stopReason).toBe('guardrailIntervened')

          if (processingMode === 'sync') {
            expect(response1.toString()).toContain(REDACT_MESSAGE)
            expect(response2.stopReason).not.toBe('guardrailIntervened')
            expect(response2.toString()).not.toContain(REDACT_MESSAGE)
          } else {
            // In async mode, either:
            // - CACTUS was returned and blocked by input guardrail on next turn, or
            // - CACTUS was blocked in response1, allowing normal response2
            const cactusCaughtByInputGuardrail = response2.toString().includes(BLOCKED_INPUT)
            const cactusBlockedAllowsNextResponse =
              !response2.toString().includes(REDACT_MESSAGE) && response2.stopReason !== 'guardrailIntervened'
            expect(cactusCaughtByInputGuardrail || cactusBlockedAllowsNextResponse).toBe(true)
          }
        },
        30000
      )

      it('captures redactedContent from modelOutput in sync mode', async () => {
        const REDACT_MESSAGE = 'Content blocked.'
        const model = bedrock.createModel({
          region: 'us-east-1',
          guardrailConfig: {
            guardrailIdentifier: GUARDRAIL_ID!,
            guardrailVersion: 'DRAFT',
            streamProcessingMode: 'sync',
            trace: 'enabled_full', // Need full trace to get modelOutput
            redaction: {
              output: true,
              outputMessage: REDACT_MESSAGE,
            },
          },
          temperature: 0,
        })

        const messages = [new Message({ role: 'user', content: [new TextBlock('Say CACTUS.')] })]

        // Collect streaming events to check for redactedContent
        const events: any[] = []
        for await (const event of model.stream(messages)) {
          events.push(event)
        }

        // Find the ModelRedactionEvent with outputRedaction
        const redactEvent = events.find((e) => e.type === 'modelRedactionEvent' && e.outputRedaction) as
          | ModelRedactionEvent
          | undefined

        expect(redactEvent).toBeDefined()
        expect(redactEvent?.outputRedaction?.replaceContent).toBe(REDACT_MESSAGE)

        // In sync mode with full trace, we should get the original content
        // The exact content may vary, but if blocked, redactedContent should be present
        if (redactEvent?.outputRedaction?.redactedContent) {
          expect(redactEvent.outputRedaction.redactedContent).toContain('CACTUS')
        }
      }, 30000)
    })

    describe('Tool Result Redaction', () => {
      it.each(['sync', 'async'] as const)(
        'properly redacts tool result in %s mode',
        async (processingMode) => {
          const INPUT_REDACT_MESSAGE = 'Input redacted.'
          const OUTPUT_REDACT_MESSAGE = 'Output redacted.'

          const model = bedrock.createModel({
            region: 'us-east-1',
            guardrailConfig: {
              guardrailIdentifier: GUARDRAIL_ID!,
              guardrailVersion: 'DRAFT',
              streamProcessingMode: processingMode,
              redaction: {
                input: true,
                inputMessage: INPUT_REDACT_MESSAGE,
                output: true,
                outputMessage: OUTPUT_REDACT_MESSAGE,
              },
            },
          })

          const listUsers = new FunctionTool({
            name: 'list_users',
            description: 'List my users',
            inputSchema: { type: 'object', properties: {} },
            callback: async () => {
              return '[{"name": "Jerry Merry"}, {"name": "Mr. CACTUS"}]'
            },
          })

          const agent = new Agent({
            model,
            systemPrompt: 'You are a helpful assistant.',
            tools: [listUsers],
            printer: false,
          })

          const response1 = await agent.invoke('List my users.')
          const response2 = await agent.invoke('Thank you!')

          /*
           * Message sequence:
           * 0 (user): request1
           * 1 (assistant): reasoning + tool call
           * 2 (user): tool result
           * 3 (assistant): response1 -> output guardrail intervenes
           * 4 (user): request2
           * 5 (assistant): response2
           *
           * Guardrail intervened on output in message 3 will cause
           * the redaction of the preceding input (message 2) and message 3.
           */

          expect(response1.stopReason).toBe('guardrailIntervened')

          if (processingMode === 'sync') {
            // In sync mode the guardrail processing is blocking
            expect(response1.toString()).toContain(OUTPUT_REDACT_MESSAGE)
            expect(response2.toString()).not.toContain(OUTPUT_REDACT_MESSAGE)
          }

          // In both sync and async with output redaction:
          // 1. Content should be properly redacted so response2 is not blocked
          expect(response2.stopReason).not.toBe('guardrailIntervened')

          // 2. Tool result block should be redacted properly
          const toolUseMessage = agent.messages[1]
          const toolResultMessage = agent.messages[2]

          expect(toolUseMessage).toBeDefined()
          expect(toolResultMessage).toBeDefined()

          const toolUseBlock = toolUseMessage?.content.find((b) => b.type === 'toolUseBlock')
          const toolResultBlock = toolResultMessage?.content.find((b) => b.type === 'toolResultBlock')

          expect(toolUseBlock).toBeDefined()
          expect(toolResultBlock).toBeDefined()

          if (toolUseBlock?.type === 'toolUseBlock' && toolResultBlock?.type === 'toolResultBlock') {
            expect(toolResultBlock.toolUseId).toBe(toolUseBlock.toolUseId)
            const firstContent = toolResultBlock.content[0]
            expect(firstContent).toBeDefined()
            if (firstContent?.type === 'textBlock') {
              expect((firstContent as TextBlock).text).toBe(INPUT_REDACT_MESSAGE)
            }
          }
        },
        30000
      )
    })

    describe('guardLatestUserMessage', () => {
      it('allows conversation when latest user message is clean even if earlier messages would trigger guardrails', async () => {
        // Load test image
        const imageBytes = await loadFixture(yellowPngUrl)

        // Create model with guardLatestUserMessage enabled
        const model = bedrock.createModel({
          region: 'us-east-1',
          guardrailConfig: {
            guardrailIdentifier: GUARDRAIL_ID!,
            guardrailVersion: 'DRAFT',
            guardLatestUserMessage: true,
          },
        })

        // Create agent with previous messages that CONTAIN blocked content (CACTUS)
        // When guardLatestUserMessage is enabled, these earlier messages should NOT trigger the guardrail
        const agent = new Agent({
          model,
          printer: false,
          systemPrompt: 'You are a helpful assistant.',
          messages: [
            new Message({
              role: 'user',
              content: [
                new TextBlock('Dont Say CACTUS'),
                new ImageBlock({ format: 'png', source: { bytes: imageBytes } }),
              ],
            }),
            new Message({ role: 'assistant', content: [new TextBlock('Hello!')] }),
          ],
        })

        // Send a clean message - should NOT trigger guardrail because only the latest message is evaluated
        const response = await agent.invoke('Hello!')

        expect(response.stopReason).not.toBe('guardrailIntervened')
      }, 30000)

      it('blocks conversation when latest user message contains blocked content', async () => {
        // Create model with guardLatestUserMessage enabled
        const model = bedrock.createModel({
          region: 'us-east-1',
          guardrailConfig: {
            guardrailIdentifier: GUARDRAIL_ID!,
            guardrailVersion: 'DRAFT',
            guardLatestUserMessage: true,
          },
        })

        // Send message with blocked content
        const agent = new Agent({
          model,
          printer: false,
          systemPrompt: 'You are a helpful assistant.',
        })

        const response = await agent.invoke('Tell me about CACTUS plants')

        // The guardrail should have intervened
        expect(response.stopReason).toBe('guardrailIntervened')
        expect(response.toString()).toContain(BLOCKED_INPUT)
      }, 30000)
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
      const model = bedrock.createModel()
      const result = await model.countTokens(messages)
      expect(typeof result).toBe('number')
      expect(result).toBeGreaterThan(0)
    })

    it.concurrent('should return more tokens with tools and system prompt', async () => {
      const model = bedrock.createModel()
      const without = await model.countTokens(messages)
      const withTools = await model.countTokens(messages, { toolSpecs, systemPrompt: 'Be helpful.' })
      expect(withTools).toBeGreaterThan(without)
    })
  })
})
