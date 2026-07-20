import { describe, expect, it } from 'vitest'
import { z } from 'zod'
import { tool } from '../../tools/tool-factory.js'
import {
  CachePointBlock,
  GuardContentBlock,
  JsonBlock,
  Message,
  ReasoningBlock,
  TextBlock,
  ToolResultBlock,
  ToolUseBlock,
} from '../../types/messages.js'
import { DocumentBlock, ImageBlock, VideoBlock } from '../../types/media.js'
import { CitationsBlock } from '../../types/citations.js'
import { CheckpointError } from '../../errors.js'
import { formatHarnessInput, formatHarnessSystemPrompt, formatHarnessTools } from '../request-formatting.js'
import type { InvokeArgs } from '../../types/agent.js'
import type { ContentBlock, ToolResultContent } from '../../types/messages.js'

describe('formatHarnessInput', () => {
  it.each([
    { desc: 'string', args: 'Hello', expected: [{ text: 'Hello' }] },
    { desc: 'ContentBlock[]', args: [new TextBlock('From block')], expected: [{ text: 'From block' }] },
    { desc: 'ContentBlockData[]', args: [{ text: 'From data' }], expected: [{ text: 'From data' }] },
    {
      desc: 'Message[]',
      args: [new Message({ role: 'user', content: [new TextBlock('From message')] })],
      expected: [{ text: 'From message' }],
    },
    {
      desc: 'MessageData[]',
      args: [{ role: 'user', content: [{ text: 'From message data' }] }],
      expected: [{ text: 'From message data' }],
    },
  ])('normalizes $desc input into Harness messages', ({ args, expected }) => {
    expect(formatHarnessInput(args as InvokeArgs)).toStrictEqual([{ role: 'user', content: expected }])
  })

  it.each([
    ['an empty content array', []],
    ['an empty string', ''],
    ['an empty message', [new Message({ role: 'user', content: [] })]],
    ['an empty text block', [new TextBlock('')]],
  ])('rejects %s', (_description, args) => {
    expect(() => formatHarnessInput(args as InvokeArgs)).toThrow()
  })

  it('rejects interrupt-response input', () => {
    expect(() => formatHarnessInput([{ interruptResponse: { interruptId: 'i1', response: 'ok' } }])).toThrow(
      /interrupt-response/
    )
  })

  it('rejects checkpoint-resume input', () => {
    expect(() =>
      formatHarnessInput({
        checkpointResume: {
          checkpoint: { position: 'afterModel', cycleIndex: 0, schemaVersion: '1.0' },
        },
      })
    ).toThrow(
      new CheckpointError('Received a checkpointResume block but AgentCoreHarnessAgent does not support checkpointing.')
    )
  })

  it('formats every supported message content variant', () => {
    const redactedContent = new Uint8Array([1, 2, 3])

    expect(
      formatHarnessInput([
        new TextBlock('hello'),
        new ToolUseBlock({ toolUseId: 'tu-1', name: 'lookup', input: { query: 'weather' } }),
        new ToolResultBlock({
          toolUseId: 'tu-1',
          status: 'success',
          content: [new TextBlock('sunny'), new JsonBlock({ json: { temperature: 72 } })],
        }),
        new ReasoningBlock({ text: 'thinking', signature: 'sig-1' }),
        new ReasoningBlock({ redactedContent }),
      ])
    ).toStrictEqual([
      {
        role: 'user',
        content: [
          { text: 'hello' },
          {
            toolUse: {
              toolUseId: 'tu-1',
              name: 'lookup',
              input: { query: 'weather' },
              type: 'tool_use',
            },
          },
          {
            toolResult: {
              toolUseId: 'tu-1',
              content: [{ text: 'sunny' }, { text: '{"temperature":72}' }],
              status: 'success',
              type: 'tool_use',
            },
          },
          { reasoningContent: { reasoningText: { text: 'thinking', signature: 'sig-1' } } },
          { reasoningContent: { redactedContent } },
        ],
      },
    ])
  })

  const unsupportedContentBlocks: { description: string; block: ContentBlock }[] = [
    {
      description: 'cache point',
      block: new CachePointBlock({ cacheType: 'default' }),
    },
    {
      description: 'guard content',
      block: new GuardContentBlock({ text: { text: 'guard this', qualifiers: ['guard_content'] } }),
    },
    {
      description: 'image',
      block: new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1]) } }),
    },
    {
      description: 'video',
      block: new VideoBlock({ format: 'mp4', source: { bytes: new Uint8Array([2]) } }),
    },
    {
      description: 'document',
      block: new DocumentBlock({ name: 'notes', format: 'txt', source: { text: 'hello' } }),
    },
    {
      description: 'citations',
      block: new CitationsBlock({ citations: [], content: [{ text: 'cited answer' }] }),
    },
  ]

  it.each(unsupportedContentBlocks)('rejects unsupported $description content without dropping it', ({ block }) => {
    expect(() => formatHarnessInput([new TextBlock('keep me'), block])).toThrow(
      `Content block at index 1 has unsupported type '${block.type}'.`
    )
  })

  it('rejects an unknown runtime content block type', () => {
    expect(() => formatHarnessInput([{ type: 'futureBlock' } as never])).toThrow(
      "Content block at index 0 has unknown type 'futureBlock'."
    )
  })

  it.each([
    {
      description: 'no content',
      block: new ReasoningBlock({}),
      message: 'must contain text, a signature, or redacted content',
    },
    {
      description: 'empty text',
      block: new ReasoningBlock({ text: '' }),
      message: 'contains empty reasoning text',
    },
    {
      description: 'text and redacted content',
      block: new ReasoningBlock({ text: 'thinking', redactedContent: new Uint8Array([1]) }),
      message: 'contains both reasoning text and redacted content',
    },
  ])('rejects reasoning content with $description', ({ block, message }) => {
    expect(() => formatHarnessInput([block])).toThrow(message)
  })

  it('rejects a tool-use reasoning signature that the Harness cannot represent', () => {
    expect(() =>
      formatHarnessInput([
        new ToolUseBlock({
          toolUseId: 'tu-1',
          name: 'lookup',
          input: {},
          reasoningSignature: 'signature',
        }),
      ])
    ).toThrow('has a reasoningSignature, which AgentCore Harness cannot represent')
  })

  const unsupportedToolResultContent: { description: string; content: ToolResultContent }[] = [
    {
      description: 'image',
      content: new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1]) } }),
    },
    {
      description: 'video',
      content: new VideoBlock({ format: 'mp4', source: { bytes: new Uint8Array([2]) } }),
    },
    {
      description: 'document',
      content: new DocumentBlock({ name: 'notes', format: 'txt', source: { text: 'hello' } }),
    },
  ]

  it.each(unsupportedToolResultContent)('rejects unsupported $description tool-result content', ({ content }) => {
    expect(() =>
      formatHarnessInput([
        new ToolResultBlock({
          toolUseId: 'tu-1',
          status: 'success',
          content: [new TextBlock('keep me'), content],
        }),
      ])
    ).toThrow(`item index 1 has unsupported type '${content.type}'`)
  })

  it('rejects an empty tool result', () => {
    expect(() =>
      formatHarnessInput([new ToolResultBlock({ toolUseId: 'tu-1', status: 'success', content: [] })])
    ).toThrow('must contain at least one result item')
  })

  it('rejects an unknown runtime tool-result content type', () => {
    expect(() =>
      formatHarnessInput([
        new ToolResultBlock({
          toolUseId: 'tu-1',
          status: 'success',
          content: [{ type: 'futureToolResult' } as never],
        }),
      ])
    ).toThrow("item index 0 has unknown type 'futureToolResult'")
  })

  it('serializes tool-result content to non-empty text', () => {
    expect(
      formatHarnessInput([
        new ToolResultBlock({
          toolUseId: 'tu-1',
          status: 'success',
          content: [new TextBlock(''), new JsonBlock({ json: { a: 1 } })],
        }),
      ])
    ).toStrictEqual([
      {
        role: 'user',
        content: [
          {
            toolResult: {
              toolUseId: 'tu-1',
              content: [{ text: '""' }, { text: '{"a":1}' }],
              status: 'success',
              type: 'tool_use',
            },
          },
        ],
      },
    ])
  })
})

describe('formatHarnessTools', () => {
  it('formats host tools as inline functions', () => {
    const hostTool = tool({
      name: 'get_weather',
      description: 'Get weather',
      inputSchema: z.object({ city: z.string() }),
      callback: () => 'sunny',
    })

    expect(formatHarnessTools([hostTool])).toStrictEqual([
      {
        type: 'inline_function',
        name: 'get_weather',
        config: {
          inlineFunction: {
            description: 'Get weather',
            inputSchema: {
              type: 'object',
              properties: { city: { type: 'string' } },
              required: ['city'],
              additionalProperties: false,
            },
          },
        },
      },
    ])
  })
})

describe('formatHarnessSystemPrompt', () => {
  it('formats strings and non-empty text blocks', () => {
    expect(formatHarnessSystemPrompt('Be concise.')).toStrictEqual([{ text: 'Be concise.' }])
    expect(formatHarnessSystemPrompt([new TextBlock('Be concise.'), new TextBlock('Be terse.')])).toStrictEqual([
      { text: 'Be concise.' },
      { text: 'Be terse.' },
    ])
  })

  it.each(['', []])('rejects an empty prompt %#', (prompt) => {
    expect(() => formatHarnessSystemPrompt(prompt as string | TextBlock[])).toThrow(
      'systemPrompt must contain non-empty text when provided'
    )
  })

  it('rejects an empty block instead of partially sending a mixed prompt', () => {
    expect(() => formatHarnessSystemPrompt([new TextBlock('Keep me'), new TextBlock('')])).toThrow(
      'systemPrompt block at index 1 must contain non-empty text'
    )
  })
})
