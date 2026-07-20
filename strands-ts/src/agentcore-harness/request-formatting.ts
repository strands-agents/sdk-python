import type {
  HarnessContentBlock,
  HarnessInlineFunctionConfig,
  HarnessMessage,
  HarnessSystemContentBlock,
  HarnessTool,
  HarnessToolResultContentBlock,
} from '@aws-sdk/client-bedrock-agentcore'
import { CheckpointError } from '../errors.js'
import { isInterruptResponseContent } from '../types/interrupt.js'
import { Message, TextBlock } from '../types/messages.js'
import type { InvokeArgs } from '../types/agent.js'
import type { ContentBlock, ContentBlockData, MessageData, ToolResultContent } from '../types/messages.js'
import type { Tool } from '../tools/tool.js'

/**
 * Converts invocation input into Harness messages.
 *
 * @internal
 * @param args - Arguments supplied to the agent invocation
 * @returns Harness messages suitable for an InvokeHarness request
 */
export function formatHarnessInput(args: InvokeArgs): HarnessMessage[] {
  const messages = normalizeArgs(args)
  if (messages.length === 0) {
    throw new TypeError('AgentCoreHarnessAgent input must contain at least one message.')
  }
  return messages.map((message) => formatHarnessMessage(message))
}

/**
 * Converts a Strands message into a Harness message.
 *
 * @internal
 * @param message - Strands message to convert
 * @returns Harness wire message
 */
export function formatHarnessMessage(message: Message): HarnessMessage {
  if (message.content.length === 0) {
    throw new TypeError(`AgentCore Harness ${message.role} messages must contain at least one content block.`)
  }
  const content = message.content.map((block, index) => formatContentBlock(block, index))
  return { role: message.role, content }
}

/**
 * Converts host tools into Harness inline-function definitions.
 *
 * @internal
 * @param tools - Host tools to expose to the Harness
 * @returns Harness tool definitions
 */
export function formatHarnessTools(tools: Tool[]): HarnessTool[] {
  return tools.map((tool): HarnessTool => {
    const inlineFunction: HarnessInlineFunctionConfig = {
      description: tool.toolSpec.description,
      // Strands input schemas are JSON Schema, a valid AWS document value at runtime. The smithy
      // DocumentType is recursive and JSONSchema7 does not match it structurally, so bridge via unknown.
      inputSchema: (tool.toolSpec.inputSchema ?? {
        type: 'object',
        properties: {},
      }) as unknown as HarnessInlineFunctionConfig['inputSchema'],
    }
    return { type: 'inline_function', name: tool.name, config: { inlineFunction } }
  })
}

/**
 * Converts a system prompt override into Harness system content blocks.
 *
 * @internal
 * @param systemPrompt - System prompt override
 * @returns Harness system content blocks, or undefined when omitted
 */
export function formatHarnessSystemPrompt(
  systemPrompt: string | TextBlock[] | undefined
): HarnessSystemContentBlock[] | undefined {
  if (systemPrompt === undefined) return undefined
  if (typeof systemPrompt === 'string') {
    if (systemPrompt.length === 0) {
      throw new TypeError('systemPrompt must contain non-empty text when provided')
    }
    return [{ text: systemPrompt }]
  }
  if (systemPrompt.length === 0) {
    throw new TypeError('systemPrompt must contain non-empty text when provided')
  }
  return systemPrompt.map((block, index) => {
    if (block.text.length === 0) {
      throw new TypeError(`systemPrompt block at index ${index} must contain non-empty text`)
    }
    return { text: block.text }
  })
}

/** Normalizes invocation arguments into Strands messages. */
function normalizeArgs(args: InvokeArgs): Message[] {
  if (typeof args === 'string') {
    return [new Message({ role: 'user', content: [new TextBlock(args)] })]
  }
  if (typeof args === 'object' && args !== null && !Array.isArray(args) && 'checkpointResume' in args) {
    throw new CheckpointError(
      'Received a checkpointResume block but AgentCoreHarnessAgent does not support checkpointing.'
    )
  }
  if (!Array.isArray(args) || args.length === 0) {
    return []
  }
  const first = args[0]!
  // Interrupt responses resume a local Agent's paused loop; the Harness runs its own loop and
  // has no resume-from-interrupt concept. Host callbacks provide the Harness's approval path.
  if (isInterruptResponseContent(first)) {
    throw new Error(
      'AgentCoreHarnessAgent does not support interrupt-response inputs. For human-in-the-loop, pass a custom tool whose callback performs the approval; the harness pauses on the tool call and resumes with the result.'
    )
  }
  if (typeof first === 'object' && 'role' in first) {
    return (args as (Message | MessageData)[]).map((message) =>
      message instanceof Message ? message : Message.fromMessageData(message)
    )
  }
  if (typeof first === 'object' && 'type' in first) {
    return [new Message({ role: 'user', content: args as ContentBlock[] })]
  }
  return [Message.fromMessageData({ role: 'user', content: args as ContentBlockData[] })]
}

/** Converts a Strands content block into a Harness content block. */
function formatContentBlock(block: ContentBlock, index: number): HarnessContentBlock {
  switch (block.type) {
    case 'textBlock':
      if (block.text.length === 0) {
        throw new TypeError(`Content block at index ${index} contains empty text, which AgentCore Harness rejects.`)
      }
      return { text: block.text }
    case 'toolUseBlock':
      if (block.reasoningSignature !== undefined) {
        throw new TypeError(
          `Tool-use content block at index ${index} has a reasoningSignature, which AgentCore Harness cannot represent.`
        )
      }
      return { toolUse: { toolUseId: block.toolUseId, name: block.name, input: block.input, type: 'tool_use' } }
    case 'toolResultBlock': {
      if (block.content.length === 0) {
        throw new TypeError(`Tool-result content block at index ${index} must contain at least one result item.`)
      }
      return {
        toolResult: {
          toolUseId: block.toolUseId,
          content: block.content.map((content, contentIndex) => formatToolResultContent(content, index, contentIndex)),
          status: block.status,
          type: 'tool_use',
        },
      }
    }
    case 'reasoningBlock': {
      const hasReasoningText = block.text !== undefined || block.signature !== undefined
      const hasRedactedContent = block.redactedContent !== undefined
      if (hasReasoningText && hasRedactedContent) {
        throw new TypeError(
          `Reasoning content block at index ${index} contains both reasoning text and redacted content, which AgentCore Harness cannot represent together.`
        )
      }
      if (hasReasoningText) {
        if (block.text === '' && block.signature === undefined) {
          throw new TypeError(`Reasoning content block at index ${index} contains empty reasoning text.`)
        }
        return {
          reasoningContent: {
            reasoningText: {
              text: block.text ?? '',
              ...(block.signature !== undefined && { signature: block.signature }),
            },
          },
        }
      }
      if (hasRedactedContent) {
        return { reasoningContent: { redactedContent: block.redactedContent } }
      }
      throw new TypeError(
        `Reasoning content block at index ${index} must contain text, a signature, or redacted content.`
      )
    }
    case 'cachePointBlock':
    case 'guardContentBlock':
    case 'imageBlock':
    case 'videoBlock':
    case 'documentBlock':
    case 'citationsBlock':
      throw new TypeError(
        `Content block at index ${index} has unsupported type '${block.type}'. AgentCore Harness supports only text, tool-use, tool-result, and reasoning content.`
      )
    default:
      throw new TypeError(
        `Content block at index ${index} has unknown type '${String((block as { type?: unknown }).type)}'.`
      )
  }
}

/** Converts tool-result content into its Harness request representation. */
function formatToolResultContent(
  content: ToolResultContent,
  blockIndex: number,
  contentIndex: number
): HarnessToolResultContentBlock {
  switch (content.type) {
    case 'textBlock':
      // Harness text must be non-empty; JSON string syntax preserves an empty string value.
      return { text: content.text.length > 0 ? content.text : '""' }
    case 'jsonBlock':
      // The live Harness request path rejects JSON results despite the generated union member.
      return { text: JSON.stringify(content.json) }
    case 'imageBlock':
    case 'videoBlock':
    case 'documentBlock':
      throw new TypeError(
        `Tool-result content at block index ${blockIndex}, item index ${contentIndex} has unsupported type '${content.type}'. AgentCore Harness host tools can return only text or JSON-compatible values.`
      )
    default:
      throw new TypeError(
        `Tool-result content at block index ${blockIndex}, item index ${contentIndex} has unknown type '${String((content as { type?: unknown }).type)}'.`
      )
  }
}
