import type { JSONValue } from '../types/json.js'
import { Message, TextBlock, ToolResultBlock, ToolUseBlock } from '../types/messages.js'
import { unpinMessage } from '../conversation-manager/compression/pin-message.js'
import { rehydrateStoredToolResult, type StoredBackgroundTask } from './record.js'
import type { BackgroundTask } from './types.js'

export const BACKGROUND_RESULT_TOOL_NAME = 'strands_background_task_result'

export function assertDeliveryConsumed(
  taskId: string,
  expected: readonly Message[],
  modelRequestMessages: readonly Message[]
): void {
  const candidates = findBackgroundDeliveryPairs(modelRequestMessages, taskId)
  if (candidates.length === 0) {
    throw new Error(`Background task delivery '${taskId}' was not present in the provider request`)
  }
  if (!candidates.some((actual) => deliveriesMatch(actual, expected))) {
    throw new Error(`Background task delivery '${taskId}' did not match its authoritative record`)
  }
}

export function historyContainsBackgroundDelivery(messages: readonly Message[], record: StoredBackgroundTask): boolean {
  const expected = renderBackgroundDelivery(record)
  return findBackgroundDeliveryPairs(messages, record.taskId).some((actual) => deliveriesMatch(actual, expected))
}

export function renderBackgroundDelivery(record: StoredBackgroundTask): readonly [Message, Message] {
  if (record.status !== 'completed' && record.status !== 'failed' && record.status !== 'cancelled') {
    throw new Error(`Task '${record.taskId}' is not terminal`)
  }
  const failure = record.failure
  const input: JSONValue = {
    taskId: record.taskId,
    toolName: record.descriptor.toolName,
    status: record.status,
    ...(failure && {
      error: {
        type: failure.type,
        message: failure.message,
      },
    }),
  }
  const storedResult = record.result !== undefined ? rehydrateStoredToolResult(record.result) : undefined
  const resultContent = storedResult?.content ?? []
  const toolResult = new ToolResultBlock({
    toolUseId: record.taskId,
    status: record.status === 'completed' ? 'success' : 'error',
    content: [
      new TextBlock(
        renderTerminalHeader(
          record.taskId,
          record.descriptor.toolName,
          record.status,
          failure,
          storedResult !== undefined
        )
      ),
      ...resultContent,
    ],
  })

  return [
    new Message({
      role: 'assistant',
      content: [
        new ToolUseBlock({
          name: BACKGROUND_RESULT_TOOL_NAME,
          toolUseId: record.taskId,
          input,
        }),
      ],
      metadata: deliveryMetadata(),
    }),
    new Message({
      role: 'user',
      content: [toolResult],
      metadata: deliveryMetadata(),
    }),
  ]
}

export function unpinBackgroundDeliveries(messages: Message[], taskIds: ReadonlySet<string>): void {
  for (let index = 0; index < messages.length - 1; index++) {
    const taskId = backgroundDeliveryId(messages[index]!, messages[index + 1]!)
    if (!taskId || !taskIds.has(taskId)) continue
    unpinMessage(messages, index)
    unpinMessage(messages, index + 1)
  }
}

function renderTerminalHeader(
  taskId: string,
  toolName: string,
  status: Extract<BackgroundTask['status'], 'completed' | 'failed' | 'cancelled'>,
  error: { readonly type: string; readonly message: string } | undefined,
  hasResult: boolean
): string {
  if (status === 'completed') {
    return [
      'Background task completed.',
      '',
      `Task ID: ${taskId}`,
      `Tool: ${toolName}`,
      'Status: completed',
      '',
      'The final result follows.',
    ].join('\n')
  }
  if (status === 'failed') {
    if (!error) throw new Error(`Failed background task '${taskId}' has no failure detail`)
    return [
      'Background task failed.',
      '',
      `Task ID: ${taskId}`,
      `Tool: ${toolName}`,
      'Status: failed',
      `Error type: ${error.type}`,
      `Reason: ${error.message}`,
      '',
      hasResult ? 'The tool error follows.' : 'No result is available.',
    ].join('\n')
  }
  return [
    'Background task cancelled.',
    '',
    `Task ID: ${taskId}`,
    `Tool: ${toolName}`,
    'Status: cancelled',
    '',
    'The task was cancelled before producing a final result.',
  ].join('\n')
}

function findBackgroundDeliveryPairs(
  messages: readonly Message[],
  deliveryId: string
): readonly (readonly [Message, Message])[] {
  const pairs: [Message, Message][] = []
  for (let index = 0; index < messages.length - 1; index++) {
    const toolUseMessage = messages[index]!
    const toolResultMessage = messages[index + 1]!
    if (backgroundDeliveryId(toolUseMessage, toolResultMessage) === deliveryId) {
      pairs.push([toolUseMessage, toolResultMessage])
    }
  }
  return pairs
}

function backgroundDeliveryId(toolUseMessage: Message, toolResultMessage: Message): string | undefined {
  if (toolUseMessage.role !== 'assistant' || toolResultMessage.role !== 'user') return undefined
  const toolUse = toolUseMessage.content.find(
    (block) => block.type === 'toolUseBlock' && block.name === BACKGROUND_RESULT_TOOL_NAME
  )
  if (
    toolUse?.type !== 'toolUseBlock' ||
    !toolResultMessage.content.some(
      (block) => block.type === 'toolResultBlock' && block.toolUseId === toolUse.toolUseId
    )
  ) {
    return undefined
  }
  return toolUse.toolUseId
}

function deliveriesMatch(left: readonly Message[], right: readonly Message[]): boolean {
  const project = (messages: readonly Message[]): unknown =>
    messages.map((message) => {
      const { role, content } = message.toJSON()
      return { role, content }
    })
  return stableStringify(project(left)) === stableStringify(project(right))
}

/** Stable JSON serialization for internal background task comparisons. @internal */
export function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`
  if (value !== null && typeof value === 'object') {
    return `{${Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nested]) => `${JSON.stringify(key)}:${stableStringify(nested)}`)
      .join(',')}}`
  }
  return JSON.stringify(value)
}

function deliveryMetadata(): NonNullable<Message['metadata']> {
  return {
    custom: {
      pinned: true,
    },
  }
}
