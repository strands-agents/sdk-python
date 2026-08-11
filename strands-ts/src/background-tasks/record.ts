import { isSpanContextValid, type SpanContext } from '@opentelemetry/api'
import { InterruptState, type InterruptStateData } from '../interrupt.js'
import type { InvocationState } from '../types/agent.js'
import type { JSONValue, Serialized } from '../types/json.js'
import { ToolResultBlock, type ToolResultBlockData, type ToolResultContentData } from '../types/messages.js'
import { validateStoredEngineTask } from './engine/record.js'
import type { StoredEngineTask } from './engine/types.js'
import type { BackgroundTask } from './types.js'

export interface ToolTaskDescriptor {
  readonly originalToolUseId: string
  readonly toolName: string
  readonly input: JSONValue
  readonly invocationState: InvocationState
  readonly originTraceContext?: SpanContext
}

export type StoredBackgroundTask = StoredEngineTask<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData>

export function validateStoredTask(value: unknown): asserts value is StoredBackgroundTask {
  validateStoredEngineTask(value)
  const record = value as unknown as StoredBackgroundTask
  validateToolTaskDescriptor(record.descriptor)
  if (record.result !== undefined) validateToolTaskResult(record.result)
  if (record.state !== undefined) validateToolTaskState(record.state)
  if (
    record.failure !== undefined &&
    !['toolError', 'executionError', 'timeout', 'recoveryError'].includes(record.failure.type)
  ) {
    invalid('task.failure.type', `unknown failure type '${record.failure.type}'`)
  }
}

function validateToolTaskDescriptor(value: unknown): asserts value is ToolTaskDescriptor {
  const descriptor = requireObject(value, 'task.descriptor')
  for (const key of ['originalToolUseId', 'toolName'] as const) {
    requireString(descriptor[key], `task.descriptor.${key}`)
  }
  requireObject(descriptor.invocationState, 'task.descriptor.invocationState')
  if (descriptor.originTraceContext !== undefined) validateTraceContext(descriptor.originTraceContext)
}

export function toBackgroundTask(record: StoredBackgroundTask): BackgroundTask {
  const resultBlock = record.result ? rehydrateStoredToolResult(record.result) : undefined
  const result: BackgroundTask['result'] = resultBlock
    ? {
        content: resultBlock.toJSON().toolResult.content as Serialized<ToolResultContentData>[],
      }
    : undefined
  const error: BackgroundTask['error'] = record.failure
    ? {
        type: record.failure.type as NonNullable<BackgroundTask['error']>['type'],
        message: record.failure.message,
      }
    : undefined
  const interrupts = record.state ? InterruptState.fromJSON(record.state).getUnansweredInterrupts() : []
  return {
    taskId: record.taskId,
    toolUseId: record.descriptor.originalToolUseId,
    toolName: record.descriptor.toolName,
    status: record.status,
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    ...(result && { result }),
    ...(error && { error }),
    ...(interrupts.length > 0 && { interrupts }),
  }
}

export function rehydrateStoredToolResult(result: ToolResultBlockData): ToolResultBlock {
  try {
    return ToolResultBlock.fromJSON({ toolResult: result })
  } catch (error) {
    throw new Error('Stored tool result cannot be reconstructed', { cause: error })
  }
}

function validateToolTaskResult(value: unknown): asserts value is ToolResultBlockData {
  try {
    ToolResultBlock.fromJSON({ toolResult: value as ToolResultBlockData })
  } catch (error) {
    throw new Error('task.result cannot be reconstructed', { cause: error })
  }
}

function validateToolTaskState(value: unknown): asserts value is InterruptStateData {
  try {
    InterruptState.fromJSON(value as InterruptStateData)
  } catch (error) {
    throw new Error('task.state cannot be reconstructed', {
      cause: error,
    })
  }
}

function validateTraceContext(value: unknown): void {
  const traceContext = requireObject(value, 'task.descriptor.originTraceContext')
  if (!isSpanContextValid(traceContext as unknown as SpanContext)) {
    invalid('task.descriptor.originTraceContext', 'must be a valid span context')
  }
  if (traceContext.isRemote !== undefined && typeof traceContext.isRemote !== 'boolean') {
    invalid('task.descriptor.originTraceContext.isRemote', 'must be a boolean')
  }
}

function requireObject(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) invalid(path, 'must be an object')
  return value as Record<string, unknown>
}

function requireString(value: unknown, path: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) invalid(path, 'must be a non-empty string')
}

function invalid(path: string, message: string): never {
  throw new Error(`${path} ${message}`)
}
