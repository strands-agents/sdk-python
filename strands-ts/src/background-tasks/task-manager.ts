import type { SpanContext } from '@opentelemetry/api'

import type { InvocationState } from '../types/agent.js'
import type { JSONValue } from '../types/json.js'
import type { BackgroundTask } from './types.js'

/** Submission accepted by a background task manager. @internal */
export interface TaskSubmission {
  /** Identifies the submission shape for routing and narrowing. */
  readonly kind: string
}

/** Approved in-process tool call submitted for background execution. @internal */
export interface ToolCallSubmission extends TaskSubmission {
  /** Identifies an in-process tool call submission. */
  readonly kind: 'toolCall'
  /** Registered name of the tool to execute. */
  readonly toolName: string
  /** Tool-use identifier from the originating model request. */
  readonly originalToolUseId: string
  /** Tool input with framework-owned routing fields removed. */
  readonly input: JSONValue
  /** Invocation-scoped state propagated to the tool. */
  readonly invocationState: InvocationState
  /** Agent-loop pass used to deduplicate task admission. */
  readonly passId: string
  /** Trace context captured when the tool call was submitted. */
  readonly originSpanContext?: SpanContext
}

/**
 * Background task lifecycle operations used by the Background Tasks plugin.
 *
 * @typeParam TSubmission - Submission shape accepted by this manager implementation.
 * @internal
 */
export interface TaskManager<TSubmission extends TaskSubmission> {
  initialize(): Promise<void>
  registerHooks(): void
  appStateLoaded(): void
  submitTask(submission: TSubmission): Promise<BackgroundTask>
  getTask(taskId: string): Promise<BackgroundTask | undefined>
  listTasks(): Promise<readonly BackgroundTask[]>
  cancelTask(taskId: string): Promise<BackgroundTask>
  waitForTasks(options?: { readonly timeout?: number }): Promise<void>
}
