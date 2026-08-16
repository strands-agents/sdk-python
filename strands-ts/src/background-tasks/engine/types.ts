import type { JSONValue } from '../../types/json.js'

/** Background task engine lifecycle statuses. @internal */
export const ENGINE_TASK_STATUSES = ['queued', 'working', 'paused', 'completed', 'failed', 'cancelled'] as const

/** Value that excludes `undefined`. @internal */
export type DefinedValue = NonNullable<unknown> | null

/** Configuration for bounded background task execution. @internal */
export interface BackgroundTaskEngineConfig<Descriptor, Result extends DefinedValue, State extends DefinedValue> {
  /** Maximum number of task executions that may run concurrently. */
  readonly maxConcurrency: number
  /** Per-execution timeout in milliseconds, or `Infinity` to disable timeouts. */
  readonly timeout: number
  /** Executes one task attempt. */
  readonly execute: (
    context: BackgroundTaskExecutionContext<Descriptor, State>
  ) => Promise<BackgroundTaskExecutionOutcome<Result, State>>
  /** Synchronously persists each record before commit; must not re-enter the engine, and throwing stops it. */
  readonly onTaskUpdated?: (task: StoredEngineTask<Descriptor, Result, State>) => void
  /** Observes committed lifecycle events without re-entering the engine; thrown errors are ignored. */
  readonly onEvent?: (event: BackgroundTaskEngineEvent<Descriptor, Result, State>) => void
}

/** Context supplied to one background task execution. @internal */
export interface BackgroundTaskExecutionContext<Descriptor, State extends DefinedValue> {
  /** Identifier of the logical task. */
  readonly taskId: string
  /** Snapshot of the submitted execution descriptor. */
  readonly descriptor: Descriptor
  /** Persisted state supplied when resuming a paused task. */
  readonly state?: State
  /** One-based logical attempt number. */
  readonly attempt: number
  /** Identifier retained across executions of the same paused attempt. */
  readonly attemptId: string
  /** Identifier unique to this execution. */
  readonly executionId: string
  /** Signal aborted when this execution should stop. */
  readonly cancelSignal: AbortSignal
}

/** Outcome returned by one background task execution. @internal */
export type BackgroundTaskExecutionOutcome<Result extends DefinedValue, State extends DefinedValue> =
  | {
      readonly status: 'completed'
      readonly result: Result
      readonly state?: State
    }
  | {
      readonly status: 'paused'
      readonly state: State
    }
  | {
      readonly status: 'failed'
      readonly failure: {
        readonly type: string
        readonly message: string
      }
      readonly result?: Result
      readonly state?: State
    }

/** Persisted background task engine record. @internal */
export interface StoredEngineTask<
  Descriptor = JSONValue,
  Result extends DefinedValue = JSONValue,
  State extends DefinedValue = JSONValue,
> {
  /** Identifier of the logical task. */
  readonly taskId: string
  /** Optional key used to deduplicate task admission. */
  readonly idempotencyKey?: string
  /** Snapshot of the submitted execution descriptor. */
  readonly descriptor: Descriptor
  /** Current task lifecycle status. */
  status: (typeof ENGINE_TASK_STATUSES)[number]
  /** Number of logical attempts started. */
  attemptCount: number
  /** Identifier retained while an attempt is active, paused, or queued for resumption. */
  attemptId?: string
  /** Reason recorded when the task is cancelled. */
  cancellationReason?: string
  /** Persisted state, required while paused or queued for resumption. */
  state?: State
  /** Execution result, required while completed and optional while failed. */
  result?: Result
  /** Failure details recorded when the task fails. */
  failure?: {
    readonly type: string
    readonly message: string
  }
  /** ISO-8601 creation timestamp. */
  readonly createdAt: string
  /** ISO-8601 timestamp of the latest transition. */
  updatedAt: string
}

/** Background task engine lifecycle notification. @internal */
export type BackgroundTaskEngineEvent<Descriptor, Result extends DefinedValue, State extends DefinedValue> =
  | {
      readonly type: 'admitted'
      readonly task: StoredEngineTask<Descriptor, Result, State>
    }
  | {
      readonly type: 'executionStarted'
      readonly task: StoredEngineTask<Descriptor, Result, State>
      readonly resumed: boolean
      readonly queueDuration: number
    }
  | {
      readonly type: 'executionFinished'
      readonly task: StoredEngineTask<Descriptor, Result, State>
      readonly duration: number
    }
  | {
      readonly type: 'cancelled'
      readonly task: StoredEngineTask<Descriptor, Result, State>
    }
