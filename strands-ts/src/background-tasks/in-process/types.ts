/** Background task lifecycle statuses. @internal */
export const TASK_STATUSES = ['queued', 'working', 'paused', 'completed', 'failed', 'cancelled'] as const

/** Background task lifecycle status. @internal */
export type TaskStatus = (typeof TASK_STATUSES)[number]

/** Configuration for bounded in-process task execution. @internal */
export interface InProcessTaskEngineConfig<Descriptor, Result, State> {
  /** Maximum number of task executions that may run concurrently. */
  readonly maxConcurrency: number
  /** Per-execution timeout in milliseconds, or `Infinity` to disable timeouts. */
  readonly timeout: number
  /** Executes one task. */
  readonly execute: (
    context: InProcessTaskExecutionContext<Descriptor, State>
  ) => Promise<TaskExecutionOutcome<Result, State>>
  /** Synchronously updates manager-owned state before commit; must not re-enter the engine, and throwing stops it. */
  readonly onTaskUpdated: (task: StoredInProcessTask<Descriptor, Result, State>) => void
}

/** Context supplied to one in-process task execution. @internal */
export interface InProcessTaskExecutionContext<Descriptor, State> {
  /** Identifier of the logical task. */
  readonly taskId: string
  /** Snapshot of the submitted execution descriptor. */
  readonly descriptor: Descriptor
  /** Execution state supplied when resuming a paused task. */
  readonly state?: Exclude<State, undefined>
  /** Signal aborted when this execution should stop. */
  readonly cancelSignal: AbortSignal
}

/** Outcome returned by one in-process task execution. @internal */
export type TaskExecutionOutcome<Result, State> =
  | {
      readonly status: 'completed'
      readonly result: Exclude<Result, undefined>
    }
  | {
      readonly status: 'paused'
      readonly state: Exclude<State, undefined>
    }
  | {
      readonly status: 'failed'
      readonly failure: {
        readonly type: string
        readonly message: string
      }
      readonly result?: Exclude<Result, undefined>
    }

/** Engine-owned in-process task record. @internal */
export interface StoredInProcessTask<Descriptor, Result, State> {
  /** Identifier of the logical task. */
  readonly taskId: string
  /** Optional key used to deduplicate task admission. */
  readonly idempotencyKey?: string
  /** Snapshot of the submitted execution descriptor. */
  readonly descriptor: Descriptor
  /** Current task lifecycle status. */
  status: TaskStatus
  /** Execution state, required while paused and retained while resuming. */
  state?: Exclude<State, undefined>
  /** Execution result, required while completed and optional while failed. */
  result?: Exclude<Result, undefined>
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
