import type { JSONValue } from '../../types/json.js'

export interface BackgroundTaskEngineConfig<Descriptor, Result, State> {
  readonly maxConcurrency: number
  readonly timeout: number
  readonly execute: (
    context: BackgroundTaskExecutionContext<Descriptor, State>
  ) => Promise<BackgroundTaskExecutionOutcome<Result, State>>
  readonly onTaskUpdated?: (task: StoredEngineTask<Descriptor, Result, State>) => void
  readonly onEvent?: (event: BackgroundTaskEngineEvent<Descriptor, Result, State>) => void
}

export interface BackgroundTaskExecutionContext<Descriptor, State> {
  readonly taskId: string
  readonly descriptor: Descriptor
  readonly state?: State
  readonly attempt: number
  readonly attemptId: string
  readonly executionId: string
  readonly cancelSignal: AbortSignal
}

export type BackgroundTaskExecutionOutcome<Result, State> =
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

export interface StoredEngineTask<Descriptor = JSONValue, Result = JSONValue, State = JSONValue> {
  readonly taskId: string
  readonly idempotencyKey?: string
  readonly descriptor: Descriptor
  status: 'queued' | 'working' | 'paused' | 'completed' | 'failed' | 'cancelled'
  attemptCount: number
  attemptId?: string
  cancellationReason?: string
  state?: State
  result?: Result
  failure?: {
    readonly type: string
    readonly message: string
  }
  readonly createdAt: string
  updatedAt: string
}

export type BackgroundTaskEngineEvent<Descriptor, Result, State> =
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
