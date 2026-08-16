import { z } from 'zod'
import { ENGINE_TASK_STATUSES } from './types.js'
import type { StoredEngineTask } from './types.js'

const nonEmptyString = z.string().min(1, { error: 'must be a non-empty string' })
const timestamp = z.iso
  .datetime({ offset: true, error: 'must be a valid ISO-8601 timestamp' })
  .refine((value) => /T\d{2}:\d{2}:\d{2}/.test(value), { error: 'must include seconds' })
const storedEngineTaskSchema = z.object({
  taskId: nonEmptyString,
  idempotencyKey: nonEmptyString.optional(),
  descriptor: z.unknown(),
  status: z.enum(ENGINE_TASK_STATUSES),
  attemptCount: z.number().int().nonnegative().safe(),
  attemptId: nonEmptyString.optional(),
  cancellationReason: nonEmptyString.optional(),
  state: z.unknown().optional(),
  result: z.unknown().optional(),
  failure: z.object({ type: nonEmptyString, message: nonEmptyString }).optional(),
  createdAt: timestamp,
  updatedAt: timestamp,
})

/** Validates a persisted engine task. @internal */
export function validateStoredEngineTask(value: unknown): void {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Stored background task must be an object')
  }
  if (!('descriptor' in value)) throw new Error('task.descriptor is required')
  const parsed = storedEngineTaskSchema.safeParse(value)
  if (!parsed.success) {
    const issue = parsed.error.issues[0]!
    throw new Error(`${['task', ...issue.path].join('.')} ${issue.message}`)
  }

  const record = value as unknown as StoredEngineTask
  validateLifecycle(record)
}

/** Returns whether execution has permanently stopped. @internal */
export function isEngineTerminalStatus(status: StoredEngineTask['status']): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled'
}

function validateLifecycle(record: StoredEngineTask): void {
  if (record.status !== 'cancelled' && record.cancellationReason !== undefined) {
    throw new Error('task.cancellationReason is only valid while cancelled')
  }

  switch (record.status) {
    case 'queued': {
      requireAbsent(record, 'failure', 'result')
      if (record.attemptCount === 0) {
        requireAbsent(record, 'attemptId', 'state')
        return
      }
      requirePresent(record.attemptId, 'task.attemptId is required while queued for resumption')
      requirePresent(record.state, 'task.state is required while queued for resumption')
      return
    }
    case 'working':
      requirePositiveAttemptCount(record)
      requirePresent(record.attemptId, 'task.attemptId is required while working')
      requireAbsent(record, 'failure', 'result')
      return
    case 'paused':
      requirePositiveAttemptCount(record)
      requirePresent(record.attemptId, 'task.attemptId is required while paused')
      requirePresent(record.state, 'task.state is required while paused')
      requireAbsent(record, 'failure', 'result')
      return
    case 'completed':
      requirePositiveAttemptCount(record)
      requirePresent(record.result, 'task.result is required while completed')
      requireAbsent(record, 'attemptId', 'failure')
      return
    case 'failed':
      requirePositiveAttemptCount(record)
      requirePresent(record.failure, 'task.failure is required while failed')
      requireAbsent(record, 'attemptId')
      return
    case 'cancelled':
      requirePresent(record.cancellationReason, 'task.cancellationReason is required while cancelled')
      requireAbsent(record, 'attemptId', 'failure', 'result')
  }
}

function requirePresent(value: unknown, message: string): void {
  if (value === undefined) throw new Error(message)
}

function requirePositiveAttemptCount(record: StoredEngineTask): void {
  if (record.attemptCount === 0) throw new Error(`task.attemptCount must be positive while ${record.status}`)
}

function requireAbsent(record: StoredEngineTask, ...fields: readonly (keyof StoredEngineTask)[]): void {
  for (const field of fields) {
    if (record[field] !== undefined) throw new Error(`task.${String(field)} is not valid while ${record.status}`)
  }
}
