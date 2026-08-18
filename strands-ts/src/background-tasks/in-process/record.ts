import { z } from 'zod'
import { TASK_STATUSES } from './types.js'
import type { StoredInProcessTask, TaskStatus } from './types.js'

const nonEmptyString = z.string().min(1, { error: 'must be a non-empty string' })
const timestamp = z.iso.datetime({ offset: true, error: 'must be a valid ISO-8601 timestamp' })
const storedInProcessTaskSchema = z.object({
  taskId: nonEmptyString,
  idempotencyKey: nonEmptyString.optional(),
  descriptor: z.unknown(),
  status: z.enum(TASK_STATUSES),
  state: z.unknown().optional(),
  result: z.unknown().optional(),
  failure: z.object({ type: nonEmptyString, message: nonEmptyString }).optional(),
  createdAt: timestamp,
  updatedAt: timestamp,
})

/** Validates an in-process task engine record. @internal */
export function validateStoredInProcessTask(value: unknown): void {
  const parsed = storedInProcessTaskSchema.safeParse(value)
  if (!parsed.success) {
    const issue = parsed.error.issues[0]!
    throw new Error(`${['task', ...issue.path].join('.')} ${issue.message}`)
  }

  const record = value as unknown as StoredInProcessTask<unknown, unknown, unknown>
  validateLifecycle(record)
}

/** Returns whether execution has permanently stopped. @internal */
export function isInProcessTaskTerminalStatus(status: TaskStatus): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled'
}

function validateLifecycle(record: StoredInProcessTask<unknown, unknown, unknown>): void {
  switch (record.status) {
    case 'queued':
      requireAbsent(record, 'failure', 'result')
      return
    case 'working':
      requireAbsent(record, 'failure', 'result')
      return
    case 'paused':
      requirePresent(record.state, 'task.state is required while paused')
      requireAbsent(record, 'failure', 'result')
      return
    case 'completed':
      requirePresent(record.result, 'task.result is required while completed')
      requireAbsent(record, 'failure', 'state')
      return
    case 'failed':
      requirePresent(record.failure, 'task.failure is required while failed')
      requireAbsent(record, 'state')
      return
    case 'cancelled':
      requireAbsent(record, 'failure', 'result', 'state')
  }
}

function requirePresent<Value>(value: Value | undefined, message: string): asserts value is Value {
  if (value === undefined) throw new Error(message)
}

function requireAbsent(
  record: StoredInProcessTask<unknown, unknown, unknown>,
  ...fields: readonly (keyof StoredInProcessTask<unknown, unknown, unknown>)[]
): void {
  for (const field of fields) {
    if (record[field] !== undefined) throw new Error(`task.${String(field)} is not valid while ${record.status}`)
  }
}
