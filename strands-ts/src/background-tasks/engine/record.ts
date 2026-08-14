import type { StoredEngineTask } from './types.js'

const STATUSES = new Set<StoredEngineTask['status']>([
  'queued',
  'working',
  'paused',
  'completed',
  'failed',
  'cancelled',
])

/** Validates a persisted engine task. @internal */
export function validateStoredEngineTask(value: unknown): asserts value is StoredEngineTask {
  if (!isObject(value)) throw new Error('Stored background task must be an object')
  requireString(value.taskId, 'task.taskId')
  if (!('descriptor' in value)) throw new Error('task.descriptor is required')
  if (!STATUSES.has(value.status as StoredEngineTask['status'])) {
    throw new Error(`task.status '${String(value.status)}' is invalid`)
  }
  requireNonNegativeInteger(value.attemptCount, 'task.attemptCount')
  requireDate(value.createdAt, 'task.createdAt')
  requireDate(value.updatedAt, 'task.updatedAt')

  if (value.attemptId !== undefined) requireString(value.attemptId, 'task.attemptId')
  if (value.cancellationReason !== undefined) requireString(value.cancellationReason, 'task.cancellationReason')
  if (value.failure !== undefined) validateFailure(value.failure)

  const record = value as unknown as StoredEngineTask
  if (record.status === 'paused' && record.state === undefined) {
    throw new Error('task.state is required while paused')
  }
  if (record.status === 'completed' && record.result === undefined) {
    throw new Error('task.result is required while completed')
  }
  if (record.status === 'failed' && record.failure === undefined) {
    throw new Error('task.failure is required while failed')
  }
  if (record.status === 'cancelled' && record.cancellationReason === undefined) {
    throw new Error('task.cancellationReason is required while cancelled')
  }
}

/** Returns whether execution has permanently stopped. @internal */
export function isEngineTerminalStatus(status: StoredEngineTask['status']): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled'
}

function validateFailure(value: unknown): void {
  if (!isObject(value)) throw new Error('task.failure must be an object')
  requireString(value.type, 'task.failure.type')
  requireString(value.message, 'task.failure.message')
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function requireString(value: unknown, path: string): asserts value is string {
  if (typeof value !== 'string' || value.length === 0) throw new Error(`${path} must be a non-empty string`)
}

function requireNonNegativeInteger(value: unknown, path: string): void {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${path} must be a non-negative integer`)
  }
}

function requireDate(value: unknown, path: string): void {
  requireString(value, path)
  if (Number.isNaN(Date.parse(value))) throw new Error(`${path} must be an ISO-8601 timestamp`)
}
