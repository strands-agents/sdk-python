import { describe, expect, it } from 'vitest'
import { validateStoredEngineTask } from '../record.js'
import { createStoredTask } from './engine-test-helpers.js'
import type { TestTask } from './engine-test-helpers.js'

describe('validateStoredEngineTask', () => {
  it.each<TestTask>([
    createStoredTask({ status: 'queued' }),
    createStoredTask({ status: 'queued', attemptCount: 1, attemptId: 'attempt', state: { phase: 'resuming' } }),
    createStoredTask({ status: 'working', attemptCount: 1, attemptId: 'attempt' }),
    createStoredTask({ status: 'paused', attemptCount: 1, attemptId: 'attempt', state: { phase: 'waiting' } }),
    createStoredTask({ status: 'completed', attemptCount: 1, result: { value: 'done' } }),
    createStoredTask({ status: 'failed', attemptCount: 1, failure: { type: 'toolError', message: 'failed' } }),
    createStoredTask({ status: 'cancelled', cancellationReason: 'cancelled' }),
    createStoredTask({ status: 'queued', createdAt: '2024-01-01T00:00:00.000+00:00' }),
  ])('accepts a valid $status record', (record) => {
    expect(() => validateStoredEngineTask(record)).not.toThrow()
  })

  it.each<readonly [unknown, string]>([
    [{ ...createStoredTask({ status: 'queued' }), idempotencyKey: '' }, 'task.idempotencyKey'],
    [createStoredTask({ status: 'queued', attemptId: 'attempt' }), 'task.attemptId is not valid while queued'],
    [
      createStoredTask({ status: 'queued', attemptCount: 1, attemptId: 'attempt' }),
      'task.state is required while queued for resumption',
    ],
    [createStoredTask({ status: 'working', attemptId: 'attempt' }), 'task.attemptCount must be positive while working'],
    [createStoredTask({ status: 'paused', attemptCount: 1, state: { phase: 'waiting' } }), 'task.attemptId'],
    [
      createStoredTask({ status: 'paused', attemptCount: 1, attemptId: 'attempt' }),
      'task.state is required while paused',
    ],
    [createStoredTask({ status: 'completed', attemptCount: 1 }), 'task.result is required while completed'],
    [
      createStoredTask({
        status: 'completed',
        attemptCount: 1,
        result: { value: 'done' },
        failure: { type: 'toolError', message: 'contradiction' },
      }),
      'task.failure is not valid while completed',
    ],
    [
      createStoredTask({ status: 'failed', attemptCount: 1, failure: { type: '', message: 'failed' } }),
      'task.failure.type',
    ],
    [createStoredTask({ status: 'failed', attemptCount: 1 }), 'task.failure is required while failed'],
    [
      createStoredTask({ status: 'queued', cancellationReason: 'stale cancellation' }),
      'task.cancellationReason is only valid while cancelled',
    ],
    [createStoredTask({ status: 'cancelled' }), 'task.cancellationReason is required while cancelled'],
    [{ ...createStoredTask({ status: 'queued' }), createdAt: '08/15/2026' }, 'valid ISO-8601'],
    [{ ...createStoredTask({ status: 'queued' }), createdAt: '2026-08-15T12:00Z' }, 'include seconds'],
    [{ ...createStoredTask({ status: 'queued' }), updatedAt: '2026-02-30T00:00:00.000Z' }, 'valid ISO-8601'],
  ])('rejects an invalid record', (record, message) => {
    expect(() => validateStoredEngineTask(record)).toThrow(message)
  })
})
