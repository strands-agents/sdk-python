import { describe, expect, it } from 'vitest'
import { validateStoredInProcessTask } from '../record.js'
import { createStoredTask } from './engine-test-helpers.js'

describe('validateStoredInProcessTask', () => {
  it.each<readonly [unknown, string]>([
    [{ ...createStoredTask({ status: 'queued' }), idempotencyKey: '' }, 'task.idempotencyKey'],
    [createStoredTask({ status: 'paused' }), 'task.state is required while paused'],
    [createStoredTask({ status: 'completed' }), 'task.result is required while completed'],
    [
      createStoredTask({
        status: 'completed',
        result: { value: 'done' },
        failure: { type: 'toolError', message: 'contradiction' },
      }),
      'task.failure is not valid while completed',
    ],
    [createStoredTask({ status: 'completed', result: { value: 'done' }, state: { phase: 'stale' } }), 'task.state'],
    [createStoredTask({ status: 'failed', failure: { type: '', message: 'failed' } }), 'task.failure.type'],
    [createStoredTask({ status: 'failed' }), 'task.failure is required while failed'],
    [
      createStoredTask({
        status: 'failed',
        failure: { type: 'toolError', message: 'failed' },
        state: { phase: 'stale' },
      }),
      'task.state',
    ],
    [createStoredTask({ status: 'cancelled', state: { phase: 'waiting' } }), 'task.state is not valid while cancelled'],
    [{ ...createStoredTask({ status: 'queued' }), createdAt: '08/15/2026' }, 'valid ISO-8601'],
  ])('rejects an invalid record', (record, message) => {
    expect(() => validateStoredInProcessTask(record)).toThrow(message)
  })
})
