import { afterEach, describe, expect, it } from 'vitest'
import {
  abortable,
  createEngine,
  createStoredTask,
  deferred,
  expectTask,
  initialize,
  shutdownEngines,
  waitForStatus,
} from './engine-test-helpers.js'
import type { TestContext, TestOutcome } from './engine-test-helpers.js'

afterEach(shutdownEngines)

describe('BackgroundTaskEngine', () => {
  describe('recovery and shutdown', () => {
    it('rejects duplicate restored identities', () => {
      const first = createStoredTask({
        status: 'completed',
        attemptCount: 1,
        result: { value: 'completed' },
      })
      expect(() =>
        initialize(
          createEngine(async () => ({ status: 'completed', result: { value: 'unused' } })),
          [first, { ...first }]
        )
      ).toThrow('Duplicate background task ID')
      expect(() =>
        initialize(
          createEngine(async () => ({ status: 'completed', result: { value: 'unused' } })),
          [
            { ...first, taskId: 'first', idempotencyKey: 'duplicate' },
            { ...first, taskId: 'second', idempotencyKey: 'duplicate' },
          ]
        )
      ).toThrow('Duplicate background task idempotency key')
    })

    it('restores queued and terminal tasks and fails interrupted work', async () => {
      const attempted = { attemptCount: 1, attemptId: 'attempt' }
      const records = [
        createStoredTask({ status: 'queued' }),
        createStoredTask({ status: 'queued', ...attempted, state: { phase: 'resuming' } }),
        createStoredTask({ status: 'working', ...attempted }),
        createStoredTask({ status: 'paused', ...attempted, state: { phase: 'waiting' } }),
        createStoredTask({ status: 'completed', attemptCount: 1, result: { value: 'completed' } }),
        createStoredTask({
          status: 'failed',
          attemptCount: 1,
          failure: { type: 'toolError', message: 'failed' },
        }),
        createStoredTask({ status: 'cancelled', cancellationReason: 'cancelled' }),
      ]
      const queuedResult = deferred<TestOutcome>()
      const executions: TestContext[] = []
      const engine = initialize(
        createEngine(async (context) => {
          executions.push(context)
          return queuedResult.promise
        }),
        records
      )

      await Promise.all(records.slice(0, 2).map((record) => waitForStatus(engine, record.taskId, 'working')))
      expect(executions).toEqual([
        expect.objectContaining({ taskId: records[0]!.taskId, attempt: 1 }),
        expect.objectContaining({
          taskId: records[1]!.taskId,
          attempt: 1,
          attemptId: 'attempt',
          state: { phase: 'resuming' },
        }),
      ])
      expectTask(engine.get(records[2]!.taskId), records[2]!, {
        status: 'failed',
        failure: {
          type: 'recoveryError',
          message: 'Background task execution was interrupted while restoring persisted state',
        },
      })
      for (const record of records.slice(3)) expect(await engine.wait(record.taskId)).toEqual(record)

      queuedResult.resolve({ status: 'completed', result: { value: 'replayed' } })
      for (const record of records.slice(0, 2)) {
        expectTask(await engine.wait(record.taskId), record, {
          status: 'completed',
          result: { value: 'replayed' },
        })
      }
    })

    it('drains or cancels cleanly during shutdown', async () => {
      const finish = deferred<TestOutcome>()
      const draining = initialize(createEngine(async () => finish.promise))
      const task = draining.submit({ descriptor: { value: 'drain' } })
      const shutdown = draining.shutdown({ mode: 'drain', timeout: 1_000 })
      expect(() => draining.submit({ descriptor: { value: 'rejected' } })).toThrow(/admission is closed/)
      finish.resolve({ status: 'completed', result: { value: 'done' } })
      await shutdown
      expectTask(draining.get(task.taskId), task, { status: 'completed', result: { value: 'done' } })

      const cancelling = initialize(createEngine(abortable))
      const cancelled = cancelling.submit({ descriptor: { value: 'cancel' } })
      await waitForStatus(cancelling, cancelled.taskId, 'working')
      await cancelling.shutdown({ mode: 'cancel', timeout: 1_000 })
      expectTask(cancelling.get(cancelled.taskId), cancelled, {
        status: 'cancelled',
        cancellationReason: 'Coordinator shutdown',
      })
    })

    it('cleans up timed-out shutdown observations so shutdown can be retried', async () => {
      const finish = deferred<TestOutcome>()
      const engine = initialize(createEngine(async () => finish.promise))
      const task = engine.submit({ descriptor: { value: 'work' } })
      await waitForStatus(engine, task.taskId, 'working')

      for (let attempt = 0; attempt < 2; attempt++) {
        await expect(engine.shutdown({ mode: 'drain', timeout: 10 })).rejects.toThrow('shutdown timed out after 10ms')
      }
      finish.resolve({ status: 'completed', result: { value: 'done' } })
      await engine.shutdown({ mode: 'drain', timeout: 1_000 })
    })

    it('keeps execution closed after shutdown', async () => {
      const uninitialized = createEngine(async () => ({ status: 'completed', result: { value: 'unexpected' } }))
      await uninitialized.shutdown({ mode: 'drain', timeout: 1_000 })
      expect(() =>
        uninitialized.initialize([
          createStoredTask({ taskId: 'restored-after-shutdown', descriptor: { value: 'work' }, status: 'queued' }),
        ])
      ).toThrow('Background task execution is closed')

      let executions = 0
      const engine = initialize(
        createEngine(async ({ state }) => {
          executions++
          return state
            ? { status: 'completed', result: { value: 'resumed' } }
            : { status: 'paused', state: { phase: 'waiting' } }
        })
      )
      const task = engine.submit({ descriptor: { value: 'work' } })
      await engine.wait(task.taskId)
      await engine.shutdown({ mode: 'drain', timeout: 1_000 })
      expect(() => engine.resume(task.taskId, (state) => ({ state, ready: true }))).toThrow(
        'Background task execution is closed'
      )
      expectTask(engine.get(task.taskId), task, { status: 'paused', state: { phase: 'waiting' } })
      expect(executions).toBe(1)
    })
  })
})
