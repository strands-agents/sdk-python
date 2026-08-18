import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  createEngine,
  createStoredTask,
  deferred,
  expectTask,
  initialize,
  shutdownEngines,
  waitForStatus,
} from './engine-test-helpers.js'
import type { TestOutcome } from './engine-test-helpers.js'

afterEach(shutdownEngines)

describe('InProcessTaskEngine', () => {
  describe('recovery and shutdown', () => {
    it('rejects invalid restored records without initializing', () => {
      const engine = createEngine(async () => ({ status: 'completed', result: { value: 'unexpected' } }))

      expect(() => engine.initialize([createStoredTask({ status: 'paused' })])).toThrow(
        'task.state is required while paused'
      )
      expect(() => engine.list()).toThrow('is not initialized')
    })

    it('restores terminal tasks and fails nonterminal work without replaying it', async () => {
      const records = [
        createStoredTask({ status: 'queued', state: { phase: 'resuming' } }),
        createStoredTask({ status: 'working' }),
        createStoredTask({ status: 'paused', state: { phase: 'waiting' } }),
        createStoredTask({ status: 'completed', result: { value: 'completed' } }),
        createStoredTask({
          status: 'failed',
          failure: { type: 'toolError', message: 'failed' },
        }),
        createStoredTask({ status: 'cancelled' }),
      ]
      const execute = vi.fn(async (): Promise<TestOutcome> => ({
        status: 'completed',
        result: { value: 'unexpected' },
      }))
      const engine = initialize(createEngine(execute), records)

      for (const record of records.slice(0, 3)) {
        expectTask(await engine.wait(record.taskId), record, {
          status: 'failed',
          failure: {
            type: 'recoveryError',
            message: 'Background task execution was interrupted while restoring persisted state',
          },
        })
      }
      for (const record of records.slice(3)) expect(await engine.wait(record.taskId)).toEqual(record)
      expect(execute).not.toHaveBeenCalled()
    })

    it('cancels cleanly during shutdown', async () => {
      const finish = deferred<TestOutcome>()
      const engine = initialize(createEngine(async () => finish.promise))
      const task = engine.submit({ descriptor: { value: 'work' } })
      await waitForStatus(engine, task.taskId, 'working')
      const shutdown = engine.shutdown({ timeout: 1_000 })

      expect(() => engine.submit({ descriptor: { value: 'rejected' } })).toThrow(/admission is closed/)
      expectTask(await engine.wait(task.taskId), task, {
        status: 'cancelled',
      })
      finish.resolve({ status: 'completed', result: { value: 'late' } })
      await shutdown
    })

    it('cleans up a timed-out shutdown so it can be retried', async () => {
      const finish = deferred<TestOutcome>()
      const engine = initialize(createEngine(async () => finish.promise))
      const task = engine.submit({ descriptor: { value: 'work' } })
      await waitForStatus(engine, task.taskId, 'working')

      await expect(engine.shutdown({ timeout: 10 })).rejects.toThrow('shutdown timed out after 10ms')
      finish.resolve({ status: 'completed', result: { value: 'done' } })
      await engine.shutdown({ timeout: 1_000 })
    })

    it('keeps execution closed after shutdown', async () => {
      const uninitialized = createEngine(async () => ({ status: 'completed', result: { value: 'unexpected' } }))
      await uninitialized.shutdown({ timeout: 1_000 })
      expect(() =>
        uninitialized.initialize([
          createStoredTask({ taskId: 'restored-after-shutdown', descriptor: { value: 'work' }, status: 'queued' }),
        ])
      ).toThrow('Background task execution is closed')

      const engine = initialize(
        createEngine(async ({ state }) => {
          return state
            ? { status: 'completed', result: { value: 'resumed' } }
            : { status: 'paused', state: { phase: 'waiting' } }
        })
      )
      const task = engine.submit({ descriptor: { value: 'work' } })
      await engine.wait(task.taskId)
      await engine.shutdown({ timeout: 1_000 })
      expect(() => engine.resume(task.taskId, (state) => ({ state, ready: true }))).toThrow(
        'Background task execution is closed'
      )
      expectTask(engine.get(task.taskId), task, {
        status: 'cancelled',
      })
    })
  })
})
