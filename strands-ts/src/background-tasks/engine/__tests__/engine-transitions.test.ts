import { afterEach, describe, expect, it } from 'vitest'
import {
  createEngine,
  deferred,
  expectTask,
  initialize,
  shutdownEngines,
  waitForStatus,
} from './engine-test-helpers.js'
import type { TestContext, TestOutcome } from './engine-test-helpers.js'

afterEach(shutdownEngines)

describe('BackgroundTaskEngine', () => {
  describe('waiting and transitions', () => {
    it('cancels observations without cancelling work', async () => {
      const finish = deferred<TestOutcome>()
      const engine = initialize(createEngine(async () => finish.promise))
      const task = engine.submit({ descriptor: { value: 'work' } })
      const taskController = new AbortController()
      const idleController = new AbortController()
      const taskWaiting = engine.wait(task.taskId, { cancelSignal: taskController.signal })
      const idleWaiting = engine.waitForIdle({ cancelSignal: idleController.signal })

      taskController.abort(null)
      idleController.abort(new Error('stop idle observation'))
      await expect(taskWaiting).rejects.toMatchObject({ name: 'AbortError' })
      await expect(idleWaiting).rejects.toThrow('stop idle observation')
      expectTask(engine.get(task.taskId), task, {
        status: 'working',
      })

      finish.resolve({ status: 'completed', result: { value: 'done' } })
      await engine.wait(task.taskId)
    })

    it('pauses, persists state, and resumes the same attempt', async () => {
      const contexts: TestContext[] = []
      const engine = initialize(
        createEngine(async (context) => {
          contexts.push(context)
          return context.state
            ? { status: 'completed', result: { value: 'resumed' } }
            : { status: 'paused', state: { phase: 'waiting' } }
        })
      )
      const task = engine.submit({ descriptor: { value: 'work' } })
      const paused = await engine.wait(task.taskId)
      expectTask(paused, task, {
        status: 'paused',
        state: { phase: 'waiting' },
      })
      expect(() =>
        engine.resume(task.taskId, () => ({ state: undefined as unknown as { phase: string }, ready: false }))
      ).toThrow('task.state is required while paused')
      expectTask(
        engine.resume(task.taskId, () => ({ state: { phase: 'still waiting' }, ready: false })),
        task,
        { status: 'paused', state: { phase: 'still waiting' } }
      )
      expect(contexts).toHaveLength(1)
      engine.resume(task.taskId, (state) => ({ state, ready: true }))
      expectTask(await engine.wait(task.taskId), task, {
        status: 'completed',
        result: { value: 'resumed' },
      })
      expect(contexts[0]!.attemptId).toBe(contexts[1]!.attemptId)
      expect(contexts[0]!.executionId).not.toBe(contexts[1]!.executionId)
    })

    it('cancels running work, wakes waiters, and removes after execution settles', async () => {
      const finish = deferred<TestOutcome>()
      const events: string[] = []
      const engine = initialize(createEngine(async () => finish.promise, { onEvent: ({ type }) => events.push(type) }))
      const task = engine.submit({ descriptor: { value: 'work' } })
      await waitForStatus(engine, task.taskId, 'working')
      const waiting = engine.wait(task.taskId)

      engine.cancel(task.taskId, { reason: 'Stop work' })
      expectTask(await waiting, task, {
        status: 'cancelled',
        cancellationReason: 'Stop work',
      })
      engine.remove(task.taskId)
      expect(engine.get(task.taskId)).toBeUndefined()
      expect(() => engine.cancel(task.taskId, { reason: 'Again' })).toThrow(
        `Background task '${task.taskId}' was not found`
      )
      finish.resolve({ status: 'completed', result: { value: 'late' } })
      await engine.shutdown({ mode: 'drain', timeout: 1_000 })
      expect(events).toEqual(['admitted', 'executionStarted', 'cancelled', 'executionFinished'])
      expect(engine.get(task.taskId)).toBeUndefined()
    })

    it('rejects invalid configuration and record values without stopping work', async () => {
      const complete = async (): Promise<TestOutcome> => ({ status: 'completed', result: { value: 'done' } })
      expect(() => createEngine(complete, { maxConcurrency: 0 })).toThrow('maxConcurrency must be a positive')
      expect(() => createEngine(complete, { timeout: 0 })).toThrow('timeout must be a positive')
      expect(() => createEngine(complete, { timeout: 2 ** 31 - 1 })).not.toThrow()
      expect(() => createEngine(complete, { timeout: 2 ** 31 })).toThrow('timeout must be at most')

      const finish = deferred<TestOutcome>()
      const engine = initialize(createEngine(async () => finish.promise))
      expect(() => engine.submit({ descriptor: { value: 'invalid' }, idempotencyKey: '' })).toThrow(
        'task.idempotencyKey must be a non-empty string'
      )
      const task = engine.submit({ descriptor: { value: 'work' } })
      await waitForStatus(engine, task.taskId, 'working')
      expect(() => engine.cancel(task.taskId, { reason: '' })).toThrow(
        'task.cancellationReason must be a non-empty string'
      )
      finish.resolve({ status: 'completed', result: { value: 'done' } })
      await engine.wait(task.taskId)
      await expect(engine.shutdown({ mode: 'drain', timeout: 0 })).rejects.toThrow(
        'shutdown timeout must be a positive'
      )
      await expect(engine.shutdown({ mode: 'drain', timeout: 2 ** 31 })).rejects.toThrow(
        'shutdown timeout must be at most'
      )
    })
  })
})
