import { afterEach, describe, expect, it } from 'vitest'
import {
  createEngine,
  deferred,
  expectTask,
  initialize,
  shutdownEngines,
  waitForStatus,
} from './engine-test-helpers.js'
import type { TestOutcome } from './engine-test-helpers.js'

afterEach(shutdownEngines)

describe('InProcessTaskEngine', () => {
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

    it('pauses, persists state, and resumes execution', async () => {
      const engine = initialize(
        createEngine(async ({ state }) =>
          state
            ? { status: 'completed', result: { value: state.phase } }
            : { status: 'paused', state: { phase: 'waiting' } }
        )
      )
      const task = engine.submit({ descriptor: { value: 'work' } })
      const paused = await engine.wait(task.taskId)
      expectTask(paused, task, {
        status: 'paused',
        state: { phase: 'waiting' },
      })
      expectTask(
        engine.resume(task.taskId, () => ({ state: { phase: 'still waiting' }, ready: false })),
        task,
        { status: 'paused', state: { phase: 'still waiting' } }
      )
      engine.resume(task.taskId, (state) => ({ state, ready: true }))
      expectTask(await engine.wait(task.taskId), task, {
        status: 'completed',
        result: { value: 'still waiting' },
      })
    })

    it('cancels running work, wakes waiters, and removes after execution settles', async () => {
      const finish = deferred<TestOutcome>()
      let executionSignal: AbortSignal | undefined
      const engine = initialize(
        createEngine(async ({ cancelSignal }) => {
          executionSignal = cancelSignal
          return finish.promise
        })
      )
      const task = engine.submit({ descriptor: { value: 'work' } })
      await waitForStatus(engine, task.taskId, 'working')
      const waiting = engine.wait(task.taskId)

      engine.cancel(task.taskId, { reason: 'Stop work' })
      expect(executionSignal?.reason).toBe('Stop work')
      expectTask(await waiting, task, {
        status: 'cancelled',
      })
      engine.remove(task.taskId)
      expect(engine.get(task.taskId)).toBeUndefined()
      expect(() => engine.cancel(task.taskId, { reason: 'Again' })).toThrow(
        `Background task '${task.taskId}' was not found`
      )
      finish.resolve({ status: 'completed', result: { value: 'late' } })
      await engine.shutdown({ timeout: 1_000 })
    })

    it('cancels queued work without executing it', async () => {
      const finish = deferred<TestOutcome>()
      const executions: string[] = []
      const engine = initialize(
        createEngine(
          async ({ descriptor }) => {
            executions.push(descriptor.value)
            return finish.promise
          },
          { maxConcurrency: 1 }
        )
      )
      const running = engine.submit({ descriptor: { value: 'running' } })
      await waitForStatus(engine, running.taskId, 'working')
      const queued = engine.submit({ descriptor: { value: 'queued' } })

      expectTask(engine.cancel(queued.taskId, { reason: 'No longer needed' }), queued, {
        status: 'cancelled',
      })
      finish.resolve({ status: 'completed', result: { value: 'done' } })
      await engine.waitForIdle()
      expect(executions).toEqual(['running'])
    })

    it('rejects invalid timer configuration without stopping work', async () => {
      const complete = async (): Promise<TestOutcome> => ({ status: 'completed', result: { value: 'done' } })
      expect(() => createEngine(complete, { maxConcurrency: 0 })).toThrow('maxConcurrency must be a positive')
      expect(() => createEngine(complete, { timeout: 0 })).toThrow('timeout must be a positive')
      expect(() => createEngine(complete, { timeout: 2 ** 31 - 1 })).not.toThrow()
      expect(() => createEngine(complete, { timeout: 2 ** 31 })).toThrow('timeout must be at most')

      const finish = deferred<TestOutcome>()
      const engine = initialize(createEngine(async () => finish.promise))
      const task = engine.submit({ descriptor: { value: 'work' } })
      await waitForStatus(engine, task.taskId, 'working')
      finish.resolve({ status: 'completed', result: { value: 'done' } })
      await engine.wait(task.taskId)
      await expect(engine.shutdown({ timeout: 0 })).rejects.toThrow('shutdown timeout must be a positive')
      await expect(engine.shutdown({ timeout: 2 ** 31 })).rejects.toThrow('shutdown timeout must be at most')
    })
  })
})
