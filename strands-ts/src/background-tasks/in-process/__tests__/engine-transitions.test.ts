import { describe, expect, it } from 'vitest'
import {
  createDescriptor,
  createEngine,
  createResult,
  createState,
  deferred,
  expectTask,
  getStateValue,
} from './engine-test-helpers.js'
import type { TestOutcome } from './engine-test-helpers.js'

describe('InProcessTaskEngine', () => {
  describe('waiting and transitions', () => {
    it('cancels idle observation without cancelling work', async () => {
      const finish = deferred<TestOutcome>()
      const engine = createEngine(async () => finish.promise)
      const task = engine.submit({ descriptor: createDescriptor('work') })
      const idleController = new AbortController()
      const idleWaiting = engine.waitForIdle({ cancelSignal: idleController.signal })

      idleController.abort(new Error('stop idle observation'))
      await expect(idleWaiting).rejects.toThrow('stop idle observation')
      expectTask(engine.get(task.taskId), task, {
        status: 'working',
      })

      finish.resolve({ status: 'completed', result: createResult('done') })
      await engine.waitForIdle()
    })

    it('pauses, updates state, and resumes execution', async () => {
      const engine = createEngine(async ({ state }) =>
        state
          ? { status: 'completed', result: createResult(getStateValue(state)) }
          : { status: 'paused', state: createState('waiting') }
      )
      const task = engine.submit({ descriptor: createDescriptor('work') })
      await engine.waitForIdle()
      expectTask(engine.get(task.taskId), task, {
        status: 'paused',
        state: createState('waiting'),
      })
      expectTask(
        engine.resume(task.taskId, () => ({ state: createState('still waiting'), ready: false })),
        task,
        { status: 'paused', state: createState('still waiting') }
      )
      engine.resume(task.taskId, (state) => ({ state, ready: true }))
      await engine.waitForIdle()
      expectTask(engine.get(task.taskId), task, {
        status: 'completed',
        result: createResult('still waiting'),
      })
    })

    it('cancels running work and removes it before execution settles', async () => {
      const finish = deferred<TestOutcome>()
      let executionSignal: AbortSignal | undefined
      const engine = createEngine(async ({ cancelSignal }) => {
        executionSignal = cancelSignal
        return finish.promise
      })
      const task = engine.submit({ descriptor: createDescriptor('work') })

      expectTask(engine.cancel(task.taskId, { reason: 'Stop work' }), task, { status: 'cancelled' })
      expect(executionSignal?.reason).toBe('Stop work')
      engine.remove(task.taskId)
      expect(engine.get(task.taskId)).toBeUndefined()
      expect(() => engine.cancel(task.taskId, { reason: 'Again' })).toThrow(
        `Background task '${task.taskId}' was not found`
      )
      finish.resolve({ status: 'completed', result: createResult('late') })
      await engine.waitForIdle()
    })

    it('cancels queued work without executing it', async () => {
      const finish = deferred<TestOutcome>()
      const executions: string[] = []
      const engine = createEngine(
        async ({ descriptor }) => {
          executions.push(descriptor.toolName)
          return finish.promise
        },
        { maxConcurrency: 1 }
      )
      engine.submit({ descriptor: createDescriptor('running') })
      const queued = engine.submit({ descriptor: createDescriptor('queued') })

      expectTask(engine.cancel(queued.taskId, { reason: 'No longer needed' }), queued, {
        status: 'cancelled',
      })
      finish.resolve({ status: 'completed', result: createResult('done') })
      await engine.waitForIdle()
      expect(executions).toEqual(['running'])
    })

    it('rejects invalid execution configuration', () => {
      const complete = async (): Promise<TestOutcome> => ({ status: 'completed', result: createResult('done') })
      expect(() => createEngine(complete, { maxConcurrency: 0 })).toThrow('maxConcurrency must be a positive')
      expect(() => createEngine(complete, { timeout: 0 })).toThrow('timeout must be a positive')
      expect(() => createEngine(complete, { timeout: 2 ** 31 - 1 })).not.toThrow()
      expect(() => createEngine(complete, { timeout: 2 ** 31 })).toThrow('timeout must be at most')
    })
  })
})
