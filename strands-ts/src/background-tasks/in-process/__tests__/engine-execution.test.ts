import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  createEngine,
  deferred,
  expectTask,
  initialize,
  shutdownEngines,
  waitForStatus,
} from './engine-test-helpers.js'
import type { TestOutcome, TestResult, TestTask } from './engine-test-helpers.js'

afterEach(shutdownEngines)

describe('InProcessTaskEngine', () => {
  describe('admission and isolation', () => {
    it('executes, reports updates, lists, and deduplicates work', async () => {
      const statuses: TestTask['status'][] = []
      const engine = initialize(
        createEngine(
          async ({ descriptor }) => ({ status: 'completed', result: { value: descriptor.value.toUpperCase() } }),
          {
            onTaskUpdated: (task) => statuses.push(task.status),
          }
        )
      )
      expect(() => engine.submit({ descriptor: { value: 'invalid' }, idempotencyKey: '' })).toThrow(
        'task.idempotencyKey must be a non-empty string'
      )
      const admitted = engine.submit({ descriptor: { value: 'hello' }, idempotencyKey: 'work-1' })
      const duplicate = engine.submit({ descriptor: { value: 'hello' }, idempotencyKey: 'work-1' })

      expect(duplicate.taskId).toBe(admitted.taskId)
      const completed = await engine.wait(admitted.taskId)
      expectTask(completed, admitted, { status: 'completed', result: { value: 'HELLO' } })
      expect(engine.list()).toEqual([completed])
      expect(statuses).toEqual(['queued', 'working', 'completed'])

      const external = engine.get(admitted.taskId)
      ;(external!.result as { value: string }).value = 'changed'
      expect(engine.get(admitted.taskId)).toEqual(completed)
      engine.remove(admitted.taskId)
      expect(engine.list()).toEqual([])
    })

    it('snapshots submitted descriptors and outcomes', async () => {
      const blocker = deferred<TestOutcome>()
      const result = { value: 'result' }
      const engine = initialize(
        createEngine(
          async ({ descriptor }) => (descriptor.value === 'block' ? blocker.promise : { status: 'completed', result }),
          { maxConcurrency: 1 }
        )
      )
      const blocking = engine.submit({ descriptor: { value: 'block' } })
      await waitForStatus(engine, blocking.taskId, 'working')
      const descriptor = { value: 'original' }
      const admitted = engine.submit({ descriptor })

      descriptor.value = 'caller mutation'
      blocker.resolve({ status: 'completed', result: { value: 'blocker' } })
      await Promise.all([engine.wait(blocking.taskId), engine.wait(admitted.taskId)])
      result.value = 'executor mutation'

      expectTask(engine.get(admitted.taskId), admitted, {
        status: 'completed',
        result: { value: 'result' },
      })
    })

    it('stops coherently when task storage fails', async () => {
      const admissionError = new Error('admission persistence failed')
      const admissionEngine = initialize(
        createEngine(async () => ({ status: 'completed', result: { value: 'unexpected' } }), {
          onTaskUpdated: () => {
            throw admissionError
          },
        })
      )
      expect(() => admissionEngine.submit({ descriptor: { value: 'admission' } })).toThrow(admissionError)
      expect(admissionEngine.list()).toEqual([])
      await expect(admissionEngine.waitForIdle()).rejects.toBe(admissionError)

      const terminalError = new Error('terminal persistence failed')
      const terminalEngine = initialize(
        createEngine(async () => ({ status: 'completed', result: { value: 'done' } }), {
          onTaskUpdated: (task) => {
            if (task.status === 'completed') throw terminalError
          },
        })
      )
      const terminal = terminalEngine.submit({ descriptor: { value: 'terminal' } })
      await expect(terminalEngine.wait(terminal.taskId)).rejects.toBe(terminalError)
      expectTask(terminalEngine.get(terminal.taskId), terminal, {
        status: 'working',
      })
      expect(() => terminalEngine.submit({ descriptor: { value: 'again' } })).toThrow(terminalError)
      await expect(terminalEngine.shutdown({ timeout: 1_000 })).resolves.toBeUndefined()
    })
  })

  describe('execution', () => {
    it('fails invalid execution data without stopping unrelated work', async () => {
      const healthyResult = deferred<TestOutcome>()
      const engine = initialize(
        createEngine(async ({ descriptor }) => {
          if (descriptor.value === 'healthy') return healthyResult.promise
          if (descriptor.value === 'missing') {
            return { status: 'completed', result: undefined as unknown as TestResult }
          }
          return {
            status: 'completed',
            result: { value: 'invalid', uncloneable: () => undefined } as unknown as TestResult,
          }
        })
      )
      const healthy = engine.submit({ descriptor: { value: 'healthy' } })
      const invalid = ['uncloneable', 'missing'].map((value) => engine.submit({ descriptor: { value } }))
      const failed: TestTask[] = []

      for (const task of invalid) {
        const result = await engine.wait(task.taskId)
        expectTask(result, task, {
          status: 'failed',
          failure: { type: 'executionError', message: expect.any(String) },
        })
        failed.push(result)
      }
      expectTask(engine.get(healthy.taskId), healthy, {
        status: 'working',
      })
      healthyResult.resolve({ status: 'completed', result: { value: 'healthy' } })
      expectTask(await engine.wait(healthy.taskId), healthy, {
        status: 'completed',
        result: { value: 'healthy' },
      })

      const restoredExecute = vi.fn(async (): Promise<TestOutcome> => ({
        status: 'completed',
        result: { value: 'unexpected' },
      }))
      expect(initialize(createEngine(restoredExecute), failed).list()).toEqual(failed)
      expect(restoredExecute).not.toHaveBeenCalled()
    })

    it('bounds execution concurrency', async () => {
      const releases = [deferred<TestOutcome>(), deferred<TestOutcome>(), deferred<TestOutcome>()]
      let active = 0
      let maximum = 0
      const engine = initialize(
        createEngine(
          async ({ descriptor }) => {
            maximum = Math.max(maximum, ++active)
            const outcome = await releases[Number(descriptor.value)]!.promise
            active--
            return outcome
          },
          { maxConcurrency: 2 }
        )
      )
      const tasks = ['0', '1', '2'].map((value) => engine.submit({ descriptor: { value } }))
      await Promise.all(tasks.slice(0, 2).map((task) => waitForStatus(engine, task.taskId, 'working')))
      expectTask(engine.get(tasks[2]!.taskId), tasks[2]!, { status: 'queued' })

      releases[0]!.resolve({ status: 'completed', result: { value: '0' } })
      await waitForStatus(engine, tasks[2]!.taskId, 'working')
      releases[1]!.resolve({ status: 'completed', result: { value: '1' } })
      releases[2]!.resolve({ status: 'completed', result: { value: '2' } })
      await Promise.all(tasks.map((task) => engine.wait(task.taskId)))
      expect(maximum).toBe(2)
    })

    it('records classified failures', async () => {
      const thrownEngine = initialize(
        createEngine(async ({ descriptor }) => {
          if (descriptor.value === 'opaque') throw Object.create(null)
          throw new TypeError('Execution exploded')
        })
      )
      const thrown = thrownEngine.submit({ descriptor: { value: 'throw' } })
      expectTask(await thrownEngine.wait(thrown.taskId), thrown, {
        status: 'failed',
        failure: { type: 'executionError', message: 'Execution exploded' },
      })
      const opaque = thrownEngine.submit({ descriptor: { value: 'opaque' } })
      expectTask(await thrownEngine.wait(opaque.taskId), opaque, {
        status: 'failed',
        failure: { type: 'executionError', message: 'Background task execution failed' },
      })

      const returnedEngine = initialize(
        createEngine(async ({ descriptor }) =>
          descriptor.value === 'empty'
            ? { status: 'failed', failure: { type: 'executionError', message: '' } }
            : {
                status: 'failed',
                failure: { type: 'toolError', message: 'Tool failed' },
                result: { value: 'tool detail' },
              }
        )
      )
      const returned = returnedEngine.submit({ descriptor: { value: 'return' } })
      expectTask(await returnedEngine.wait(returned.taskId), returned, {
        status: 'failed',
        failure: { type: 'toolError', message: 'Tool failed' },
        result: { value: 'tool detail' },
      })
      const empty = returnedEngine.submit({ descriptor: { value: 'empty' } })
      const emptyRecord = await returnedEngine.wait(empty.taskId)
      expectTask(emptyRecord, empty, {
        status: 'failed',
        failure: { type: 'executionError', message: 'Background task execution failed' },
      })
    })

    it('does not release capacity until a timed-out execution settles', async () => {
      const release = deferred<TestOutcome>()
      let hungSignal: AbortSignal | undefined
      const engine = initialize(
        createEngine(
          async ({ descriptor, cancelSignal }) => {
            if (descriptor.value === 'hang') {
              hungSignal = cancelSignal
              return release.promise
            }
            return { status: 'completed', result: { value: 'next' } }
          },
          { maxConcurrency: 1, timeout: 10 }
        )
      )
      const hung = engine.submit({ descriptor: { value: 'hang' } })
      const next = engine.submit({ descriptor: { value: 'next' } })
      await engine.wait(hung.taskId)
      try {
        expect(hungSignal?.aborted).toBe(true)
        expectTask(engine.get(next.taskId), next, { status: 'queued' })
      } finally {
        release.resolve({ status: 'completed', result: { value: 'late' } })
      }
      expectTask(await engine.wait(next.taskId), next, {
        status: 'completed',
        result: { value: 'next' },
      })
      expectTask(engine.get(hung.taskId), hung, {
        status: 'failed',
        failure: { type: 'timeout', message: 'Timed out after 10ms' },
      })
    })

    it('allows Infinity to disable execution timeouts', async () => {
      const finish = deferred<TestOutcome>()
      const timeoutSpy = vi.spyOn(globalThis, 'setTimeout')
      const engine = initialize(createEngine(async () => finish.promise, { timeout: Infinity }))
      const task = engine.submit({ descriptor: { value: 'work' } })
      await waitForStatus(engine, task.taskId, 'working')
      expect(timeoutSpy).not.toHaveBeenCalled()
      finish.resolve({ status: 'completed', result: { value: 'done' } })
      await engine.wait(task.taskId)
      timeoutSpy.mockRestore()
    })
  })
})
