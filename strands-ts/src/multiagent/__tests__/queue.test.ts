import { beforeEach, describe, expect, it } from 'vitest'
import { Queue } from '../queue.js'
import type { QueueData } from '../queue.js'
import type { Node } from '../nodes.js'
import { NodeResult, Status } from '../state.js'

describe('Queue', () => {
  let queue: Queue
  let mockNode: Node

  beforeEach(() => {
    mockNode = { id: 'node-1' } as Node
    queue = new Queue()
  })

  describe('push and shift', () => {
    it('dequeues in FIFO order', () => {
      const data1: QueueData = {
        type: 'result',
        node: mockNode,
        result: new NodeResult({ nodeId: 'node-1', status: Status.COMPLETED, duration: 10 }),
      }
      const data2: QueueData = { type: 'error', node: mockNode, error: new Error('fail') }

      queue.push(data1)
      queue.push(data2)

      expect(queue.shift()?.data).toBe(data1)
      expect(queue.shift()?.data).toBe(data2)
    })

    it('returns undefined when empty', () => {
      expect(queue.shift()).toBeUndefined()
    })

    it('provides a no-op ack for fire-and-forget pushes', () => {
      queue.push({ type: 'error', node: mockNode, error: new Error('a') })
      const entry = queue.shift()!
      expect(() => entry.ack()).not.toThrow()
    })
  })

  describe('send', () => {
    it('resolves when consumer calls ack', async () => {
      const data: QueueData = { type: 'error', node: mockNode, error: new Error('a') }
      let resolved = false

      const waiting = queue.send(data).then(() => {
        resolved = true
      })

      await Promise.resolve()
      expect(resolved).toBe(false)

      const entry = queue.shift()!
      expect(entry.data).toBe(data)

      await Promise.resolve()
      expect(resolved).toBe(false)

      entry.ack()
      await waiting
      expect(resolved).toBe(true)
    })
  })

  describe('size', () => {
    it('reflects the current number of entries', () => {
      expect(queue.size).toBe(0)

      queue.push({ type: 'error', node: mockNode, error: new Error('a') })
      queue.push({ type: 'error', node: mockNode, error: new Error('b') })
      expect(queue.size).toBe(2)

      queue.shift()
      expect(queue.size).toBe(1)
    })
  })

  describe('wait', () => {
    it('resolves immediately when entries are available', async () => {
      queue.push({ type: 'error', node: mockNode, error: new Error('a') })

      await queue.wait()

      expect(queue.size).toBe(1)
    })

    it('blocks until data is pushed', async () => {
      let resolved = false

      const waiting = queue.wait().then(() => {
        resolved = true
      })

      await Promise.resolve()
      expect(resolved).toBe(false)

      queue.push({ type: 'error', node: mockNode, error: new Error('a') })

      await waiting
      expect(resolved).toBe(true)
    })

    it('blocks until data is sent', async () => {
      let resolved = false

      const waiting = queue.wait().then(() => {
        resolved = true
      })

      await Promise.resolve()
      expect(resolved).toBe(false)

      const data: QueueData = { type: 'error', node: mockNode, error: new Error('a') }
      // Don't await send — it won't resolve until ack
      const sending = queue.send(data)

      await waiting
      expect(resolved).toBe(true)

      // Clean up: ack so send resolves
      queue.shift()!.ack()
      await sending
    })
  })

  describe('dispose', () => {
    it('resolves pending send acks and drains entries', async () => {
      let resolved = false
      const data: QueueData = { type: 'error', node: mockNode, error: new Error('a') }
      const sending = queue.send(data).then(() => {
        resolved = true
      })

      await Promise.resolve()
      expect(resolved).toBe(false)
      expect(queue.size).toBe(1)

      queue.dispose()

      await sending
      expect(resolved).toBe(true)
      expect(queue.size).toBe(0)
    })

    it('causes future send calls to resolve immediately', async () => {
      queue.dispose()

      const data: QueueData = { type: 'error', node: mockNode, error: new Error('a') }
      await queue.send(data)

      expect(queue.size).toBe(0)
    })
  })
})
