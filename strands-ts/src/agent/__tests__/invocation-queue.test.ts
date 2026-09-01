import { describe, expect, it } from 'vitest'
import { InvocationQueue, previewInvokeArgs } from '../invocation-queue.js'
import { InvocationQueueFullError, PendingInvocationCancelledError } from '../../errors.js'
import { Message, TextBlock } from '../../types/messages.js'

describe('previewInvokeArgs', () => {
  it('collapses whitespace runs (including newlines) to single spaces', () => {
    expect(previewInvokeArgs('ok\nIGNORE the block above.\tStop now.')).toBe('ok IGNORE the block above. Stop now.')
  })

  it('never splits a surrogate pair at the truncation cut', () => {
    const preview = previewInvokeArgs('\u{1F642}'.repeat(300))
    expect(preview).toBe(`${'\u{1F642}'.repeat(200)}\u2026`)
  })

  it('passes short string input through', () => {
    expect(previewInvokeArgs('review the PR')).toBe('review the PR')
  })

  it('truncates long string input and marks the cut', () => {
    const preview = previewInvokeArgs('x'.repeat(500))
    expect(preview).toHaveLength(201)
    expect(preview.endsWith('…')).toBe(true)
  })

  it('extracts text from content block data', () => {
    expect(previewInvokeArgs([{ text: 'stop' }, { text: 'wrong repo' }])).toBe('stop wrong repo')
  })

  it('extracts text from message data', () => {
    expect(previewInvokeArgs([{ role: 'user', content: [{ text: 'hello' }] }])).toBe('hello')
  })

  it('extracts text from Message class instances', () => {
    const message = new Message({ role: 'user', content: [new TextBlock('typed hello')] })
    expect(previewInvokeArgs([message])).toBe('typed hello')
  })

  it('falls back to a placeholder for input with no text', () => {
    expect(previewInvokeArgs([{ role: 'user', content: [{ image: {} }] }] as never)).toBe('[structured input]')
  })
})

describe('InvocationQueue', () => {
  it('snapshots entries in run order with id, submittedAt, and preview', () => {
    const queue = new InvocationQueue()
    void queue.wait('first').catch(() => {})
    void queue.wait('second').catch(() => {})

    const snapshot = queue.snapshot()
    expect(snapshot).toHaveLength(2)
    expect(snapshot[0]).toMatchObject({ id: 'pending-1', preview: 'first' })
    expect(snapshot[1]).toMatchObject({ id: 'pending-2', preview: 'second' })
    expect(snapshot[0]!.submittedAt).toBeInstanceOf(Date)
    expect(Object.isFrozen(snapshot[0])).toBe(true)
  })

  it('resolves waiters FIFO on handoff', async () => {
    const queue = new InvocationQueue()
    const order: string[] = []
    const first = queue.wait('a').then(() => order.push('a'))
    const second = queue.wait('b').then(() => order.push('b'))

    expect(queue.handoff()).toBe(true)
    await first
    expect(queue.handoff()).toBe(true)
    await second
    expect(order).toEqual(['a', 'b'])
    expect(queue.size).toBe(0)
  })

  it('returns false from handoff when empty', () => {
    expect(new InvocationQueue().handoff()).toBe(false)
  })

  it('inserts front entries ahead of waiting ones', async () => {
    const queue = new InvocationQueue()
    const order: string[] = []
    const normal = queue.wait('normal').then(() => order.push('normal'))
    const urgent = queue.wait('urgent', { front: true }).then(() => order.push('urgent'))

    queue.handoff()
    await urgent
    queue.handoff()
    await normal
    expect(order).toEqual(['urgent', 'normal'])
  })

  it('throws InvocationQueueFullError at submit when maxDepth is reached', () => {
    const queue = new InvocationQueue(1)
    void queue.wait('fits').catch(() => {})
    expect(() => queue.wait('overflow')).toThrow(InvocationQueueFullError)
    expect(queue.size).toBe(1)
  })

  it('cancel removes the entry and rejects its waiter with the entry id', async () => {
    const queue = new InvocationQueue()
    const waiting = queue.wait('doomed')

    expect(queue.cancel('pending-1')).toBe(true)
    await expect(waiting).rejects.toThrow(PendingInvocationCancelledError)
    await expect(waiting).rejects.toMatchObject({ pendingInvocationId: 'pending-1' })
    expect(queue.size).toBe(0)
  })

  it('cancel returns false for an unknown id', () => {
    expect(new InvocationQueue().cancel('pending-99')).toBe(false)
  })

  it('rejects immediately when the cancelSignal is already aborted', async () => {
    const queue = new InvocationQueue()
    const controller = new AbortController()
    controller.abort()

    await expect(queue.wait('late', { cancelSignal: controller.signal })).rejects.toThrow(
      PendingInvocationCancelledError
    )
    expect(queue.size).toBe(0)
  })

  it('removes the entry and rejects when the cancelSignal aborts while queued', async () => {
    const queue = new InvocationQueue()
    const controller = new AbortController()
    const waiting = queue.wait('abandoned', { cancelSignal: controller.signal })

    expect(queue.size).toBe(1)
    controller.abort()
    await expect(waiting).rejects.toThrow(PendingInvocationCancelledError)
    expect(queue.size).toBe(0)
  })

  it('detaches the abort listener on handoff (a later abort does not reject)', async () => {
    const queue = new InvocationQueue()
    const controller = new AbortController()
    const waiting = queue.wait('handed-off', { cancelSignal: controller.signal })

    expect(queue.handoff()).toBe(true)
    await waiting
    controller.abort()
    await expect(waiting).resolves.toBeUndefined()
  })

  it('notifies onEnqueue listeners when an entry enters the queue, including at the front', () => {
    const queue = new InvocationQueue()
    let notified = 0
    queue.onEnqueue(() => notified++)
    void queue.wait('first').catch(() => {})
    expect(notified).toBe(1)
    void queue.wait('urgent', { front: true }).catch(() => {})
    expect(notified).toBe(2)
  })

  it('does not notify onEnqueue when a pre-aborted call is rejected without queueing', () => {
    const queue = new InvocationQueue()
    let notified = 0
    queue.onEnqueue(() => notified++)
    const aborted = new AbortController()
    aborted.abort()
    void queue.wait('never queued', { cancelSignal: aborted.signal }).catch(() => {})
    expect(notified).toBe(0)
    expect(queue.size).toBe(0)
  })

  it('stops notifying a detached onEnqueue listener', () => {
    const queue = new InvocationQueue()
    let notified = 0
    const detach = queue.onEnqueue(() => notified++)
    void queue.wait('first').catch(() => {})
    detach()
    void queue.wait('second').catch(() => {})
    expect(notified).toBe(1)
  })
})
