import { describe, expect, it } from 'vitest'
import { AsyncLock } from '../async-lock.js'

describe('AsyncLock', () => {
  it('serializes critical sections in FIFO order', async () => {
    const lock = new AsyncLock()
    const order: number[] = []

    async function task(id: number, delayMs: number): Promise<void> {
      const release = await lock.acquire()
      try {
        order.push(id)
        await new Promise((resolve) => setTimeout(resolve, delayMs))
      } finally {
        release()
      }
    }

    // Start three tasks "concurrently". Despite descending delays, the lock
    // forces them to run one at a time in acquisition order.
    await Promise.all([task(1, 15), task(2, 10), task(3, 5)])

    expect(order).toStrictEqual([1, 2, 3])
  })

  it('prevents overlap of holders', async () => {
    const lock = new AsyncLock()
    let active = 0
    let maxActive = 0

    async function task(): Promise<void> {
      const release = await lock.acquire()
      try {
        active++
        maxActive = Math.max(maxActive, active)
        await new Promise((resolve) => setTimeout(resolve, 5))
        active--
      } finally {
        release()
      }
    }

    await Promise.all([task(), task(), task(), task()])

    expect(maxActive).toBe(1)
  })

  it('releases the lock even if the holder throws', async () => {
    const lock = new AsyncLock()

    const release = await lock.acquire()
    try {
      throw new Error('boom')
    } catch {
      // swallow
    } finally {
      release()
    }

    // A subsequent acquire must resolve (would hang if the lock leaked).
    const release2 = await lock.acquire()
    release2()
    expect(true).toBe(true)
  })
})
