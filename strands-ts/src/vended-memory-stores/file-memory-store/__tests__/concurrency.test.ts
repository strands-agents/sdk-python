import { describe, it, expect } from 'vitest'
import { mapWithConcurrency } from '../concurrency.js'

describe('mapWithConcurrency', () => {
  it('preserves input order regardless of completion order', async () => {
    const delays = [30, 5, 20, 1]
    const result = await mapWithConcurrency(delays, 2, async (delay) => {
      await new Promise((resolve) => setTimeout(resolve, delay))
      return delay
    })

    expect(result).toEqual(delays)
  })

  it('never runs more than `limit` calls at once', async () => {
    let active = 0
    let peak = 0
    const items = Array.from({ length: 12 }, (_, index) => index)

    await mapWithConcurrency(items, 3, async (item) => {
      active++
      peak = Math.max(peak, active)
      await new Promise((resolve) => setTimeout(resolve, 5))
      active--
      return item
    })

    expect(peak).toBeLessThanOrEqual(3)
  })

  it('returns an empty array for no items', async () => {
    expect(await mapWithConcurrency([], 4, async (item) => item)).toEqual([])
  })

  it('propagates an error thrown by the worker function', async () => {
    await expect(
      mapWithConcurrency([1, 2, 3], 2, async (item) => {
        if (item === 2) throw new Error('boom')
        return item
      })
    ).rejects.toThrow('boom')
  })
})
