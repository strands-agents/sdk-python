import { describe, expect, it, vi, afterEach } from 'vitest'

import { logger } from '../../logging/logger.js'

import { cacheReadCount, cacheWriteCount, normalizeUsage } from '../usage.js'
import {
  accumulateUsage,
  contextTokenCount,
  promptTokenCount,
  readReportedUsage,
  repairPersistedUsage,
  warnIfUsageInconsistent,
} from '../streaming.js'
import type { Usage } from '../streaming.js'

/** Sums the four disjoint billed counters, the way a cost calculation would. */
function billedTotal(usage: Usage): number {
  return usage.inputTokens + usage.outputTokens + (usage.cacheReadInputTokens ?? 0) + (usage.cacheWriteInputTokens ?? 0)
}

describe('normalizeUsage', () => {
  describe('conventions', () => {
    it('keeps inputTokens and adds cache to the total for a disjoint provider', () => {
      expect(
        normalizeUsage({ inputTokens: 10, outputTokens: 4, cacheWriteTokens: 5848, inputIncludesCache: false })
      ).toEqual({
        inputTokens: 10,
        outputTokens: 4,
        totalTokens: 5862,
        cacheWriteInputTokens: 5848,
      })
    })

    it('subtracts cache out of inputTokens for an inclusive provider', () => {
      expect(
        normalizeUsage({ inputTokens: 2048, outputTokens: 256, cacheReadTokens: 1920, inputIncludesCache: true })
      ).toEqual({
        inputTokens: 128,
        outputTokens: 256,
        totalTokens: 2304,
        cacheReadInputTokens: 1920,
      })
    })
  })

  describe('optional fields', () => {
    it('omits cache and reasoning fields when zero', () => {
      expect(normalizeUsage({ inputTokens: 100, outputTokens: 20, inputIncludesCache: true })).toEqual({
        inputTokens: 100,
        outputTokens: 20,
        totalTokens: 120,
      })
    })

    it('reports reasoning tokens without inflating the total', () => {
      expect(
        normalizeUsage({ inputTokens: 100, outputTokens: 500, reasoningTokens: 400, inputIncludesCache: true })
      ).toEqual({
        inputTokens: 100,
        outputTokens: 500,
        totalTokens: 600,
        reasoningOutputTokens: 400,
      })
    })
  })

  describe('malformed provider counts', () => {
    it('clamps to zero rather than going negative when cache exceeds input', () => {
      expect(
        normalizeUsage({ inputTokens: 100, outputTokens: 10, cacheReadTokens: 500, inputIncludesCache: true })
      ).toEqual({
        inputTokens: 0,
        outputTokens: 10,
        totalTokens: 510,
        cacheReadInputTokens: 500,
      })
    })

    // Expected values match the Python SDK's as_token_count exactly; any divergence between the
    // two SDKs would silently change reported cost for one language only.
    it.each([
      { value: 7, expected: 7 },
      { value: '42', expected: 42 },
      { value: '12.5', expected: 12 },
      { value: '1e3', expected: 1000 },
      { value: Infinity, expected: 0 },
      { value: -Infinity, expected: 0 },
      { value: NaN, expected: 0 },
      { value: -5, expected: 0 },
      { value: 'abc', expected: 0 },
      // Forms only one language's own parser accepts, which each SDK rejects so the two agree:
      // Number() takes the prefixes and Python's float() takes the separator.
      { value: '0x10', expected: 0 },
      { value: '0o17', expected: 0 },
      { value: '0b101', expected: 0 },
      { value: '1_000', expected: 0 },
      // Separators Python's str.strip() removes but JavaScript's trim() does not, and the reverse.
      { value: '\x1c12', expected: 0 },
      { value: '\x8512', expected: 0 },
      { value: '٤٢', expected: 0 },
      // Past the range representable exactly, both SDKs read the count as absent.
      { value: 2 ** 53 - 1, expected: 2 ** 53 - 1 },
      { value: 2 ** 53, expected: 0 },
    ])('coerces $value to $expected, consistently with the Python SDK', ({ value, expected }) => {
      const usage = normalizeUsage({
        inputTokens: value as unknown as number,
        outputTokens: 0,
        inputIncludesCache: true,
      })

      expect(usage.inputTokens).toBe(expected)
    })

    // Elapsed time is the only observable, since the regex engine exposes no step count, but the
    // threshold is far from arbitrary: this reads in a few milliseconds, while the backtracking
    // shape it guards against takes seconds at a fifteenth of this length and grows fourfold each
    // time the length doubles.
    it('reads a long digit run with an invalid tail in linear time', () => {
      const started = Date.now()

      const usage = normalizeUsage({
        inputTokens: ('0'.repeat(500_000) + 'x') as unknown as number,
        outputTokens: 0,
        inputIncludesCache: true,
      })

      expect(usage.inputTokens).toBe(0)

      expect(Date.now() - started).toBeLessThan(5_000)
    })

    it('treats undefined counts as zero', () => {
      expect(normalizeUsage({ inputTokens: undefined, outputTokens: undefined, inputIncludesCache: true })).toEqual({
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
      })
    })
  })
})

describe('warnIfUsageInconsistent', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it.each([
    { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
    { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
    // Cache counters that account for the total, as every built-in adapter reports them.
    { inputTokens: 10, outputTokens: 4, totalTokens: 5862, cacheReadInputTokens: 5848 },
    { inputTokens: 2, outputTokens: 5, totalTokens: 6457, cacheWriteInputTokens: 6450 },
  ])('accepts conforming usage without warning', (usage) => {
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})

    warnIfUsageInconsistent(usage)

    expect(warn).not.toHaveBeenCalled()
  })

  // Expected outcomes match the Python SDK's _warn_if_usage_inconsistent exactly. Counts are
  // coerced the way a provider's own are, so malformed usage is reported on rather than crashing.
  it.each([
    { name: 'string counts', usage: { inputTokens: '10', outputTokens: '5', totalTokens: '15' }, warns: false },
    { name: 'null input', usage: { inputTokens: null, outputTokens: 5, totalTokens: 5 }, warns: false },
    {
      name: 'null cache counter',
      usage: { inputTokens: 12, outputTokens: 4, totalTokens: 16, cacheReadInputTokens: null },
      warns: false,
    },
    { name: 'absent total', usage: { inputTokens: 12, outputTokens: 4 }, warns: true },
    { name: 'null total', usage: { inputTokens: 12, outputTokens: 4, totalTokens: null }, warns: true },
  ])('handles $name without throwing', ({ usage, warns }) => {
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})

    warnIfUsageInconsistent(usage as never)

    expect(warn.mock.calls.length > 0).toBe(warns)
  })

  it('warns when the counters do not account for the total', () => {
    // A custom model reporting cache tokens inside inputTokens would inflate cost silently.
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})

    warnIfUsageInconsistent({
      inputTokens: 6452,
      outputTokens: 5,
      totalTokens: 6457,
      cacheReadInputTokens: 6450,
    })

    expect(warn).toHaveBeenCalledWith(expect.stringContaining('counted_tokens=<12907>, total_tokens=<6457>'))
  })
})

describe('readReportedUsage', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  // Expected outcomes match the Python SDK's extract_usage_metrics exactly.
  it.each([
    { name: 'conforming counts', usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 } },
    { name: 'a genuinely zero turn', usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } },
    {
      name: 'cache counters',
      usage: { inputTokens: 13, outputTokens: 4, totalTokens: 21019, cacheReadInputTokens: 21002 },
    },
    {
      name: 'a reasoning subset',
      usage: { inputTokens: 10, outputTokens: 100, totalTokens: 110, reasoningOutputTokens: 90 },
    },
    {
      // Bedrock Converse reports cacheDetails alongside the counters; it is a structured
      // breakdown rather than a count, so it must not read as an unusable number.
      name: 'a structured field alongside the counters',
      usage: {
        inputTokens: 13,
        outputTokens: 1,
        totalTokens: 21016,
        cacheWriteInputTokens: 21002,
        cacheDetails: [{ ttl: '5m', inputTokens: 21002 }],
      } as unknown as Usage,
    },
  ])('does not report $name as unusable', ({ usage }) => {
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})

    readReportedUsage(usage)

    expect(warn).not.toHaveBeenCalledWith(expect.stringContaining('not usable numbers'))
  })

  it('reports the counts it could not use and coerces them', () => {
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})

    const tru_usage = readReportedUsage({
      inputTokens: '9000',
      outputTokens: '100',
      totalTokens: '9100',
    } as unknown as Usage)

    // A count left as a string concatenates instead of adding, and the OpenTelemetry counters
    // drop it outright, so it is coerced before anything downstream reads it.
    expect(tru_usage).toEqual({ inputTokens: 9000, outputTokens: 100, totalTokens: 9100 })
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('fields=<inputTokens, outputTokens, totalTokens>'))
  })

  // A model that reports no usage at all commonly yields the field as null rather than omitting
  // it, and nothing constrains what it puts there. Both SDKs read that as no usage.
  it.each([null, undefined, 'usage', 42, [1, 2, 3], true])('reads %s as no usage', (reported) => {
    expect(readReportedUsage(reported as unknown as Usage)).toEqual({
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    })
  })

  it('warns about a count that cannot be serialized rather than throwing', () => {
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})
    const circular: Record<string, unknown> = { inputTokens: 'abc', outputTokens: 5, totalTokens: 15 }
    circular.self = circular

    // Reporting a count as a BigInt or a self-referencing object would make the payload itself
    // unserializable, and the point of reading it here is that nothing a model reports throws.
    expect(
      readReportedUsage({ inputTokens: BigInt(10), outputTokens: 5, totalTokens: 15 } as unknown as Usage)
    ).toEqual({ inputTokens: 0, outputTokens: 5, totalTokens: 15 })
    expect(readReportedUsage(circular as unknown as Usage)).toEqual({
      inputTokens: 0,
      outputTokens: 5,
      totalTokens: 15,
    })
    expect(warn).toHaveBeenCalledTimes(2)
  })

  // A counter read twice would report the difference between the reads as an unusable count. A
  // payload's own accessor need not answer with the same value twice, so the coercion and the
  // usability check must read from one snapshot rather than reading the payload each. Expected
  // outcomes match the Python SDK's extract_usage_metrics exactly.
  it('reads each counter once', () => {
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})
    const reads: Record<string, number> = {}
    const reported = new Proxy({ inputTokens: 10, outputTokens: 5, totalTokens: 15 } as Record<string, unknown>, {
      get(target, key: string) {
        reads[key] = (reads[key] ?? 0) + 1
        return target[key]
      },
    })

    const usage = readReportedUsage(reported as unknown as Usage)

    // The whole map, so the assertion still fails if a counter stops being read at all.
    expect(reads).toEqual({
      inputTokens: 1,
      outputTokens: 1,
      totalTokens: 1,
      cacheReadInputTokens: 1,
      cacheWriteInputTokens: 1,
      reasoningOutputTokens: 1,
    })
    expect(usage).toEqual({ inputTokens: 10, outputTokens: 5, totalTokens: 15 })
    expect(warn).not.toHaveBeenCalledWith(expect.stringContaining('not usable numbers'))
  })
})

describe('cacheReadCount', () => {
  // A gateway fronting Anthropic lifts both cache counters onto the usage object as a pair, and
  // builds the details payload only from counts that arrive as integers — so one reporting a float
  // or a string leaves the details absent while the usage object still carries the counters. Reading
  // only the details name bills a cache hit at the full input rate, which is the discount the cache
  // exists for, and nothing warns because the counters still sum. Matches Python's cache_read_count.
  it.each([
    { name: 'openai name on the details', usage: {}, details: { cached_tokens: 3000 } },
    { name: 'anthropic name on the usage', usage: { cache_read_input_tokens: 3000 }, details: undefined },
    { name: 'deepseek name on the usage', usage: { prompt_cache_hit_tokens: 3000 }, details: undefined },
    { name: 'float count with no details', usage: { cache_read_input_tokens: 3000.0 }, details: undefined },
  ])('reads a cache hit reported as the $name', ({ usage, details }) => {
    expect(cacheReadCount(usage, details)).toBe(3000)
  })

  it.each([
    { name: 'no cache hit at all', usage: {}, details: { cache_write_tokens: 500 } },
    { name: 'an explicit zero', usage: {}, details: { cached_tokens: 0 } },
    { name: 'a payload that is not an object', usage: undefined, details: undefined },
  ])('reads $name as no cache hit', ({ usage, details }) => {
    expect(cacheReadCount(usage, details)).toBe(0)
  })
})

describe('cacheWriteCount', () => {
  // A gateway speaking the OpenAI shape reports a cache write under one of three names, and the
  // OpenAI provider is the documented way to reach one. Reading a single name leaves the write
  // folded inside the prompt count, billed at the full input rate, and nothing downstream can see
  // it because the counters still sum. Expected values match the Python SDK's cache_write_count.
  it.each([
    { name: 'openai name on the details', usage: {}, details: { cache_write_tokens: 2000 } },
    { name: 'litellm name on the details', usage: {}, details: { cache_creation_tokens: 2000 } },
    { name: 'anthropic name on the usage', usage: { cache_creation_input_tokens: 2000 }, details: {} },
  ])('reads a cache write reported as the $name', ({ usage, details }) => {
    expect(cacheWriteCount(usage, details)).toBe(2000)
  })

  it.each([
    { name: 'no cache write at all', usage: {}, details: { cached_tokens: 3000 } },
    { name: 'an explicit zero', usage: {}, details: { cache_write_tokens: 0 } },
    { name: 'a payload that is not an object', usage: undefined, details: undefined },
  ])('reads $name as no cache write', ({ usage, details }) => {
    expect(cacheWriteCount(usage, details)).toBe(0)
  })
})

describe('accumulateUsage', () => {
  // A running total is a sum this SDK computed, not a count a provider claimed. Applying the
  // ceiling meant for a reported count to an accumulated one would reset the total once it grew
  // past that ceiling, discarding everything counted so far.
  it('keeps a running total past the reported-count ceiling', () => {
    const target: Usage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
    const source: Usage = { inputTokens: 0, outputTokens: 0, totalTokens: Number.MAX_SAFE_INTEGER }

    const totals: number[] = []
    for (let turn = 0; turn < 4; turn++) {
      accumulateUsage(target, source)
      totals.push(target.totalTokens)
    }

    expect(totals).toEqual([1, 2, 3, 4].map((turn) => Number.MAX_SAFE_INTEGER * turn))
  })
})

describe('contextTokenCount', () => {
  // Expected values match the Python SDK's context_token_count exactly. Understating the window
  // skips compaction and overflows the model instead, so it must never report less than the truth.
  it.each([
    // Live captures: Bedrock Converse, Bedrock Mantle Responses, and Google.
    { usage: { inputTokens: 10, outputTokens: 4, totalTokens: 5862, cacheReadInputTokens: 5848 }, expected: 5862 },
    { usage: { inputTokens: 2, outputTokens: 5, totalTokens: 6457, cacheReadInputTokens: 6450 }, expected: 6457 },
    { usage: { inputTokens: 8, outputTokens: 35, totalTokens: 23327, cacheReadInputTokens: 23284 }, expected: 23327 },
    { usage: { inputTokens: 100, outputTokens: 20, totalTokens: 120 }, expected: 120 },
    // A total that accounts for nothing must not shrink the estimate below the counters.
    { usage: { inputTokens: 100, outputTokens: 20, totalTokens: 0 }, expected: 120 },
  ])('reports $expected without understating the context window', ({ usage, expected }) => {
    expect(contextTokenCount(usage)).toBe(expected)
  })

  // A gateway reporting counts as strings must not concatenate them into a nonsense size.
  // Expected values match the Python SDK's context_token_count exactly.
  it.each([
    { usage: { inputTokens: '10', outputTokens: '5', totalTokens: '15' }, expected: 15 },
    { usage: { inputTokens: 10, outputTokens: 5, totalTokens: 105, cacheReadInputTokens: '90' }, expected: 105 },
    { usage: { inputTokens: 'abc', outputTokens: {}, totalTokens: [] }, expected: 0 },
  ])('coerces malformed counts to $expected', ({ usage, expected }) => {
    expect(contextTokenCount(usage as never)).toBe(expected)
    expect(typeof promptTokenCount(usage as never)).toBe('number')
  })

  it.each([
    // Recorded before cache tokens became separate counters, so inputTokens already contained
    // them. Reading high is the safe direction and self-corrects after the next model call.
    { usage: { inputTokens: 6452, outputTokens: 10, totalTokens: 6462, cacheReadInputTokens: 6450 }, atLeast: 6462 },
    { usage: { inputTokens: 5000, outputTokens: 100, totalTokens: 5100, cacheReadInputTokens: 1000 }, atLeast: 6100 },
  ])('never reports below $atLeast for usage recorded before the contract', ({ usage, atLeast }) => {
    expect(contextTokenCount(usage)).toBeGreaterThanOrEqual(atLeast)
  })

  // These helpers read counts directly rather than through normalizeUsage, so a negative count is
  // clamped where it is read. A window size below zero would shrink the estimate and skip the
  // compaction that keeps the next call inside the model's limit. Matches Python's context_token_count.
  it('clamps a negative count rather than shrinking the window', () => {
    const usage = { inputTokens: -500, outputTokens: 20, totalTokens: 0 } as Usage

    expect(promptTokenCount(usage)).toBe(0)
    expect(contextTokenCount(usage)).toBe(20)
  })
})

describe('repairPersistedUsage', () => {
  // Expected values match the Python SDK's _parse_usage exactly.
  it.each([
    // A cache count larger than inputTokens proves the counters are already disjoint, so the
    // total is the only thing that can be wrong.
    {
      usage: { inputTokens: 10, outputTokens: 4, totalTokens: 14, cacheReadInputTokens: 5848 },
      expected: { inputTokens: 10, outputTokens: 4, totalTokens: 5862, cacheReadInputTokens: 5848 },
    },
    {
      usage: { inputTokens: 10, outputTokens: 4, totalTokens: 14, cacheWriteInputTokens: 5848 },
      expected: { inputTokens: 10, outputTokens: 4, totalTokens: 5862, cacheWriteInputTokens: 5848 },
    },
  ])('repairs a total that omitted the cache counters', ({ usage, expected }) => {
    expect(repairPersistedUsage(usage)).toEqual(expected)
  })

  it.each([
    // Counters a pre-contract session may have omitted entirely, defaulted the way the Python SDK
    // defaults them.
    {
      usage: { outputTokens: 4, totalTokens: 4, cacheReadInputTokens: 100 },
      expected: { inputTokens: 0, outputTokens: 4, totalTokens: 104, cacheReadInputTokens: 100 },
    },
    {
      usage: { inputTokens: 10, outputTokens: 4, cacheReadInputTokens: 100 },
      expected: { inputTokens: 10, outputTokens: 4, totalTokens: 0, cacheReadInputTokens: 100 },
    },
  ])('defaults counters the payload omits', ({ usage, expected }) => {
    expect(repairPersistedUsage(usage as never)).toEqual(expected)
  })

  // Session state is user-editable, so the payload itself can be edited into something that is
  // not an object of counts. Both SDKs read that as no usage rather than raising out of restore.
  it.each([null, undefined, 'not-an-object', 42, [1, 2, 3]])('reads %s as no usage', (persisted) => {
    expect(repairPersistedUsage(persisted as never)).toEqual({
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    })
  })

  it('leaves the caller payload untouched and drops unknown keys', () => {
    const persisted = { inputTokens: 10, outputTokens: 4, totalTokens: 14, cacheReadInputTokens: 100, extra: 7 }

    const repaired = repairPersistedUsage(persisted as never)

    expect(repaired).toEqual({ inputTokens: 10, outputTokens: 4, totalTokens: 114, cacheReadInputTokens: 100 })
    expect(persisted.totalTokens).toBe(14)
  })

  it.each([
    // Ambiguous: whether inputTokens contained the cache count is unrecoverable, so nothing is
    // guessed at.
    { inputTokens: 5000, outputTokens: 4, totalTokens: 5004, cacheReadInputTokens: 3000 },
    { inputTokens: 100, outputTokens: 4, totalTokens: 104, cacheReadInputTokens: 80 },
    { inputTokens: 80, outputTokens: 4, totalTokens: 84, cacheReadInputTokens: 80 },
    // Already conforming.
    { inputTokens: 452, outputTokens: 10, totalTokens: 6462, cacheReadInputTokens: 6000 },
    { inputTokens: 10, outputTokens: 4, totalTokens: 5862, cacheReadInputTokens: 5848 },
    { inputTokens: 100, outputTokens: 20, totalTokens: 120 },
  ])('leaves $inputTokens input tokens untouched', (usage) => {
    expect(repairPersistedUsage({ ...usage })).toEqual(usage)
  })
})
