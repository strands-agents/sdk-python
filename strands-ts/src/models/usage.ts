/**
 * Normalization of provider-reported token counts into the SDK's {@link Usage} contract.
 *
 * Model providers report prompt-cache tokens under two incompatible conventions:
 *
 * - **Disjoint** (Bedrock Converse, the Anthropic API): the input field carries only the tokens
 *   that were neither read from nor written to the cache, and the cache counters are additional
 *   tokens. The full prompt is `input + cacheRead + cacheWrite`.
 * - **Inclusive** (OpenAI Chat Completions and Responses, Google, LiteLLM): the input field
 *   carries the *whole* prompt and the cache counters break out a subset of it. The full prompt
 *   is just `input`.
 *
 * {@link Usage} is defined as disjoint, so an inclusive provider's counts must have their cache
 * tokens subtracted out. Routing every adapter through {@link normalizeUsage} keeps the invariant
 * in one place instead of re-deriving it per adapter.
 *
 * @internal
 */

import { logger } from '../logging/logger.js'
import type { Usage } from './streaming.js'

/**
 * Provider-reported token counts, before normalization.
 *
 * @internal
 */
export interface RawUsage {
  /** The provider's prompt token count. */
  inputTokens: number | undefined
  /** The provider's completion token count, including reasoning tokens. */
  outputTokens: number | undefined
  /** Prompt tokens served from the cache. */
  cacheReadTokens?: number | undefined
  /** Prompt tokens written to the cache. */
  cacheWriteTokens?: number | undefined
  /** Tokens spent on internal reasoning; a subset of `outputTokens`. */
  reasoningTokens?: number | undefined
  /**
   * Whether `inputTokens` already contains the cache counts. Pass `true` for providers reporting
   * the full prompt in their input field (OpenAI, Google, LiteLLM), `false` for providers
   * reporting only net new tokens (Bedrock Converse, Anthropic).
   */
  inputIncludesCache: boolean
}

/**
 * A number in the decimal forms JSON can carry, which is the overlap of both SDKs' parsing.
 *
 * Every character is spelled out rather than left to `\d` and `trim()`, which do not accept the
 * same set as their Python counterparts.
 *
 * The fraction nests inside the integer alternative rather than following it, so a long run of
 * digits with an invalid tail is retried once per digit rather than once per split point, which
 * would cost time quadratic in the length of a count a gateway could send.
 */
const DECIMAL_NUMBER = /^[ \t\n\r\f\v]*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?[ \t\n\r\f\v]*$/

/**
 * Coerces a provider-reported token count to a non-negative integer.
 *
 * Providers omit counters they do not populate, and OpenAI-compatible gateways occasionally
 * return a string. Anything non-numeric is treated as absent so a malformed count degrades the
 * metric instead of throwing mid-stream.
 *
 * @param value - The raw value reported by the provider
 * @returns The value as a non-negative integer, or `0` when absent or not numeric
 *
 * @internal
 */
export function asTokenCount(value: unknown): number {
  let parsed = value
  if (typeof parsed === 'string') {
    // Matching first keeps the two SDKs agreeing on every string: Number() would otherwise take
    // the "0x10" prefixes Python's float() rejects, and float() would take the "1_000" separators
    // Number() rejects.
    if (!DECIMAL_NUMBER.test(parsed)) {
      return 0
    }
    parsed = Number(parsed)
  }
  // Past what JavaScript represents exactly a count is read as absent rather than silently
  // rounded, which is also where the Python SDK stops so the two report the same number.
  if (typeof parsed !== 'number' || !Number.isFinite(parsed) || parsed > Number.MAX_SAFE_INTEGER) {
    return 0
  }
  return Math.max(0, Math.trunc(parsed))
}

/**
 * Reads a cache hit off an OpenAI-compatible usage payload, whichever name carries it.
 *
 * A gateway fronting Anthropic lifts the two cache counters onto the usage object as a pair, and
 * builds the details payload only from counts that arrive as integers — so one reporting a count as
 * a JSON float leaves the details absent while both counters sit on the usage object. A
 * DeepSeek-shaped gateway spells the same count `prompt_cache_hit_tokens`.
 *
 * Reading a single name leaves a hit folded inside the prompt count, billed at the full input rate
 * rather than the cache-read rate, and nothing downstream can see it: the counters still sum to the
 * reported total, so the disjointness check does not fire.
 *
 * @param usage - The vendor usage payload, which may carry the count as a property
 * @param promptDetails - The payload's prompt token-details breakdown
 * @returns The number of cache read tokens, or `0` when the payload reports none
 *
 * @internal
 */
export function cacheReadCount(usage: unknown, promptDetails: unknown): number {
  const onUsage = usage as { cache_read_input_tokens?: unknown; prompt_cache_hit_tokens?: unknown } | null | undefined
  return (
    asTokenCount((promptDetails as { cached_tokens?: unknown } | null | undefined)?.cached_tokens) ||
    asTokenCount(onUsage?.cache_read_input_tokens) ||
    asTokenCount(onUsage?.prompt_cache_hit_tokens)
  )
}

/**
 * Reads a cache write off an OpenAI-compatible usage payload, whichever name carries it.
 *
 * The same count arrives under three names. A gateway fronting Anthropic lifts it to the usage
 * object as `cache_creation_input_tokens`, while one speaking the OpenAI shape leaves it on the
 * prompt details under either its own `cache_creation_tokens` or OpenAI's `cache_write_tokens`.
 *
 * More than one can carry a value at once, so the order is the contract rather than a convenience:
 * the count on the usage object is read first because it is the one the gateway itself derived, and
 * a details payload can carry a differently-derived number under the same name.
 *
 * Reading a single name leaves a write folded inside the prompt count, billed at the full input
 * rate rather than the cache-write rate, and nothing downstream can see it: the counters still sum
 * to the reported total, so the disjointness check does not fire.
 *
 * @param usage - The vendor usage payload, which may carry the count as a property
 * @param promptDetails - The payload's prompt token-details breakdown
 * @returns The number of cache write tokens, or `0` when the payload reports none
 *
 * @internal
 */
export function cacheWriteCount(usage: unknown, promptDetails: unknown): number {
  const onUsage = (usage as { cache_creation_input_tokens?: unknown } | null | undefined)?.cache_creation_input_tokens
  const details = promptDetails as { cache_creation_tokens?: unknown; cache_write_tokens?: unknown } | null | undefined
  return (
    asTokenCount(onUsage) || asTokenCount(details?.cache_creation_tokens) || asTokenCount(details?.cache_write_tokens)
  )
}

/**
 * Normalizes provider token counts into a disjoint {@link Usage}.
 *
 * The cache and reasoning fields are set only when non-zero, so a provider that does not report
 * them produces the same shape it always has.
 *
 * @param raw - The provider-reported counts and the convention they follow
 * @returns A `Usage` whose four billed counters are disjoint and sum to `totalTokens`
 *
 * @internal
 */
export function normalizeUsage(raw: RawUsage): Usage {
  const inputTokens = asTokenCount(raw.inputTokens)
  const outputTokens = asTokenCount(raw.outputTokens)
  const cacheReadTokens = asTokenCount(raw.cacheReadTokens)
  const cacheWriteTokens = asTokenCount(raw.cacheWriteTokens)
  const reasoningTokens = asTokenCount(raw.reasoningTokens)

  let netInputTokens = inputTokens
  if (raw.inputIncludesCache) {
    netInputTokens = inputTokens - cacheReadTokens - cacheWriteTokens
    if (netInputTokens < 0) {
      // A provider reporting cache counts exceeding its own prompt total is self-inconsistent;
      // clamp so the invariant holds rather than emit a negative.
      logger.warn(
        `input_tokens=<${inputTokens}>, cache_read_tokens=<${cacheReadTokens}>, ` +
          `cache_write_tokens=<${cacheWriteTokens}> | cache tokens exceed reported input tokens ` +
          `| clamping net input to zero`
      )
      netInputTokens = 0
    }
  }

  return {
    inputTokens: netInputTokens,
    outputTokens,
    totalTokens: netInputTokens + outputTokens + cacheReadTokens + cacheWriteTokens,
    ...(cacheReadTokens > 0 && { cacheReadInputTokens: cacheReadTokens }),
    ...(cacheWriteTokens > 0 && { cacheWriteInputTokens: cacheWriteTokens }),
    ...(reasoningTokens > 0 && { reasoningOutputTokens: reasoningTokens }),
  }
}
