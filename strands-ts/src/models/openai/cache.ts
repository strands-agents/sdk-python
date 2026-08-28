/**
 * Shared prompt-caching translation for OpenAI adapters.
 *
 * OpenAI caches prompt prefixes automatically server-side and routes reads on a caller-supplied
 * `prompt_cache_key`. It exposes no cache-point placement knobs, so of `CacheConfig` only `cacheKey`
 * (and, when it already names a valid retention literal, `ttl`) maps onto the request.
 *
 * @internal
 */

import type { CacheConfig } from '../model.js'
import { logger } from '../../logging/logger.js'
import { warnOnce } from '../../logging/warn-once.js'

// OpenAI's prompt_cache_retention accepts only these literals. ttl maps through only on an exact
// match — the SDK never guesses a conversion from an arbitrary duration string.
type RetentionLiteral = 'in_memory' | '24h'
const RETENTION_LITERALS: ReadonlySet<string> = new Set(['in_memory', '24h'])

/**
 * The `CacheConfig`-derived fields both OpenAI request shapes share.
 */
interface CacheableRequest {
  prompt_cache_key?: string
  // Deprecated in openai 6.45.0 in favor of prompt_cache_options.ttl, whose only accepted value today
  // is '30m' — so '24h'/'in_memory' remain expressible only through this field.
  prompt_cache_retention?: RetentionLiteral | null
}

/**
 * The OpenAI caching values derived from a `CacheConfig`, before any wire casing or container choice.
 *
 * `cacheKey`/`retention` are the values to write; `unsupportedTtl` carries a `ttl` that named no
 * retention literal, so a caller warns only when it would otherwise have written the retention slot.
 *
 * @internal
 */
interface ResolvedOpenAICache {
  cacheKey?: string
  retention?: RetentionLiteral
  unsupportedTtl?: string
}

/**
 * Reports whether any placement field is set to something other than its default.
 *
 * OpenAI has no cache-point placement knobs, so these fields never reach the request; a configured
 * value signals a config carried over from a content-addressed provider and warrants a one-time nudge.
 */
function hasPlacementConfig(cacheConfig: CacheConfig): boolean {
  return (
    (cacheConfig.strategy !== undefined && cacheConfig.strategy !== 'auto') ||
    (cacheConfig.toolsTTL !== undefined && cacheConfig.toolsTTL !== true) ||
    (cacheConfig.systemPromptTTL !== undefined && cacheConfig.systemPromptTTL !== true) ||
    (cacheConfig.messagesTTL !== undefined && cacheConfig.messagesTTL !== true)
  )
}

/**
 * Resolves a `CacheConfig` into OpenAI caching values, independent of wire casing or container.
 *
 * Warns once about placement fields, which OpenAI cannot honor. The explicit-value-wins write and the
 * unsupported-`ttl` warning stay with the caller, since both depend on what already occupies the
 * caller's target request/options — see {@link warnUnsupportedRetention}.
 *
 * @param cacheConfig - The provider's configured cache settings.
 * @returns The cache key, the retention literal, and any unsupported `ttl` for the caller to warn on.
 * @internal
 */
export function resolveOpenAICache(cacheConfig: CacheConfig): ResolvedOpenAICache {
  const resolved: ResolvedOpenAICache = {}

  if (cacheConfig.cacheKey !== undefined) {
    resolved.cacheKey = cacheConfig.cacheKey
  }

  if (cacheConfig.ttl !== undefined) {
    if (RETENTION_LITERALS.has(cacheConfig.ttl)) resolved.retention = cacheConfig.ttl as RetentionLiteral
    else resolved.unsupportedTtl = cacheConfig.ttl
  }

  if (hasPlacementConfig(cacheConfig)) {
    warnOnce(
      logger,
      'openai caches prefixes automatically server-side | strategy, toolsTTL, systemPromptTTL and messagesTTL have no effect'
    )
  }

  return resolved
}

/**
 * Warns once that a `ttl` names no OpenAI retention literal and was ignored. Callers invoke this only
 * inside their own explicit-wins guard, so an explicit retention already on the target suppresses it.
 *
 * @param ttl - The `cacheConfig.ttl` value that matched no retention literal.
 * @internal
 */
export function warnUnsupportedRetention(ttl: string): void {
  warnOnce(logger, `ttl=<${ttl}> | cacheConfig.ttl is not an openai retention value, ignoring`)
}

/**
 * Maps a `CacheConfig` onto an OpenAI request in place.
 *
 * An explicit value already present in `request` (carried in from the user's `params`) always wins;
 * this fills in only what `params` did not set. A `ttl` that is not an OpenAI retention literal is
 * ignored, and the placement fields (`strategy`, `toolsTTL`, `systemPromptTTL`, `messagesTTL`) have no
 * effect here.
 *
 * @param request - The request being assembled; mutated in place.
 * @param cacheConfig - The provider's configured cache settings, if any.
 */
export function applyCacheConfig(request: CacheableRequest, cacheConfig: CacheConfig | undefined): void {
  if (!cacheConfig) return
  const resolved = resolveOpenAICache(cacheConfig)

  if (resolved.cacheKey !== undefined && request.prompt_cache_key === undefined) {
    request.prompt_cache_key = resolved.cacheKey
  }

  if (request.prompt_cache_retention === undefined) {
    if (resolved.retention !== undefined) request.prompt_cache_retention = resolved.retention
    else if (resolved.unsupportedTtl !== undefined) warnUnsupportedRetention(resolved.unsupportedTtl)
  }
}
