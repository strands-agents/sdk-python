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
const RETENTION_LITERALS: ReadonlySet<string> = new Set(['in_memory', '24h'])

/**
 * The `CacheConfig`-derived fields both OpenAI request shapes share.
 */
interface CacheableRequest {
  prompt_cache_key?: string
  // Deprecated in openai 6.45.0 in favor of prompt_cache_options.ttl, whose only accepted value today
  // is '30m' — so '24h'/'in_memory' remain expressible only through this field.
  prompt_cache_retention?: 'in_memory' | '24h' | null
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

  if (cacheConfig.cacheKey !== undefined && request.prompt_cache_key === undefined) {
    request.prompt_cache_key = cacheConfig.cacheKey
  }

  if (cacheConfig.ttl !== undefined && request.prompt_cache_retention === undefined) {
    if (RETENTION_LITERALS.has(cacheConfig.ttl)) {
      request.prompt_cache_retention = cacheConfig.ttl as 'in_memory' | '24h'
    } else {
      warnOnce(logger, `ttl=<${cacheConfig.ttl}> | cacheConfig.ttl is not an openai retention value, ignoring`)
    }
  }

  if (hasPlacementConfig(cacheConfig)) {
    warnOnce(
      logger,
      'openai caches prefixes automatically server-side | strategy, toolsTTL, systemPromptTTL and messagesTTL have no effect'
    )
  }
}
