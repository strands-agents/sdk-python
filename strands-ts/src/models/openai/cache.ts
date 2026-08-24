/**
 * Shared prompt-caching translation for OpenAI adapters.
 *
 * OpenAI caches prompt prefixes automatically server-side and routes reads on a caller-supplied
 * `prompt_cache_key`. It exposes no cache-point placement knobs, so of `CacheConfig` only `cacheKey`
 * (and, when it already names a valid retention literal, `ttl`) maps onto the request; the placement
 * fields (`strategy`, `toolsTTL`, `systemPromptTTL`, `messagesTTL`) have no effect here.
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
  prompt_cache_retention?: 'in_memory' | '24h' | null
}

/**
 * Maps a `CacheConfig` onto an OpenAI request in place.
 *
 * An explicit value already present in `request` (carried in from the user's `params`) always wins;
 * this fills in only what `params` did not set. A `ttl` that is not an OpenAI retention literal is
 * ignored, warned once per process to avoid flooding logs.
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
      warnOnce(
        logger,
        `ttl=<${cacheConfig.ttl}> | cacheConfig.ttl is not an openai retention value (in_memory, 24h), ignoring`
      )
    }
  }
}
