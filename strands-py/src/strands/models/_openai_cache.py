"""Shared prompt-caching translation for OpenAI model providers.

OpenAI caches prompt prefixes automatically server-side and routes reads on a caller-supplied
``prompt_cache_key``. It exposes no cache-point placement knobs, so of ``CacheConfig`` only
``cache_key`` (and, when it already names a valid retention literal, ``ttl``) maps onto the request.
"""

import logging
from typing import Any

from ..logging.warn_once import warn_once
from .model import CacheConfig

logger = logging.getLogger(__name__)

# OpenAI's prompt_cache_retention accepts only these literals. ttl maps through only on an exact
# match - the SDK never guesses a conversion from an arbitrary duration string.
#
# prompt_cache_retention is deprecated in openai 2.54.0 in favor of prompt_cache_options.ttl, whose only
# accepted value today is "30m" - so "24h"/"in_memory" remain expressible only through this field.
_RETENTION_LITERALS = frozenset({"in_memory", "24h"})


def apply_cache_config(request: dict[str, Any], cache_config: CacheConfig | None) -> None:
    """Map a ``CacheConfig`` onto an OpenAI request in place.

    An explicit value already present in ``request`` (carried in from the user's ``params``) always
    wins; this fills in only what ``params`` did not set. ``strategy`` and ``system_prompt_ttl`` are
    accepted but have no effect, and a ``ttl`` that is not an OpenAI retention literal is ignored;
    each such no-op is warned once per process.

    Args:
        request: The request dict being assembled; mutated in place.
        cache_config: The provider's configured cache settings, if any.
    """
    if cache_config is None:
        return

    if cache_config.cache_key is not None and "prompt_cache_key" not in request:
        request["prompt_cache_key"] = cache_config.cache_key

    if cache_config.ttl is not None and "prompt_cache_retention" not in request:
        if cache_config.ttl in _RETENTION_LITERALS:
            request["prompt_cache_retention"] = cache_config.ttl
        else:
            warn_once(
                logger,
                "ttl=<%s> | cache_config.ttl is not an openai retention value, ignoring",
                cache_config.ttl,
            )

    if cache_config.strategy != "auto" or cache_config.system_prompt_ttl is not True:
        warn_once(
            logger, "openai caches prefixes automatically server-side | strategy and system_prompt_ttl have no effect"
        )
