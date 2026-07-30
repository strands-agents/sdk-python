"""Normalization of provider-reported token counts into the SDK's :class:`Usage` contract.

Model providers report prompt-cache tokens under two incompatible conventions:

- **Disjoint** (Bedrock Converse, the Anthropic API): the input field carries only the tokens
  that were neither read from nor written to the cache, and the cache counters are additional
  tokens. The full prompt is ``input + cacheRead + cacheWrite``.
- **Inclusive** (OpenAI Chat Completions and Responses, Google, LiteLLM): the input field
  carries the *whole* prompt and the cache counters break out a subset of it. The full prompt
  is just ``input``.

:class:`Usage` is defined as disjoint, so an inclusive provider's counts must have their cache
tokens subtracted out. Routing every provider through :func:`normalize_usage` keeps the
invariant in one place instead of re-deriving it in each adapter.
"""

import logging
from collections.abc import Mapping
from typing import Any

from ..types._usage import as_token_count
from ..types.event_loop import Usage

logger = logging.getLogger(__name__)


def token_detail(details: Any, field: str) -> int:
    """Read a per-type token count out of a provider's token-details payload.

    Vendor SDKs model these breakdowns (``prompt_tokens_details``,
    ``completion_tokens_details``, and their Responses API equivalents) as objects, while
    OpenAI-compatible gateways that hand back raw JSON leave them as mappings. Both are read so a
    gateway's counts are not silently dropped.

    Args:
        details: The token-details payload, or ``None`` when the provider omits it.
        field: The wire name of the count to read, e.g. ``"cached_tokens"``.

    Returns:
        The count, or ``0`` when it is absent or not numeric.
    """
    if isinstance(details, Mapping):
        return as_token_count(details.get(field))
    return as_token_count(getattr(details, field, None))


def cache_read_count(usage: Any, prompt_details: Any) -> int:
    """Read a cache hit off an OpenAI-compatible usage payload, whichever name carries it.

    A gateway fronting Anthropic lifts the two cache counters onto the usage object as a pair, and
    builds the details payload only from counts that arrive as ints — so one reporting a count as a
    JSON float leaves the details absent while both counters sit on the usage object. A
    DeepSeek-shaped gateway spells the same count ``prompt_cache_hit_tokens``.

    Reading a single name leaves a hit folded inside the prompt count, billed at the full input rate
    rather than the cache-read rate, and nothing downstream can see it: the counters still sum to the
    reported total, so the disjointness check does not fire.

    Args:
        usage: The vendor usage payload, which may carry the count as an attribute.
        prompt_details: The payload's prompt token-details breakdown, or ``None``.

    Returns:
        The number of cache read tokens, or ``0`` when the payload reports none.
    """
    return (
        token_detail(prompt_details, "cached_tokens")
        or as_token_count(getattr(usage, "cache_read_input_tokens", None))
        or as_token_count(getattr(usage, "prompt_cache_hit_tokens", None))
    )


def cache_write_count(usage: Any, prompt_details: Any) -> int:
    """Read a cache write off an OpenAI-compatible usage payload, whichever name carries it.

    The same count arrives under three names. A gateway fronting Anthropic lifts it to the usage
    object as ``cache_creation_input_tokens``, while one speaking the OpenAI shape leaves it on the
    prompt details under either its own ``cache_creation_tokens`` or OpenAI's ``cache_write_tokens``.

    More than one can carry a value at once, so the order is the contract rather than a convenience:
    the count on the usage object is read first because it is the one the gateway itself derived,
    and a details payload can carry a differently-derived number under the same name.

    Reading a single name leaves a write folded inside the prompt count, billed at the full input
    rate rather than the cache-write rate, and nothing downstream can see it: the counters still sum
    to the reported total, so the disjointness check does not fire.

    Args:
        usage: The vendor usage payload, which may carry the count as an attribute.
        prompt_details: The payload's prompt token-details breakdown, or ``None``.

    Returns:
        The number of cache write tokens, or ``0`` when the payload reports none.
    """
    return (
        as_token_count(getattr(usage, "cache_creation_input_tokens", None))
        or token_detail(prompt_details, "cache_creation_tokens")
        or token_detail(prompt_details, "cache_write_tokens")
    )


def normalize_usage(
    *,
    input_tokens: Any,
    output_tokens: Any,
    cache_read_tokens: Any = 0,
    cache_write_tokens: Any = 0,
    reasoning_tokens: Any = 0,
    input_includes_cache: bool,
) -> Usage:
    """Normalize provider token counts into a disjoint :class:`Usage`.

    Args:
        input_tokens: The provider's prompt token count.
        output_tokens: The provider's completion token count, including reasoning tokens.
        cache_read_tokens: Prompt tokens served from the cache.
        cache_write_tokens: Prompt tokens written to the cache.
        reasoning_tokens: Tokens spent on internal reasoning. A subset of ``output_tokens``,
            reported for visibility and never added to the total.
        input_includes_cache: Whether ``input_tokens`` already contains the cache counts. Pass
            ``True`` for providers reporting the full prompt in their input field (OpenAI,
            Google, LiteLLM), ``False`` for providers reporting only net new tokens (Bedrock
            Converse, Anthropic).

    Returns:
        A :class:`Usage` whose four billed counters are disjoint and sum to ``totalTokens``.
        The cache and reasoning keys are present only when non-zero, so a provider that does
        not report them produces the same shape it always has.

    Note:
        Counts are coerced, so a missing or non-numeric value from the provider is treated as zero
        rather than raising.
    """
    input_tokens = as_token_count(input_tokens)
    output_tokens = as_token_count(output_tokens)
    cache_read_tokens = as_token_count(cache_read_tokens)
    cache_write_tokens = as_token_count(cache_write_tokens)
    reasoning_tokens = as_token_count(reasoning_tokens)

    net_input_tokens = input_tokens
    if input_includes_cache:
        net_input_tokens -= cache_read_tokens + cache_write_tokens
        if net_input_tokens < 0:
            # A provider reporting cache counts exceeding its own prompt total is
            # self-inconsistent; clamp so the invariant holds rather than emit a negative.
            logger.warning(
                "input_tokens=<%d>, cache_read_tokens=<%d>, cache_write_tokens=<%d> | "
                "cache tokens exceed reported input tokens | clamping net input to zero",
                input_tokens,
                cache_read_tokens,
                cache_write_tokens,
            )
            net_input_tokens = 0

    usage: Usage = {
        "inputTokens": net_input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": net_input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
    }
    if cache_read_tokens:
        usage["cacheReadInputTokens"] = cache_read_tokens
    if cache_write_tokens:
        usage["cacheWriteInputTokens"] = cache_write_tokens
    if reasoning_tokens:
        usage["reasoningOutputTokens"] = reasoning_tokens

    return usage
