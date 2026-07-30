"""Internal helpers for reading and combining :class:`~strands.types.event_loop.Usage` counts."""

import math
import re
from collections.abc import Mapping
from typing import Any

from .event_loop import Usage

_DECIMAL_NUMBER = re.compile(r"[ \t\n\r\f\v]*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?[ \t\n\r\f\v]*")
"""A number in the decimal forms JSON can carry, which is the overlap of both SDKs' parsing.

Every character is spelled out rather than left to ``\\d`` and :meth:`str.strip`, both of which
accept more than their JavaScript counterparts: ``\\d`` matches non-ASCII digits, and ``strip``
removes separators such as ``\\x1c`` that leave a remainder :func:`float` then rejects.

The fraction nests inside the integer alternative rather than following it, so a long run of digits
with an invalid tail is retried once per digit rather than once per split point, which would cost
time quadratic in the length of a count a gateway could send.
"""

_MAX_EXACT_COUNT = 2**53 - 1
"""The largest count both SDKs carry exactly, being JavaScript's ``Number.MAX_SAFE_INTEGER``."""

USAGE_COUNTERS = (
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "cacheReadInputTokens",
    "cacheWriteInputTokens",
    "reasoningOutputTokens",
)
"""The token counters :class:`Usage` declares. Providers may report other fields alongside them."""


def as_token_count(value: Any) -> int:
    """Coerce a provider-reported token count to a non-negative int.

    Providers hand back ``None`` for counters they do not populate, and OpenAI-compatible
    gateways occasionally return a string. Anything non-numeric is treated as absent so a
    malformed count degrades the metric instead of raising through the stream.

    Args:
        value: The raw value reported by the provider.

    Returns:
        The value as a non-negative int, or ``0`` if it is absent or not numeric.
    """
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return 0
        if isinstance(value, str):
            # Matching first keeps the two SDKs agreeing on every string: float() would otherwise
            # take the "1_000" JavaScript's Number() rejects, and Number() would take the "0x10"
            # prefixes float() rejects. Parsing as a float accepts the "12.5" and "1e3" forms
            # Number() does.
            if not _DECIMAL_NUMBER.fullmatch(value):
                return 0
            value = float(value)
        if isinstance(value, float) and not math.isfinite(value):
            return 0
        # An int is compared before converting: JSON carries arbitrary precision, and converting
        # one too large for a float would raise rather than degrade. Past what JavaScript
        # represents exactly a count is read as absent, since one the TypeScript SDK cannot carry
        # is not one this SDK should report either.
        return max(0, int(value)) if value <= _MAX_EXACT_COUNT else 0
    except Exception:  # noqa: BLE001 - a count describes an invocation and must never fail one
        # A mock or proxy standing in for a count answers isinstance without supporting the
        # arithmetic, and a model's reply is already complete by the time its counts are read.
        return 0


def _accumulated(count: Any) -> int:
    """Read a running total this SDK computed, which has no ceiling a reported count would have."""
    return count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else as_token_count(count)


def usage_count(payload: Any, field: str) -> Any:
    """Read one counter out of ``payload``, whatever shape it arrived in.

    A vendor usage object carries its counters as attributes while persisted session state and an
    OpenAI-compatible gateway carry them as keys, so both are read. Anything else, and any accessor
    that raises, reads as absent rather than failing the invocation the counts merely describe.

    Args:
        payload: The usage to read from, which need not be a mapping or an object.
        field: The name of the counter to read.

    Returns:
        The raw value reported for the counter, or ``None`` when it is absent or unreadable.
    """
    try:
        return payload.get(field) if isinstance(payload, Mapping) else getattr(payload, field, None)
    except Exception:  # noqa: BLE001 - an accessor that raises must not fail the invocation
        return None


def coerce_usage_counters(payload: Any) -> Usage:
    """Read the counters :class:`Usage` declares out of an arbitrary payload.

    A model implementation and a persisted session both reach the SDK as untyped data, so every
    count is coerced. The result is a new :class:`Usage`, leaving the caller's payload alone and
    keeping keys the contract does not declare out of the SDK's own type.

    Args:
        payload: The usage to read. A mapping and an object carrying the counters as attributes are
            both read, since a model implementation may hand back its vendor's own usage object
            while session state arrives as JSON. Anything else is read as no usage at all.

    Returns:
        The declared counters as non-negative ints, with absent optional ones left absent.
    """
    usage = Usage(
        inputTokens=as_token_count(usage_count(payload, "inputTokens")),
        outputTokens=as_token_count(usage_count(payload, "outputTokens")),
        totalTokens=as_token_count(usage_count(payload, "totalTokens")),
    )
    for key in ("cacheReadInputTokens", "cacheWriteInputTokens", "reasoningOutputTokens"):
        # A counter reported as None is skipped, so an absent one stays absent rather than reading
        # as a real zero.
        raw = usage_count(payload, key)
        if raw is not None:
            usage[key] = as_token_count(raw)
    return usage


def prompt_token_count(usage: Usage) -> int:
    """Return the full prompt size, including tokens served from or written to the cache.

    ``Usage.inputTokens`` counts only net new tokens, but cached tokens occupy the model's
    context window and are part of the prompt that was sent, so anything reasoning about prompt
    size (rather than cost) needs all three counters.

    Args:
        usage: The usage reported by the model.

    Returns:
        The total number of prompt tokens sent to the model.
    """
    return (
        as_token_count(usage.get("inputTokens"))
        + as_token_count(usage.get("cacheReadInputTokens"))
        + as_token_count(usage.get("cacheWriteInputTokens"))
    )


def context_token_count(usage: Usage) -> int:
    """Return how much of the context window a model call consumed, prompt plus generated output.

    Cached tokens occupy the context window like any other, so the whole prompt counts, not just
    the net new tokens ``inputTokens`` reports.

    Usage recorded before cache tokens became separate counters counted them inside
    ``inputTokens``. Such a payload is only sometimes distinguishable from a current one, so no
    attempt is made here: it reads high by the size of its cache hit, which is the safe direction —
    it compacts a conversation early rather than overflowing the window — and it corrects itself
    after the first model call of a resumed session.

    Args:
        usage: The usage reported by the model.

    Returns:
        The number of context-window tokens the call accounted for.
    """
    return prompt_token_count(usage) + as_token_count(usage.get("outputTokens"))


def accumulate_usage(target: Usage, source: Usage) -> None:
    """Add ``source`` token counts into ``target`` in place.

    Every counter is carried across, so summing usages that each satisfy the :class:`Usage`
    invariant yields a total that satisfies it too. Dropping the optional counters here would
    leave ``totalTokens`` counting cache tokens that no individual counter reports.

    The required counters are read defensively because a multi-agent node may report usage
    assembled outside a model provider, where they are not guaranteed to be present.

    Args:
        target: The usage to accumulate into.
        source: The usage to add.
    """
    target["inputTokens"] = _accumulated(target.get("inputTokens")) + as_token_count(source.get("inputTokens"))
    target["outputTokens"] = _accumulated(target.get("outputTokens")) + as_token_count(source.get("outputTokens"))
    target["totalTokens"] = _accumulated(target.get("totalTokens")) + as_token_count(source.get("totalTokens"))

    for key in ("cacheReadInputTokens", "cacheWriteInputTokens", "reasoningOutputTokens"):
        if key in source:
            target[key] = _accumulated(target.get(key)) + as_token_count(source[key])
