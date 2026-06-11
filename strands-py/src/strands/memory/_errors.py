"""Shared error helpers for the Strands memory module.

The ``AggregateMemoryError`` exception lives in the SDK's central
``strands.types.exceptions`` module alongside the other subsystem exceptions. It
is re-exported here so existing references via
``strands.memory._errors.AggregateMemoryError`` continue to resolve. This module
additionally provides the internal ``_flatten_reasons`` helper used when
surfacing multi-store failures.
"""

from ..types.exceptions import AggregateMemoryError

__all__ = ["AggregateMemoryError"]


def _flatten_reasons(reasons: list[BaseException]) -> list[BaseException]:
    """Flatten nested aggregate errors into their concrete leaf reasons.

    Mirrors the TypeScript ``_flattenReasons`` helper: any
    ``AggregateMemoryError`` in ``reasons`` is replaced by its own (recursively
    flattened) ``errors`` so the result holds concrete underlying errors rather
    than aggregates-of-aggregates.

    Args:
        reasons: The exceptions to flatten, possibly containing nested
            ``AggregateMemoryError`` instances.

    Returns:
        A flat list of concrete leaf exceptions, with no ``AggregateMemoryError``
        instances remaining.
    """
    flattened: list[BaseException] = []
    for reason in reasons:
        if isinstance(reason, AggregateMemoryError):
            flattened.extend(_flatten_reasons(reason.errors))
        else:
            flattened.append(reason)
    return flattened
