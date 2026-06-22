"""Shared logging helpers for the SDK."""

from __future__ import annotations

import logging

# Module-level set of messages already warned, scoped to the process so a recurring condition warns
# only once. Tests that exercise warn-once should clear it (``_logging._warned.clear()``) to reset the
# state between cases.
_warned: set[str] = set()


def warn_once(logger: logging.Logger, message: str, *args: object) -> None:
    """Emit a warning log at most once per unique message per process.

    Subsequent calls with the same ``message`` are no-ops, which prevents a repeated nudge from
    flooding logs when the same condition recurs (e.g. a per-call retry that keeps failing the same
    way). ``message`` is the dedupe key, so calls that should collapse together must share the same
    constant message; pass any varying context through ``args`` (lazy ``%s`` interpolation, matching
    the SDK's structured logging style) rather than baking it into ``message``.

    Args:
        logger: Logger to emit the warning on.
        message: Warning message; also used as the dedupe key.
        *args: Positional arguments interpolated into ``message`` via ``%s`` formatting.
    """
    if message in _warned:
        return
    _warned.add(message)
    logger.warning(message, *args)
