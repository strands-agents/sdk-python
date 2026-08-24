"""Process-global warn-once logging.

Mirrors the TypeScript SDK's ``warnOnce`` so both SDKs suppress repeated nudges identically.
"""

import logging
from typing import Any

# Formatted messages already warned this process, keyed on the interpolated text.
_warned: set[str] = set()


def warn_once(logger: logging.Logger, message: str, *args: Any) -> None:
    """Emit a warning at most once per unique formatted message per process.

    Subsequent calls with the same formatted message are no-ops, which keeps a repeated nudge
    (e.g. an ignored config field) from flooding logs when many instances are constructed.

    Args:
        logger: Logger to emit the warning on.
        message: ``%s``-style log message; once interpolated it is also the dedupe key.
        *args: Interpolation arguments for ``message``.
    """
    key = message % args if args else message
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)
