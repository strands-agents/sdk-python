"""Monotonic UUIDv7 generation and validation for immutable snapshot ids.

Immutable snapshot ids are UUID version 7 (RFC 9562), matching the TypeScript SDK. The 48-bit
millisecond timestamp in the high bits makes lexicographic order equal creation order, so
listing returns oldest-first without a separate index. Ids are opaque handles that callers get
from ``list_snapshot_ids`` and never construct.
"""

import os
import re
import threading
import time
import uuid

# ``\A...\Z`` (not ``^...$``) so a trailing newline cannot slip past validation into a storage key.
_SNAPSHOT_ID_PATTERN = re.compile(r"\A[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")

# UUIDv7 monotonic counter state (RFC 9562 method 2): within a single millisecond, ids that
# would otherwise collide are ordered by a counter seeded into the random bits, so sorting by
# the id preserves creation order even for sub-millisecond bursts. Guarded by a lock because
# snapshots can be written from the sync bridge and async paths on different threads.
_lock = threading.Lock()
_last_ms = -1
_counter = 0


def new_snapshot_id() -> str:
    """Return a fresh, monotonic UUIDv7 immutable snapshot id.

    Monotonic across calls so lexicographic order equals creation order, even for bursts that
    mint more than 4096 ids inside one millisecond.
    """
    global _last_ms, _counter

    with _lock:
        now_ms = int(time.time() * 1000)
        if now_ms > _last_ms:
            _last_ms = now_ms
            # Seed in the low half of the 12-bit field (fresh randomness for uniqueness) so at
            # least 2048 increments fit before overflow.
            _counter = int.from_bytes(os.urandom(2), "big") & 0x07FF
        else:
            _counter += 1
            if _counter > 0x0FFF:
                # Counter exhausted within this millisecond: borrow from the clock by advancing
                # the timestamp 1ms (RFC 9562 method 1), keeping ids strictly increasing rather
                # than wrapping the counter field backwards.
                _last_ms += 1
                _counter = int.from_bytes(os.urandom(2), "big") & 0x07FF
        counter = _counter
        timestamp_ms = _last_ms

    # Layout: 48-bit ms timestamp | version(7) | 12-bit counter | variant(0b10) | 62 random bits.
    tail = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)
    value = (timestamp_ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= (counter & 0x0FFF) << 64
    value |= 0b10 << 62
    value |= tail
    return str(uuid.UUID(int=value))


def validate_snapshot_id(snapshot_id: str) -> None:
    """Validate that a string is an SDK-vended snapshot id (a UUIDv7).

    Args:
        snapshot_id: The string to validate.

    Raises:
        ValueError: If the string is not a valid snapshot id.
    """
    if not _SNAPSHOT_ID_PATTERN.match(snapshot_id):
        raise ValueError(f"'{snapshot_id}' is not a valid snapshot id")
