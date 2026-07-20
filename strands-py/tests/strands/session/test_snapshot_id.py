"""Tests for monotonic UUIDv7 snapshot-id generation and validation."""

import uuid
from unittest.mock import patch

import pytest

from strands.session import _snapshot_id


def test_new_snapshot_id_is_valid_uuidv7():
    """Generated ids are valid version-7 UUIDs and pass validation."""
    snapshot_id = _snapshot_id.new_snapshot_id()
    parsed = uuid.UUID(snapshot_id)
    assert parsed.version == 7
    assert (parsed.int >> 62) & 0b11 == 0b10  # RFC 4122 variant
    _snapshot_id.validate_snapshot_id(snapshot_id)  # does not raise


def test_ids_sort_in_creation_order():
    """Ids minted in sequence sort lexicographically into creation order."""
    ids = [_snapshot_id.new_snapshot_id() for _ in range(1000)]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_ids_stay_monotonic_past_counter_overflow():
    """Minting >4096 ids in one millisecond keeps them unique, valid v7, and strictly increasing.

    Guards the 12-bit intra-millisecond counter: on overflow it must borrow from the clock, not
    wrap backwards (which would break the oldest-first contract of list_snapshot_ids).
    """
    with patch.object(_snapshot_id.time, "time", return_value=1_000.0):
        ids = [_snapshot_id.new_snapshot_id() for _ in range(20_000)]

    assert len(set(ids)) == len(ids)  # unique
    assert all(uuid.UUID(snapshot_id).version == 7 for snapshot_id in ids)  # valid v7
    assert all((uuid.UUID(snapshot_id).int >> 62) & 0b11 == 0b10 for snapshot_id in ids)  # variant intact
    assert all(ids[index] < ids[index + 1] for index in range(len(ids) - 1))  # strictly increasing


@pytest.mark.parametrize(
    "bad_id",
    [
        "not-a-uuid",
        "00000000-0000-4000-8000-000000000000",  # version 4, not 7
        "",
    ],
)
def test_validate_rejects_non_uuidv7(bad_id):
    """Malformed strings and non-v7 UUIDs are rejected."""
    with pytest.raises(ValueError, match="not a valid snapshot id"):
        _snapshot_id.validate_snapshot_id(bad_id)


def test_validate_rejects_trailing_newline():
    """A trailing newline cannot slip past validation (\\A...\\Z, not ^...$)."""
    with pytest.raises(ValueError, match="not a valid snapshot id"):
        _snapshot_id.validate_snapshot_id(_snapshot_id.new_snapshot_id() + "\n")
