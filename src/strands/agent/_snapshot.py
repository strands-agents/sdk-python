"""Snapshot types, constants, and helpers for agent state capture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from ..types.exceptions import SnapshotException

SnapshotField = Literal["messages", "state", "conversation_manager_state", "interrupt_state", "system_prompt"]
SnapshotPreset = Literal["session"]

ALL_SNAPSHOT_FIELDS: tuple[SnapshotField, ...] = (
    "messages",
    "state",
    "conversation_manager_state",
    "interrupt_state",
    "system_prompt",
)

SNAPSHOT_SCHEMA_VERSION = "1.0"

SNAPSHOT_PRESETS: dict[str, tuple[SnapshotField, ...]] = {
    "session": ("messages", "state", "conversation_manager_state", "interrupt_state"),
}


class TakeSnapshotOptions(TypedDict, total=False):
    """Internal options for take_snapshot. Not exported publicly."""

    preset: SnapshotPreset
    include: list[SnapshotField]
    exclude: list[SnapshotField]
    app_data: dict[str, Any]


@dataclass
class Snapshot:
    """Point-in-time capture of agent state as a versioned JSON-compatible object."""

    schema_version: str
    created_at: str  # ISO 8601 UTC
    data: dict[str, Any]
    app_data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-compatible dict."""
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "data": self.data,
            "app_data": self.app_data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Snapshot":
        """Reconstruct a Snapshot from a dict produced by to_dict().

        Raises:
            SnapshotException: If schema_version is not "1.0".
        """
        version = d.get("schema_version", "")
        if version != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotException(
                f"Unsupported snapshot schema version: {version!r}. Current version: {SNAPSHOT_SCHEMA_VERSION}"
            )
        return cls(
            schema_version=d["schema_version"],
            created_at=d["created_at"],
            data=d["data"],
            app_data=d.get("app_data", {}),
        )


def resolve_snapshot_fields(options: TakeSnapshotOptions) -> set[SnapshotField]:
    """Resolve the set of fields to capture based on options.

    Applies: preset → include → exclude (in that order).

    Raises:
        SnapshotException: If any field name is invalid or the resolved set is empty.
    """
    valid = set(ALL_SNAPSHOT_FIELDS)

    # Validate include/exclude field names
    for field in options.get("include") or []:
        if field not in valid:
            raise SnapshotException(f"Invalid snapshot field: {field!r}. Valid fields: {sorted(valid)}")
    for field in options.get("exclude") or []:
        if field not in valid:
            raise SnapshotException(f"Invalid snapshot field: {field!r}. Valid fields: {sorted(valid)}")

    # Step 1: start with preset
    preset = options.get("preset")
    if preset is not None:
        fields: set[SnapshotField] = set(SNAPSHOT_PRESETS[preset])
    else:
        fields = set()

    # Step 2: union with include
    include = options.get("include")
    if include:
        fields |= set(include)

    # Step 3: subtract exclude
    exclude = options.get("exclude")
    if exclude:
        fields -= set(exclude)

    if not fields:
        raise SnapshotException(
            "No snapshot fields resolved. Provide a preset or at least one field in 'include'."
        )

    return fields


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string ending in 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
