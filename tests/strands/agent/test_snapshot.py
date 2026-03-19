"""Tests for _snapshot.py — Snapshot dataclass and resolve_snapshot_fields."""

import re
from unittest.mock import MagicMock

import pytest

from strands import Agent
from strands.agent._snapshot import (
    ALL_SNAPSHOT_FIELDS,
    SNAPSHOT_PRESETS,
    SNAPSHOT_SCHEMA_VERSION,
    Snapshot,
    TakeSnapshotOptions,
    resolve_snapshot_fields,
)
from strands.types.exceptions import SnapshotException

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ISO_8601_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def _make_snapshot(**kwargs: object) -> Snapshot:
    defaults: dict = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": "2025-01-15T12:00:00.000000Z",
        "data": {},
        "app_data": {},
    }
    defaults.update(kwargs)
    return Snapshot(**defaults)


def _make_agent(**kwargs) -> Agent:
    """Create a minimal Agent with a mock model for testing."""
    mock_model = MagicMock()
    mock_model.get_config.return_value = {}
    return Agent(model=mock_model, callback_handler=None, **kwargs)


# ---------------------------------------------------------------------------
# Unit tests — Snapshot dataclass
# ---------------------------------------------------------------------------


def test_snapshot_from_dict_bad_version_raises():
    d = {"schema_version": "99.0", "created_at": "2025-01-15T12:00:00Z", "data": {}, "app_data": {}}
    with pytest.raises(SnapshotException, match="Unsupported snapshot schema version"):
        Snapshot.from_dict(d)


def test_snapshot_to_dict_round_trip():
    s = _make_snapshot(data={"messages": []}, app_data={"x": 1})
    assert Snapshot.from_dict(s.to_dict()) == s


# ---------------------------------------------------------------------------
# Unit tests — resolve_snapshot_fields
# ---------------------------------------------------------------------------


def test_resolve_snapshot_fields_invalid_include_raises():
    with pytest.raises(SnapshotException, match="Invalid snapshot field"):
        resolve_snapshot_fields(TakeSnapshotOptions(include=["not_a_field"]))  # type: ignore[list-item]


def test_resolve_snapshot_fields_invalid_exclude_raises():
    with pytest.raises(SnapshotException, match="Invalid snapshot field"):
        resolve_snapshot_fields(TakeSnapshotOptions(preset="session", exclude=["not_a_field"]))  # type: ignore[list-item]


def test_resolve_snapshot_fields_no_preset_no_include_raises():
    with pytest.raises(SnapshotException, match="No snapshot fields resolved"):
        resolve_snapshot_fields(TakeSnapshotOptions())


def test_resolve_snapshot_fields_session_preset():
    assert resolve_snapshot_fields(TakeSnapshotOptions(preset="session")) == set(SNAPSHOT_PRESETS["session"])


def test_resolve_snapshot_fields_include_adds_to_preset():
    fields = resolve_snapshot_fields(TakeSnapshotOptions(preset="session", include=["system_prompt"]))
    assert fields == set(SNAPSHOT_PRESETS["session"]) | {"system_prompt"}


def test_resolve_snapshot_fields_exclude_removes_from_preset():
    fields = resolve_snapshot_fields(TakeSnapshotOptions(preset="session", exclude=["messages"]))
    assert "messages" not in fields


def test_resolve_snapshot_fields_all_excluded_raises():
    with pytest.raises(SnapshotException):
        resolve_snapshot_fields(TakeSnapshotOptions(exclude=list(ALL_SNAPSHOT_FIELDS)))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Property 2: Snapshot serialization round-trip
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------

_ROUND_TRIP_CASES = [
    ({}, {}),
    ({"messages": [{"role": "user", "content": [{"text": "hi"}]}]}, {}),
    ({"state": {"k": "v", "n": 1}}, {"app": "data"}),
    ({"messages": [], "state": {}, "system_prompt": "hello"}, {"meta": True}),
    ({"state": {"nested": None}}, {"x": None, "y": 42, "z": "str"}),
]


@pytest.mark.parametrize("data,app_data", _ROUND_TRIP_CASES)
def test_snapshot_serialization_round_trip(data, app_data):
    """Property 2: Snapshot serialization round-trip. Validates: Requirements 1.4"""
    s = _make_snapshot(data=data, app_data=app_data)
    assert Snapshot.from_dict(s.to_dict()) == s


# ---------------------------------------------------------------------------
# Property 6: resolve_snapshot_fields preset → include → exclude ordering
# Validates: Requirements 2.2, 2.3, 2.4
# ---------------------------------------------------------------------------

_ORDERING_CASES = [
    # (preset, include, exclude)
    ("session", [], []),
    ("session", ["system_prompt"], []),
    ("session", [], ["messages"]),
    ("session", ["system_prompt"], ["messages", "state"]),
    (None, ["messages", "state"], []),
    (None, list(ALL_SNAPSHOT_FIELDS), []),
    (None, list(ALL_SNAPSHOT_FIELDS), ["system_prompt"]),
    ("session", ["system_prompt"], list(SNAPSHOT_PRESETS["session"])),  # exclude all preset → only system_prompt
]


@pytest.mark.parametrize("preset,include,exclude", _ORDERING_CASES)
def test_resolve_snapshot_fields_ordering(preset, include, exclude):
    """Property 6: resolve_snapshot_fields preset → include → exclude ordering."""
    expected = (set(SNAPSHOT_PRESETS[preset] if preset else []) | set(include)) - set(exclude)
    options: TakeSnapshotOptions = {}
    if preset is not None:
        options["preset"] = preset  # type: ignore[assignment]
    if include:
        options["include"] = include  # type: ignore[assignment]
    if exclude:
        options["exclude"] = exclude  # type: ignore[assignment]

    if not expected:
        with pytest.raises(SnapshotException):
            resolve_snapshot_fields(options)
    else:
        assert resolve_snapshot_fields(options) == expected


# ---------------------------------------------------------------------------
# Property 7: Empty field set raises SnapshotException
# Validates: Requirements 2.5
# ---------------------------------------------------------------------------


def test_empty_field_set_no_preset_no_include_raises():
    """Property 7: Empty field set raises SnapshotException."""
    with pytest.raises(SnapshotException):
        resolve_snapshot_fields(TakeSnapshotOptions())


def test_empty_field_set_all_excluded_raises():
    """Property 7: Excluding all fields raises SnapshotException."""
    with pytest.raises(SnapshotException):
        resolve_snapshot_fields(TakeSnapshotOptions(exclude=list(ALL_SNAPSHOT_FIELDS)))  # type: ignore[arg-type]


def test_empty_field_set_preset_fully_excluded_raises():
    """Property 7: Excluding all preset fields with no include raises SnapshotException."""
    with pytest.raises(SnapshotException):
        resolve_snapshot_fields(
            TakeSnapshotOptions(preset="session", exclude=list(SNAPSHOT_PRESETS["session"]))  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Property 1: Snapshot structural invariants
# Validates: Requirements 1.1, 1.2, 1.3
# ---------------------------------------------------------------------------

_STRUCTURAL_CASES = [
    ([], {}, None),
    ([{"role": "user", "content": [{"text": "hi"}]}], {"k": "v"}, "system prompt"),
    ([{"role": "user", "content": [{"text": "a"}]}, {"role": "user", "content": [{"text": "b"}]}], {}, None),
    ([], {"num": 42, "flag": True}, "another prompt"),
]


@pytest.mark.parametrize("messages,state_dict,system_prompt", _STRUCTURAL_CASES)
def test_snapshot_structural_invariants(messages, state_dict, system_prompt):
    """Property 1: Snapshot structural invariants. Validates: Requirements 1.1, 1.2, 1.3"""
    agent = _make_agent(messages=messages, state=state_dict, system_prompt=system_prompt)
    snapshot = agent.take_snapshot(preset="session")

    assert snapshot.schema_version == "1.0"
    assert ISO_8601_UTC_RE.match(snapshot.created_at), f"created_at={snapshot.created_at!r} not ISO 8601 UTC"
    assert isinstance(snapshot.data, dict)
    assert isinstance(snapshot.app_data, dict)
    for field in ("messages", "state", "conversation_manager_state", "interrupt_state"):
        assert field in snapshot.data
    assert "system_prompt" not in snapshot.data


# ---------------------------------------------------------------------------
# Property 4: app_data stored verbatim
# Validates: Requirements 2.6
# ---------------------------------------------------------------------------

_APP_DATA_CASES = [
    {},
    {"key": "value"},
    {"num": 42, "flag": True, "nothing": None},
    {"nested_str": "hello", "count": 0},
]


@pytest.mark.parametrize("app_data", _APP_DATA_CASES)
def test_app_data_stored_verbatim(app_data):
    """Property 4: app_data stored verbatim. Validates: Requirements 2.6"""
    agent = _make_agent()
    snapshot = agent.take_snapshot(preset="session", app_data=app_data)
    assert snapshot.app_data == app_data


# ---------------------------------------------------------------------------
# Property 3: Agent state round-trip
# Validates: Requirements 2.1, 2.2, 2.8
# ---------------------------------------------------------------------------

_ROUND_TRIP_AGENT_CASES = [
    ([], {}),
    ([{"role": "user", "content": [{"text": "hi"}]}], {"k": "v"}),
    (
        [{"role": "user", "content": [{"text": "a"}]}, {"role": "user", "content": [{"text": "b"}]}],
        {"num": 1, "flag": None},
    ),
]


@pytest.mark.parametrize("messages,state_dict", _ROUND_TRIP_AGENT_CASES)
def test_agent_state_round_trip(messages, state_dict):
    """Property 3: Agent state round-trip. Validates: Requirements 2.1, 2.2, 2.8"""
    agent = _make_agent(messages=messages, state=state_dict, system_prompt="original prompt")
    snapshot = agent.take_snapshot(preset="session")

    fresh_agent = _make_agent(system_prompt="original prompt")
    fresh_agent.load_snapshot(snapshot)

    assert fresh_agent.messages == messages
    assert fresh_agent.state.get() == state_dict
    assert fresh_agent.system_prompt == "original prompt"


# ---------------------------------------------------------------------------
# Property 5: Missing data fields leave agent state unchanged
# Validates: Requirements 2.9
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omitted_field", list(ALL_SNAPSHOT_FIELDS))
def test_missing_fields_leave_agent_unchanged(omitted_field):
    """Property 5: Missing data fields leave agent state unchanged. Validates: Requirements 2.9"""
    agent = _make_agent(
        messages=[{"role": "user", "content": [{"text": "original"}]}],
        state={"key": "original"},
        system_prompt="original prompt",
    )

    include_fields = [f for f in ALL_SNAPSHOT_FIELDS if f != omitted_field]
    snapshot = agent.take_snapshot(include=include_fields)
    assert omitted_field not in snapshot.data

    fresh_agent = _make_agent(
        messages=list(agent.messages),
        state=agent.state.get(),
        system_prompt="original prompt",
    )

    if omitted_field == "messages":
        before = list(fresh_agent.messages)
    elif omitted_field == "state":
        before = fresh_agent.state.get()
    elif omitted_field == "system_prompt":
        before = fresh_agent.system_prompt
    elif omitted_field == "conversation_manager_state":
        before = fresh_agent.conversation_manager.get_state()
    elif omitted_field == "interrupt_state":
        before = fresh_agent._interrupt_state.to_dict()

    fresh_agent.load_snapshot(snapshot)

    if omitted_field == "messages":
        assert fresh_agent.messages == before
    elif omitted_field == "state":
        assert fresh_agent.state.get() == before
    elif omitted_field == "system_prompt":
        assert fresh_agent.system_prompt == before
    elif omitted_field == "conversation_manager_state":
        assert fresh_agent.conversation_manager.get_state() == before
    elif omitted_field == "interrupt_state":
        assert fresh_agent._interrupt_state.to_dict() == before


# ---------------------------------------------------------------------------
# Import path tests
# Validates: Requirements 3.1, 3.2, 3.3
# ---------------------------------------------------------------------------


def test_import_snapshot_from_strands():
    from strands import Snapshot as S
    assert S is not None


def test_import_snapshot_from_strands_types():
    from strands.types import Snapshot as S
    assert S is not None
