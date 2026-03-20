"""Tests for _snapshot.py — Snapshot dataclass and resolve_snapshot_fields."""

import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands import Agent
from strands.types._snapshot import (
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
    defaults: dict[str, Any] = {
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


def test_snapshot_from_dict_bad_version_raises():
    d = {"schema_version": "99.0", "created_at": "2025-01-15T12:00:00Z", "data": {}, "app_data": {}}
    with pytest.raises(SnapshotException, match="Unsupported snapshot schema version"):
        Snapshot.from_dict(d)


def test_snapshot_to_dict_round_trip():
    s = _make_snapshot(data={"messages": []}, app_data={"x": 1})
    assert Snapshot.from_dict(s.to_dict()) == s


def test_resolve_snapshot_fields_invalid_include_raises():
    with pytest.raises(SnapshotException, match="Invalid snapshot field"):
        resolve_snapshot_fields({"include": ["not_a_field"]})  # type: ignore[typeddict-item]


def test_resolve_snapshot_fields_invalid_exclude_raises():
    with pytest.raises(SnapshotException, match="Invalid snapshot field"):
        resolve_snapshot_fields({"preset": "session", "exclude": ["not_a_field"]})  # type: ignore[typeddict-item]


def test_resolve_snapshot_fields_no_preset_no_include_raises():
    with pytest.raises(SnapshotException, match="No snapshot fields resolved"):
        resolve_snapshot_fields({})


def test_resolve_snapshot_fields_session_preset():
    assert resolve_snapshot_fields({"preset": "session"}) == set(SNAPSHOT_PRESETS["session"])


def test_resolve_snapshot_fields_include_adds_to_preset():
    fields = resolve_snapshot_fields({"preset": "session", "include": ["system_prompt"]})
    assert fields == set(SNAPSHOT_PRESETS["session"]) | {"system_prompt"}


def test_resolve_snapshot_fields_exclude_removes_from_preset():
    fields = resolve_snapshot_fields({"preset": "session", "exclude": ["messages"]})
    assert "messages" not in fields


def test_resolve_snapshot_fields_all_excluded_raises():
    with pytest.raises(SnapshotException):
        resolve_snapshot_fields({"exclude": list(ALL_SNAPSHOT_FIELDS)})  # type: ignore[typeddict-item]


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


_STRUCTURAL_CASES = [
    ([], {}, None),
    ([{"role": "user", "content": [{"text": "hi"}]}], {"k": "v"}, "system prompt"),
    ([{"role": "user", "content": [{"text": "a"}]}, {"role": "user", "content": [{"text": "b"}]}], {}, None),
    ([], {"num": 42, "flag": True}, "another prompt"),
]


@pytest.mark.parametrize("messages,state_dict,system_prompt", _STRUCTURAL_CASES)
def test_snapshot_structural_invariants(messages, state_dict, system_prompt):
    agent = _make_agent(messages=messages, state=state_dict, system_prompt=system_prompt)
    snapshot = agent.take_snapshot(preset="session")

    assert snapshot.schema_version == "1.0"
    assert ISO_8601_UTC_RE.match(snapshot.created_at), f"created_at={snapshot.created_at!r} not ISO 8601 UTC"
    assert isinstance(snapshot.data, dict)
    assert isinstance(snapshot.app_data, dict)
    for field in ("messages", "state", "conversation_manager_state", "interrupt_state"):
        assert field in snapshot.data
    assert "system_prompt" not in snapshot.data


_APP_DATA_CASES = [
    {"key": "value"},
    {"num": 42, "flag": True, "nothing": None},
    {"nested_str": "hello", "count": 0},
]


@pytest.mark.parametrize("app_data", _APP_DATA_CASES)
def test_app_data_stored_verbatim(app_data):
    agent = _make_agent()
    snapshot = agent.take_snapshot(preset="session", app_data=app_data)
    assert snapshot.app_data == app_data


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
    agent = _make_agent(messages=messages, state=state_dict, system_prompt="original prompt")
    snapshot = agent.take_snapshot(preset="session")

    fresh_agent = _make_agent(system_prompt="original prompt")
    fresh_agent.load_snapshot(snapshot)

    assert fresh_agent.messages == messages
    assert fresh_agent.state.get() == state_dict
    assert fresh_agent.system_prompt == "original prompt"
    assert fresh_agent.conversation_manager.get_state() == agent.conversation_manager.get_state()
    assert fresh_agent._interrupt_state.to_dict() == agent._interrupt_state.to_dict()


@pytest.mark.parametrize("omitted_field", list(ALL_SNAPSHOT_FIELDS))
def test_missing_fields_leave_agent_unchanged(omitted_field):
    agent = _make_agent(
        messages=[{"role": "user", "content": [{"text": "original"}]}],
        state={"key": "original"},
        system_prompt="original prompt",
    )

    include_fields = [f for f in ALL_SNAPSHOT_FIELDS if f != omitted_field]
    snapshot = agent.take_snapshot(include=include_fields)
    # system_prompt field is stored under the key "system_prompt" in snapshot.data
    data_key = "system_prompt" if omitted_field == "system_prompt" else omitted_field
    assert data_key not in snapshot.data

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
    else:
        pytest.fail(f"Unhandled field in test: {omitted_field!r}. Update this test when adding new snapshot fields.")

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
    else:
        pytest.fail(f"Unhandled field in test: {omitted_field!r}. Update this test when adding new snapshot fields.")


def test_snapshot_no_system_prompt_clears_target_agent_prompt():
    """Snapshot from agent with no system_prompt (field included) clears prompt on restore."""
    source_agent = _make_agent()  # no system_prompt
    snapshot = source_agent.take_snapshot(include=["system_prompt"])

    assert "system_prompt" in snapshot.data
    assert snapshot.data["system_prompt"] is None

    target_agent = _make_agent(system_prompt="existing prompt")
    target_agent.load_snapshot(snapshot)

    assert target_agent.system_prompt is None


def test_snapshot_without_system_prompt_field_preserves_target_agent_prompt():
    """Snapshot taken without system_prompt field does not override target agent's prompt."""
    source_agent = _make_agent(system_prompt="source prompt")
    snapshot = source_agent.take_snapshot(include=["messages"])  # system_prompt field excluded

    assert "system_prompt" not in snapshot.data

    target_agent = _make_agent(system_prompt="target prompt")
    target_agent.load_snapshot(snapshot)

    assert target_agent.system_prompt == "target prompt"


def test_load_snapshot_messages_are_independent_copy():
    """Messages restored from a snapshot are a copy — mutating snapshot.data after load doesn't affect the agent."""
    agent = _make_agent(messages=[{"role": "user", "content": [{"text": "hello"}]}])
    snapshot = agent.take_snapshot(preset="session")

    fresh_agent = _make_agent()
    fresh_agent.load_snapshot(snapshot)

    snapshot.data["messages"].append({"role": "user", "content": [{"text": "injected"}]})
    assert len(fresh_agent.messages) == 1


def test_take_snapshot_messages_are_independent_copy():
    """Mutating agent messages after take_snapshot doesn't corrupt the snapshot."""
    msg = {"role": "user", "content": [{"text": "original"}]}
    agent = _make_agent(messages=[msg])
    snapshot = agent.take_snapshot(preset="session")

    agent.messages[0]["content"][0]["text"] = "mutated"
    assert snapshot.data["messages"][0]["content"][0]["text"] == "original"


def test_take_snapshot_app_data_is_independent_copy():
    """Mutating app_data after take_snapshot doesn't corrupt the snapshot."""
    app_data = {"key": "original"}
    agent = _make_agent()
    snapshot = agent.take_snapshot(preset="session", app_data=app_data)

    app_data["key"] = "mutated"
    assert snapshot.app_data["key"] == "original"
