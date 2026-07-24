"""Tests for the ``stop`` tool.

The stop tool shims onto the event loop's existing termination flag:
``invocation_state["request_state"]["stop_event_loop"] = True``. Tests
exercise the flag-set behavior, input validation, and metadata rather than
running a full event loop end-to-end (that is covered by
:mod:`tests.strands.event_loop.test_event_loop`, which already asserts the flag
short-circuits the loop).
"""

from types import SimpleNamespace

import pytest

from strands.experimental.tools.stop import make_stop, stop
from strands.types.tools import ToolContext


def _tool_context(invocation_state: dict | None = None) -> ToolContext:
    """Build a ToolContext with a mutable ``invocation_state`` the tool can write to."""
    return ToolContext(
        tool_use={"name": "stop", "toolUseId": "id", "input": {}},
        agent=SimpleNamespace(),
        invocation_state=invocation_state if invocation_state is not None else {},
    )


class TestStopBehavior:
    """The tool sets the loop-termination flag and returns the message the model sees."""

    @pytest.mark.asyncio
    async def test_sets_stop_event_loop_flag(self):
        state: dict = {}
        ctx = _tool_context(state)
        await stop(tool_context=ctx)
        assert state["request_state"]["stop_event_loop"] is True

    @pytest.mark.asyncio
    async def test_returns_default_message_when_none_provided(self):
        result = await stop(tool_context=_tool_context())
        assert result == "Agent loop stopped."

    @pytest.mark.asyncio
    async def test_returns_provided_message_verbatim(self):
        result = await stop(tool_context=_tool_context(), message="all done")
        assert result == "all done"

    @pytest.mark.asyncio
    async def test_preserves_existing_request_state(self):
        state: dict = {"request_state": {"other_flag": "keep me"}}
        ctx = _tool_context(state)
        await stop(tool_context=ctx, message="bye")
        assert state["request_state"]["other_flag"] == "keep me"
        assert state["request_state"]["stop_event_loop"] is True

    @pytest.mark.asyncio
    async def test_creates_request_state_when_missing(self):
        state: dict = {}
        ctx = _tool_context(state)
        await stop(tool_context=ctx)
        assert "request_state" in state
        assert state["request_state"]["stop_event_loop"] is True

    @pytest.mark.asyncio
    async def test_empty_message_falls_back_to_default(self):
        # Kept symmetric with the TS side: an empty string falls back to the
        # default so the loop's final assistant turn is never blank.
        state: dict = {}
        ctx = _tool_context(state)
        result = await stop(tool_context=ctx, message="")
        assert result == "Agent loop stopped."
        assert state["request_state"]["stop_event_loop"] is True


class TestInputValidation:
    """The tool validates the ``message`` argument at the tool boundary."""

    @pytest.mark.asyncio
    async def test_rejects_oversized_message(self):
        oversized = "x" * 4097
        with pytest.raises(ValueError, match="exceeds the maximum"):
            await stop(tool_context=_tool_context(), message=oversized)

    @pytest.mark.asyncio
    async def test_accepts_message_at_length_cap(self):
        at_cap = "x" * 4096
        result = await stop(tool_context=_tool_context(), message=at_cap)
        assert result == at_cap

    @pytest.mark.asyncio
    async def test_rejects_non_string_message(self):
        with pytest.raises(ValueError, match="must be a string"):
            await stop(tool_context=_tool_context(), message=123)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_does_not_set_flag_when_validation_fails(self):
        state: dict = {}
        ctx = _tool_context(state)
        with pytest.raises(ValueError):
            await stop(tool_context=ctx, message="x" * 10000)
        assert "request_state" not in state

    @pytest.mark.asyncio
    async def test_configurable_max_message_length_relaxes_cap(self):
        big_stop = make_stop(max_message_length=10_000)
        message = "x" * 8000
        result = await big_stop(tool_context=_tool_context(), message=message)
        assert result == message

    @pytest.mark.asyncio
    async def test_configurable_max_message_length_still_enforces_new_cap(self):
        big_stop = make_stop(max_message_length=10_000)
        with pytest.raises(ValueError, match="exceeds the maximum of 10000"):
            await big_stop(tool_context=_tool_context(), message="x" * 10_001)

    def test_rejects_non_positive_max_message_length(self):
        with pytest.raises(ValueError, match="positive integer"):
            make_stop(max_message_length=0)
        with pytest.raises(ValueError, match="positive integer"):
            make_stop(max_message_length=-1)


class TestToolMetadata:
    """Tests for tool names, descriptions, and input schema."""

    def test_custom_name(self):
        assert make_stop(name="finish").tool_name == "finish"

    def test_custom_description(self):
        assert make_stop(description="custom desc").tool_spec["description"] == "custom desc"

    def test_schema_excludes_context(self):
        props = stop.tool_spec["inputSchema"]["json"]["properties"]
        assert "message" in props
        assert "tool_context" not in props

    def test_message_is_optional(self):
        required = stop.tool_spec["inputSchema"]["json"].get("required", [])
        assert "message" not in required
