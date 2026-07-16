"""Tests for the ``handoff_to_user`` tool.

The tool shims over :meth:`ToolContext.interrupt`. On first invocation the SDK
raises an ``InterruptException`` and the agent halts; on resume, the response
from the caller flows back through ``ToolContext.interrupt`` and is returned to
the model as a structured ``HandoffAnswer``.

These tests drive the tool through :meth:`AgentTool.stream` with a mock agent so
the interrupt/resume plumbing is exercised end-to-end without spinning up a real
model.
"""

import json
from unittest.mock import MagicMock

import pytest

from strands.interrupt import _InterruptState
from strands.types._events import ToolInterruptEvent, ToolResultEvent
from strands.vended_tools import handoff_to_user
from strands.vended_tools.handoff_to_user import (
    INTERRUPT_NAME,
    MAX_OPTION_LENGTH,
    MAX_OPTIONS_COUNT,
    MAX_QUESTION_LENGTH,
)


async def _alist(agen):
    """Drain an async iterator into a list (test helper)."""
    return [item async for item in agen]


def _tool_use(input_: dict, tool_use_id: str = "test_tool_id") -> dict:
    return {"name": "handoff_to_user", "toolUseId": tool_use_id, "input": input_}


def _invocation_state(interrupt_state: _InterruptState | None = None) -> dict:
    mock_agent = MagicMock()
    mock_agent._interrupt_state = interrupt_state or _InterruptState()
    return {"agent": mock_agent}


class TestToolMetadata:
    """Tests for the tool's user-facing surface."""

    def test_schema_excludes_context(self):
        props = handoff_to_user.tool_spec["inputSchema"]["json"]["properties"]
        assert "question" in props
        assert "options" in props
        assert "allow_free_text" in props
        assert "tool_context" not in props


class TestValidation:
    """Oversized / malformed inputs must be rejected at the tool boundary."""

    @pytest.mark.asyncio
    async def test_rejects_empty_question(self):
        events = await _alist(handoff_to_user.stream(_tool_use({"question": ""}), _invocation_state()))
        assert len(events) == 1
        assert isinstance(events[0], ToolResultEvent)
        assert events[0].tool_result["status"] == "error"
        assert "non-empty" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_rejects_oversized_question(self):
        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "x" * (MAX_QUESTION_LENGTH + 1)}), _invocation_state())
        )
        assert events[0].tool_result["status"] == "error"
        assert "maximum" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_rejects_too_many_options(self):
        events = await _alist(
            handoff_to_user.stream(
                _tool_use({"question": "pick one", "options": [f"o{i}" for i in range(MAX_OPTIONS_COUNT + 1)]}),
                _invocation_state(),
            )
        )
        assert events[0].tool_result["status"] == "error"
        assert "options count" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_rejects_oversized_option(self):
        events = await _alist(
            handoff_to_user.stream(
                _tool_use({"question": "pick one", "options": ["ok", "x" * (MAX_OPTION_LENGTH + 1)]}),
                _invocation_state(),
            )
        )
        assert events[0].tool_result["status"] == "error"
        assert "options[1] length" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_rejects_empty_option(self):
        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "pick one", "options": ["", "b"]}), _invocation_state())
        )
        assert events[0].tool_result["status"] == "error"
        assert "non-empty" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_rejects_duplicate_options(self):
        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "pick one", "options": ["a", "a"]}), _invocation_state())
        )
        assert events[0].tool_result["status"] == "error"
        assert "duplicates" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_rejects_duplicate_options_whitespace_variant(self):
        """['yes', 'yes '] should be caught as a duplicate — both trim to 'yes'."""
        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "pick one", "options": ["yes", "yes "]}), _invocation_state())
        )
        assert events[0].tool_result["status"] == "error"
        assert "duplicates" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_rejects_empty_options_list(self):
        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "pick one", "options": []}), _invocation_state())
        )
        assert events[0].tool_result["status"] == "error"
        assert "at least one" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_rejects_whitespace_only_option(self):
        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "pick one", "options": ["   ", "b"]}), _invocation_state())
        )
        assert events[0].tool_result["status"] == "error"
        assert "non-empty" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_rejects_no_answer_channel(self):
        """No options + free text disabled = question with nowhere to answer."""
        events = await _alist(
            handoff_to_user.stream(
                _tool_use({"question": "Q?", "allow_free_text": False}),
                _invocation_state(),
            )
        )
        assert events[0].tool_result["status"] == "error"
        assert "options or free text" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_rejects_non_string_option(self):
        """Locks in the ``options[i] must be a string`` guard, mirroring the TS
        Zod-schema test that rejects a boolean option entry. The tool decorator
        validates via pydantic before the callback runs, so a boolean surfaces
        as a "string_type" validation error rather than reaching the manual
        isinstance check inside ``_validate_and_build_reason``."""
        events = await _alist(
            handoff_to_user.stream(
                _tool_use({"question": "pick one", "options": [True, "b"]}),
                _invocation_state(),
            )
        )
        assert events[0].tool_result["status"] == "error"
        assert "valid string" in events[0].tool_result["content"][0]["text"]


class TestInterruptEmission:
    """The tool must emit a well-shaped interrupt via the SDK's interrupt path."""

    @pytest.mark.asyncio
    async def test_emits_interrupt_with_free_text_question(self):
        events = await _alist(handoff_to_user.stream(_tool_use({"question": "What's your name?"}), _invocation_state()))
        assert len(events) == 1
        assert isinstance(events[0], ToolInterruptEvent)
        interrupt = events[0].interrupts[0]
        assert interrupt.name == INTERRUPT_NAME
        assert interrupt.reason == {
            "question": "What's your name?",
            "options": None,
            "allow_free_text": True,
        }

    @pytest.mark.asyncio
    async def test_emits_interrupt_with_options(self):
        events = await _alist(
            handoff_to_user.stream(
                _tool_use(
                    {
                        "question": "Which environment?",
                        "options": ["dev", "staging", "prod"],
                        "allow_free_text": False,
                    }
                ),
                _invocation_state(),
            )
        )
        assert len(events) == 1
        assert isinstance(events[0], ToolInterruptEvent)
        interrupt = events[0].interrupts[0]
        assert interrupt.reason == {
            "question": "Which environment?",
            "options": ["dev", "staging", "prod"],
            "allow_free_text": False,
        }

    @pytest.mark.asyncio
    async def test_interrupt_registered_on_agent_state(self):
        """The interrupt must be visible on the agent's ``_interrupt_state`` so
        the SDK's stream/session code can serialize and return it."""
        state = _InterruptState()
        invocation_state = _invocation_state(state)

        await _alist(handoff_to_user.stream(_tool_use({"question": "Proceed?"}), invocation_state))

        assert len(state.interrupts) == 1
        interrupt = next(iter(state.interrupts.values()))
        assert interrupt.name == INTERRUPT_NAME
        assert interrupt.reason["question"] == "Proceed?"


class TestResume:
    """On resume, the caller's response is threaded back as the tool result."""

    async def _run_first_pass(self, input_: dict, state: _InterruptState) -> str:
        """Run the tool once to register an interrupt, and return its id."""
        events = await _alist(handoff_to_user.stream(_tool_use(input_), {"agent": self._mock_agent_from(state)}))
        assert isinstance(events[0], ToolInterruptEvent)
        return events[0].interrupts[0].id

    @staticmethod
    def _mock_agent_from(state: _InterruptState):
        m = MagicMock()
        m._interrupt_state = state
        return m

    @pytest.mark.asyncio
    async def test_resume_with_bare_string_wraps_as_answer(self):
        state = _InterruptState()
        interrupt_id = await self._run_first_pass({"question": "Name?"}, state)

        # Simulate a caller resume: set a response on the registered interrupt.
        state.interrupts[interrupt_id].response = "Alice"

        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "Name?"}), {"agent": self._mock_agent_from(state)})
        )
        assert isinstance(events[0], ToolResultEvent)
        assert events[0].tool_result["status"] == "success"
        # Tool return values are JSON-serialized into a text block by the decorator.
        # Deserialize and assert the full shape so an unexpected extra field would
        # be caught in one check (mirrors the TS ``toEqual`` assertions).
        text = events[0].tool_result["content"][0]["text"]
        assert json.loads(text) == {"answer": "Alice"}

    @pytest.mark.asyncio
    async def test_resume_with_dict_passes_chose_through(self):
        state = _InterruptState()
        interrupt_id = await self._run_first_pass({"question": "Which?", "options": ["a", "b"]}, state)

        state.interrupts[interrupt_id].response = {"answer": "b", "chose": "b"}

        events = await _alist(
            handoff_to_user.stream(
                _tool_use({"question": "Which?", "options": ["a", "b"]}),
                {"agent": self._mock_agent_from(state)},
            )
        )
        assert isinstance(events[0], ToolResultEvent)
        text = events[0].tool_result["content"][0]["text"]
        assert json.loads(text) == {"answer": "b", "chose": "b"}

    @pytest.mark.asyncio
    async def test_resume_with_malformed_response_errors(self):
        state = _InterruptState()
        interrupt_id = await self._run_first_pass({"question": "Q"}, state)

        # A number is neither a string nor a dict with 'answer' -> must reject.
        state.interrupts[interrupt_id].response = 42

        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "Q"}), {"agent": self._mock_agent_from(state)})
        )
        assert isinstance(events[0], ToolResultEvent)
        assert events[0].tool_result["status"] == "error"
        assert "answer" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_resume_rejects_oversized_bare_string(self):
        """Resume path enforces the same size cap as the outgoing question."""
        state = _InterruptState()
        interrupt_id = await self._run_first_pass({"question": "Q"}, state)

        state.interrupts[interrupt_id].response = "x" * (MAX_QUESTION_LENGTH + 1)

        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "Q"}), {"agent": self._mock_agent_from(state)})
        )
        assert events[0].tool_result["status"] == "error"
        assert "maximum" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_resume_rejects_non_string_answer(self):
        state = _InterruptState()
        interrupt_id = await self._run_first_pass({"question": "Q"}, state)

        state.interrupts[interrupt_id].response = {"answer": 42}

        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "Q"}), {"agent": self._mock_agent_from(state)})
        )
        assert events[0].tool_result["status"] == "error"
        assert "'answer' must be a string" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_resume_rejects_oversized_answer(self):
        state = _InterruptState()
        interrupt_id = await self._run_first_pass({"question": "Q"}, state)

        state.interrupts[interrupt_id].response = {"answer": "x" * (MAX_QUESTION_LENGTH + 1)}

        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "Q"}), {"agent": self._mock_agent_from(state)})
        )
        assert events[0].tool_result["status"] == "error"
        assert "maximum" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_resume_rejects_non_string_chose(self):
        state = _InterruptState()
        interrupt_id = await self._run_first_pass({"question": "Q"}, state)

        state.interrupts[interrupt_id].response = {"answer": "ok", "chose": 5}

        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "Q"}), {"agent": self._mock_agent_from(state)})
        )
        assert events[0].tool_result["status"] == "error"
        assert "'chose' must be a string" in events[0].tool_result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_resume_rejects_oversized_chose(self):
        state = _InterruptState()
        interrupt_id = await self._run_first_pass({"question": "Q"}, state)

        state.interrupts[interrupt_id].response = {"answer": "ok", "chose": "x" * (MAX_OPTION_LENGTH + 1)}

        events = await _alist(
            handoff_to_user.stream(_tool_use({"question": "Q"}), {"agent": self._mock_agent_from(state)})
        )
        assert events[0].tool_result["status"] == "error"
        assert "maximum" in events[0].tool_result["content"][0]["text"]
