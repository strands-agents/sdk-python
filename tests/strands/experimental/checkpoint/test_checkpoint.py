"""Tests for experimental checkpoint-based agent execution."""

import json

import pytest

from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.experimental.checkpoint import Checkpoint, invoke_with_checkpoint
from strands.experimental.checkpoint.types import CHECKPOINT_SCHEMA_VERSION
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _end_turn(text: str) -> dict:
    return {"role": "assistant", "content": [{"text": text}]}


def _tool_use(*tool_names: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"toolUse": {"toolUseId": f"tu_{name}_{i}", "name": name, "input": {}}} for i, name in enumerate(tool_names)
        ],
    }


@pytest.fixture
def echo_tool():
    from strands import tool

    @tool
    def echo(message: str = "ok") -> str:
        """Echo a message back."""
        return f"echo: {message}"

    return echo


class TestCheckpointSerialization:
    def test_round_trip(self):
        cp = Checkpoint(position="after_model", stop_reason="tool_use", cycle_index=1, app_data={"wf_id": "123"})
        restored = Checkpoint.from_dict(json.loads(json.dumps(cp.to_dict())))
        assert restored.cycle_index == 1
        assert restored.schema_version == CHECKPOINT_SCHEMA_VERSION
        assert restored.app_data["wf_id"] == "123"

    def test_schema_version_mismatch_raises(self):
        d = {"schema_version": "99.0", "position": "after_model", "stop_reason": "tool_use"}
        with pytest.raises(ValueError, match="Incompatible checkpoint schema version"):
            Checkpoint.from_dict(d)


class TestBasicFlow:
    @pytest.mark.asyncio
    async def test_no_tools_completes_immediately(self):
        agent = Agent(model=MockedModelProvider([_end_turn("Hello")]), callback_handler=None)
        r = await invoke_with_checkpoint(agent, "Hi")
        assert isinstance(r, AgentResult)
        assert len(agent.messages) == 2

    @pytest.mark.asyncio
    async def test_single_tool_three_steps(self, echo_tool):
        agent = Agent(
            model=MockedModelProvider([_tool_use("echo"), _end_turn("Done")]),
            tools=[echo_tool],
            callback_handler=None,
        )

        r = await invoke_with_checkpoint(agent, "Go")
        assert isinstance(r, Checkpoint) and r.position == "after_model"

        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, Checkpoint) and r.position == "after_tools"

        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, AgentResult)

    @pytest.mark.asyncio
    async def test_per_tool_granularity(self, echo_tool):
        """3 tools = 3 separate tool steps."""
        agent = Agent(
            model=MockedModelProvider([_tool_use("echo", "echo", "echo"), _end_turn("Done")]),
            tools=[echo_tool],
            callback_handler=None,
        )

        r = await invoke_with_checkpoint(agent, "Go")
        assert r.position == "after_model" and r.tool_index == 0

        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert r.position == "after_tool" and r.tool_index == 1

        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert r.position == "after_tool" and r.tool_index == 2

        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert r.position == "after_tools"
        assert len(agent.messages[-1]["content"]) == 3

        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, AgentResult)

    @pytest.mark.asyncio
    async def test_two_cycles(self, echo_tool):
        agent = Agent(
            model=MockedModelProvider([_tool_use("echo"), _tool_use("echo"), _end_turn("Final")]),
            tools=[echo_tool],
            callback_handler=None,
        )

        r = await invoke_with_checkpoint(agent, "Go")
        assert r.cycle_index == 0
        r = await invoke_with_checkpoint(agent, checkpoint=r)  # tool
        r = await invoke_with_checkpoint(agent, checkpoint=r)  # model

        assert r.cycle_index == 1
        r = await invoke_with_checkpoint(agent, checkpoint=r)  # tool
        r = await invoke_with_checkpoint(agent, checkpoint=r)  # model → done

        assert isinstance(r, AgentResult)

    @pytest.mark.asyncio
    async def test_agent_method(self, echo_tool):
        agent = Agent(
            model=MockedModelProvider([_tool_use("echo"), _end_turn("Done")]),
            tools=[echo_tool],
            callback_handler=None,
        )
        r = await agent.invoke_with_checkpoint("Go")
        while isinstance(r, Checkpoint):
            r = await agent.invoke_with_checkpoint(checkpoint=r)
        assert isinstance(r, AgentResult)


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_crash_between_tools_only_reruns_remaining(self):
        """3 tools. Crash after tool 2. Only tool 3 re-runs."""
        from strands import tool

        log = []

        @tool
        def tracked(label: str = "x") -> str:
            """Tracked tool."""
            log.append(label)
            return f"done: {label}"

        agent1 = Agent(
            model=MockedModelProvider([_tool_use("tracked", "tracked", "tracked"), _end_turn("Done")]),
            tools=[tracked],
            callback_handler=None,
        )

        r = await invoke_with_checkpoint(agent1, "Go")
        r = await invoke_with_checkpoint(agent1, checkpoint=r)  # tool 1
        r = await invoke_with_checkpoint(agent1, checkpoint=r)  # tool 2
        assert r.tool_index == 2 and len(r.completed_tool_results) == 2

        # "Crash" — serialize, new agent
        saved = json.dumps(r.to_dict())
        log.clear()

        agent2 = Agent(
            model=MockedModelProvider([_end_turn("Recovered")]),
            tools=[tracked],
            callback_handler=None,
        )

        r = await invoke_with_checkpoint(agent2, checkpoint=Checkpoint.from_dict(json.loads(saved)))
        assert r.position == "after_tools"
        assert len(log) == 1  # only tool 3 executed

        r = await invoke_with_checkpoint(agent2, checkpoint=r)
        assert isinstance(r, AgentResult)


class TestToolErrors:
    @pytest.mark.asyncio
    async def test_tool_error_sent_to_model(self):
        from strands import tool

        @tool
        def failing() -> str:
            """Always fails."""
            raise RuntimeError("Connection timeout")

        agent = Agent(
            model=MockedModelProvider([_tool_use("failing"), _end_turn("Tool failed.")]),
            tools=[failing],
            callback_handler=None,
        )

        r = await invoke_with_checkpoint(agent, "Go")
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert r.position == "after_tools"
        assert agent.messages[-1]["content"][0]["toolResult"]["status"] == "error"

        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, AgentResult)

    @pytest.mark.asyncio
    async def test_unknown_tool_error_sent_to_model(self):
        agent = Agent(
            model=MockedModelProvider([_tool_use("nonexistent"), _end_turn("Not found.")]),
            callback_handler=None,
        )
        r = await invoke_with_checkpoint(agent, "Go")
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert agent.messages[-1]["content"][0]["toolResult"]["status"] == "error"
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, AgentResult)


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_tool_index_out_of_range_raises(self, echo_tool):
        """Corrupted checkpoint with tool_index >= number of tool blocks."""
        agent = Agent(
            model=MockedModelProvider([_tool_use("echo"), _end_turn("x")]),
            tools=[echo_tool],
            callback_handler=None,
        )
        r = await invoke_with_checkpoint(agent, "Go")
        assert isinstance(r, Checkpoint)
        r.tool_index = 99
        with pytest.raises(ValueError, match="tool_index=99 is out of range"):
            await invoke_with_checkpoint(agent, checkpoint=r)

    @pytest.mark.asyncio
    async def test_prompt_and_checkpoint_raises(self):
        cp = Checkpoint(position="after_tools", stop_reason="tool_use")
        agent = Agent(model=MockedModelProvider([_end_turn("x")]), callback_handler=None)
        with pytest.raises(ValueError, match="Cannot provide both"):
            await invoke_with_checkpoint(agent, "Hello", checkpoint=cp)

    @pytest.mark.asyncio
    async def test_invalid_position_raises(self):
        cp = Checkpoint(position="invalid", stop_reason="end_turn")
        agent = Agent(model=MockedModelProvider([_end_turn("x")]), callback_handler=None)
        with pytest.raises(ValueError, match="Unknown checkpoint position"):
            await invoke_with_checkpoint(agent, checkpoint=cp)

    @pytest.mark.asyncio
    async def test_valid_position_wrong_stop_reason_raises(self):
        cp = Checkpoint(
            position="after_model",
            stop_reason="end_turn",
            snapshot={
                "scope": "agent",
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "created_at": "2025-01-01T00:00:00Z",
                "data": {"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
                "app_data": {},
            },
        )
        agent = Agent(model=MockedModelProvider([_end_turn("x")]), callback_handler=None)
        with pytest.raises(ValueError, match="requires stop_reason='tool_use'"):
            await invoke_with_checkpoint(agent, checkpoint=cp)
