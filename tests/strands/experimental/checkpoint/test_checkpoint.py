"""Tests for experimental checkpoint-based agent execution.

Design:
- Three checkpoint positions: after_model, after_tool, after_tools
- Per-tool granularity: each tool executes in its own step
- Model-driven: tool errors are tool results the model sees
- Returns Checkpoint (keep going) or AgentResult (done)
"""

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


# ── Serialization ──


class TestCheckpointSerialization:
    def test_round_trip(self):
        cp = Checkpoint(position="after_model", stop_reason="tool_use", cycle_index=1)
        restored = Checkpoint.from_dict(cp.to_dict())
        assert restored.cycle_index == 1
        assert restored.schema_version == CHECKPOINT_SCHEMA_VERSION

    def test_json_round_trip(self):
        cp = Checkpoint(
            position="after_tool",
            stop_reason="tool_use",
            tool_index=2,
            completed_tool_results=[{"toolUseId": "t1", "status": "success", "content": []}],
        )
        restored = Checkpoint.from_dict(json.loads(json.dumps(cp.to_dict())))
        assert restored.position == "after_tool"
        assert restored.tool_index == 2
        assert len(restored.completed_tool_results) == 1

    def test_ignores_unknown_fields(self):
        d = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "position": "after_model",
            "stop_reason": "tool_use",
            "extra": 1,
        }
        assert Checkpoint.from_dict(d).position == "after_model"

    def test_app_data_round_trip(self):
        cp = Checkpoint(position="after_model", stop_reason="tool_use", app_data={"wf_id": "123"})
        restored = Checkpoint.from_dict(json.loads(json.dumps(cp.to_dict())))
        assert restored.app_data["wf_id"] == "123"

    def test_schema_version_mismatch_raises(self):
        d = {"schema_version": "99.0", "position": "after_model", "stop_reason": "tool_use"}
        with pytest.raises(ValueError, match="Incompatible checkpoint schema version"):
            Checkpoint.from_dict(d)

    def test_schema_version_missing_raises(self):
        d = {"position": "after_model", "stop_reason": "tool_use"}
        with pytest.raises(ValueError, match="Incompatible checkpoint schema version"):
            Checkpoint.from_dict(d)


# ── No tools ──


class TestNoTools:
    @pytest.mark.asyncio
    async def test_completes_immediately(self):
        agent = Agent(model=MockedModelProvider([_end_turn("Hello")]), callback_handler=None)
        r = await invoke_with_checkpoint(agent, "Hi")
        assert isinstance(r, AgentResult)

    @pytest.mark.asyncio
    async def test_messages_appended(self):
        agent = Agent(model=MockedModelProvider([_end_turn("Hello")]), callback_handler=None)
        await invoke_with_checkpoint(agent, "Hi")
        assert len(agent.messages) == 2

    @pytest.mark.asyncio
    async def test_no_prompt_uses_existing_messages(self):
        agent = Agent(
            model=MockedModelProvider([_end_turn("Response")]),
            messages=[{"role": "user", "content": [{"text": "existing"}]}],
            callback_handler=None,
        )
        assert isinstance(await invoke_with_checkpoint(agent), AgentResult)


# ── Single tool: model → tool → model → done (3 steps) ──


class TestSingleTool:
    @pytest.mark.asyncio
    async def test_three_steps(self, echo_tool):
        agent = Agent(
            model=MockedModelProvider([_tool_use("echo"), _end_turn("Done")]),
            tools=[echo_tool],
            callback_handler=None,
        )

        r = await invoke_with_checkpoint(agent, "Go")
        assert isinstance(r, Checkpoint) and r.position == "after_model"

        # Single tool → goes directly to after_tools (no after_tool for last tool)
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, Checkpoint) and r.position == "after_tools"

        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, AgentResult)
        assert len(agent.messages) == 4


# ── Per-tool granularity: 5 tools = 5 separate steps ──


class TestPerToolGranularity:
    @pytest.mark.asyncio
    async def test_five_tools_five_steps(self, echo_tool):
        agent = Agent(
            model=MockedModelProvider(
                [
                    _tool_use("echo", "echo", "echo", "echo", "echo"),
                    _end_turn("All done"),
                ]
            ),
            tools=[echo_tool],
            callback_handler=None,
        )

        # Model call
        r = await invoke_with_checkpoint(agent, "Go")
        assert isinstance(r, Checkpoint) and r.position == "after_model"
        assert r.tool_index == 0

        # Tool 1 → after_tool (more tools remaining)
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, Checkpoint) and r.position == "after_tool"
        assert r.tool_index == 1
        assert len(r.completed_tool_results) == 1

        # Tool 2
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert r.position == "after_tool" and r.tool_index == 2

        # Tool 3
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert r.position == "after_tool" and r.tool_index == 3

        # Tool 4
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert r.position == "after_tool" and r.tool_index == 4
        assert len(r.completed_tool_results) == 4

        # Tool 5 (last) → after_tools
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, Checkpoint) and r.position == "after_tools"

        # 5 tool results in the message
        assert len(agent.messages[-1]["content"]) == 5

        # Final model call
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, AgentResult)


# ── Multi-cycle ──


class TestMultiCycle:
    @pytest.mark.asyncio
    async def test_two_cycles(self, echo_tool):
        agent = Agent(
            model=MockedModelProvider(
                [
                    _tool_use("echo"),
                    _tool_use("echo"),
                    _end_turn("Final"),
                ]
            ),
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


# ── Crash recovery ──


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_recover_after_model(self, echo_tool):
        agent1 = Agent(
            model=MockedModelProvider([_tool_use("echo"), _end_turn("Done")]),
            tools=[echo_tool],
            callback_handler=None,
        )
        r = await invoke_with_checkpoint(agent1, "Go")
        saved = json.dumps(r.to_dict())

        agent2 = Agent(
            model=MockedModelProvider([_end_turn("Recovered")]),
            tools=[echo_tool],
            callback_handler=None,
        )
        r = await invoke_with_checkpoint(agent2, checkpoint=Checkpoint.from_dict(json.loads(saved)))
        assert isinstance(r, Checkpoint) and r.position == "after_tools"
        r = await invoke_with_checkpoint(agent2, checkpoint=r)
        assert isinstance(r, AgentResult)

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
            model=MockedModelProvider(
                [
                    _tool_use("tracked", "tracked", "tracked"),
                    _end_turn("Done"),
                ]
            ),
            tools=[tracked],
            callback_handler=None,
        )

        r = await invoke_with_checkpoint(agent1, "Go")
        r = await invoke_with_checkpoint(agent1, checkpoint=r)  # tool 1
        r = await invoke_with_checkpoint(agent1, checkpoint=r)  # tool 2
        assert r.position == "after_tool" and r.tool_index == 2
        assert len(r.completed_tool_results) == 2

        # "Crash" — serialize, new agent
        saved = json.dumps(r.to_dict())
        log.clear()

        agent2 = Agent(
            model=MockedModelProvider([_end_turn("Recovered")]),
            tools=[tracked],
            callback_handler=None,
        )
        cp = Checkpoint.from_dict(json.loads(saved))

        # Only tool 3 runs
        r = await invoke_with_checkpoint(agent2, checkpoint=cp)
        assert isinstance(r, Checkpoint) and r.position == "after_tools"
        assert len(log) == 1  # only tool 3 executed

        r = await invoke_with_checkpoint(agent2, checkpoint=r)
        assert isinstance(r, AgentResult)


# ── Model-driven tool errors ──


class TestModelDrivenToolErrors:
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
        r = await invoke_with_checkpoint(agent, checkpoint=r)  # tool fails → after_tools
        assert isinstance(r, Checkpoint) and r.position == "after_tools"
        assert agent.messages[-1]["content"][0]["toolResult"]["status"] == "error"

        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, AgentResult)

    @pytest.mark.asyncio
    async def test_five_tools_third_fails_model_retries(self):
        from strands import tool

        log = []
        deploy_n = {"n": 0}

        @tool
        def fetch() -> str:
            """Fetch."""
            log.append("fetch")
            return "ok"

        @tool
        def validate() -> str:
            """Validate."""
            log.append("validate")
            return "ok"

        @tool
        def deploy() -> str:
            """Deploy."""
            log.append("deploy")
            deploy_n["n"] += 1
            if deploy_n["n"] == 1:
                raise RuntimeError("timeout")
            return "deployed"

        @tool
        def notify() -> str:
            """Notify."""
            log.append("notify")
            return "ok"

        @tool
        def cleanup() -> str:
            """Cleanup."""
            log.append("cleanup")
            return "ok"

        agent = Agent(
            model=MockedModelProvider(
                [
                    _tool_use("fetch", "validate", "deploy", "notify", "cleanup"),
                    _tool_use("deploy"),
                    _end_turn("Pipeline complete."),
                ]
            ),
            tools=[fetch, validate, deploy, notify, cleanup],
            callback_handler=None,
        )

        # Cycle 1: model → 5 tools (per-tool) → model sees deploy error
        r = await invoke_with_checkpoint(agent, "Run pipeline")
        for _ in range(5):
            r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert isinstance(r, Checkpoint) and r.position == "after_tools"
        assert log == ["fetch", "validate", "deploy", "notify", "cleanup"]

        r = await invoke_with_checkpoint(agent, checkpoint=r)  # model sees results
        assert isinstance(r, Checkpoint) and r.position == "after_model"

        # Cycle 2: model retries just deploy
        r = await invoke_with_checkpoint(agent, checkpoint=r)  # deploy succeeds
        assert isinstance(r, Checkpoint) and r.position == "after_tools"
        r = await invoke_with_checkpoint(agent, checkpoint=r)  # model done
        assert isinstance(r, AgentResult)
        assert deploy_n["n"] == 2


# ── Agent method ──


class TestAgentMethod:
    @pytest.mark.asyncio
    async def test_method_style(self, echo_tool):
        agent = Agent(
            model=MockedModelProvider([_tool_use("echo"), _end_turn("Done")]),
            tools=[echo_tool],
            callback_handler=None,
        )
        r = await agent.invoke_with_checkpoint("Go")
        while isinstance(r, Checkpoint):
            r = await agent.invoke_with_checkpoint(checkpoint=r)
        assert isinstance(r, AgentResult)


# ── Edge cases ──


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_tool_index_out_of_range_raises(self):
        """Corrupted checkpoint with tool_index >= number of tool blocks."""
        agent = Agent(
            model=MockedModelProvider([_tool_use("echo"), _end_turn("x")]),
            tools=[echo_tool],
            callback_handler=None,
        )
        # Get a valid after_model checkpoint
        r = await invoke_with_checkpoint(agent, "Go")
        assert isinstance(r, Checkpoint)
        # Tamper with tool_index
        r.tool_index = 99
        with pytest.raises(ValueError, match="tool_index=99 is out of range"):
            await invoke_with_checkpoint(agent, checkpoint=r)

    @pytest.mark.asyncio
    async def test_invalid_position_raises(self):
        cp = Checkpoint(position="invalid", stop_reason="end_turn")
        agent = Agent(model=MockedModelProvider([_end_turn("x")]), callback_handler=None)
        with pytest.raises(ValueError, match="Unknown checkpoint position"):
            await invoke_with_checkpoint(agent, checkpoint=cp)

    @pytest.mark.asyncio
    async def test_prompt_and_checkpoint_raises(self):
        """Cannot provide both prompt and checkpoint."""
        cp = Checkpoint(position="after_tools", stop_reason="tool_use")
        agent = Agent(model=MockedModelProvider([_end_turn("x")]), callback_handler=None)
        with pytest.raises(ValueError, match="Cannot provide both"):
            await invoke_with_checkpoint(agent, "Hello", checkpoint=cp)

    @pytest.mark.asyncio
    async def test_valid_position_wrong_stop_reason_raises(self):
        """after_model with stop_reason != 'tool_use' raises specific error."""
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

    @pytest.mark.asyncio
    async def test_tool_returning_no_result(self):
        """A tool whose stream yields nothing gets an error result."""
        from unittest.mock import AsyncMock, PropertyMock

        from strands.types.tools import AgentTool

        mock_tool = AsyncMock(spec=AgentTool)
        type(mock_tool).tool_name = PropertyMock(return_value="empty")
        type(mock_tool).tool_spec = PropertyMock(
            return_value={
                "name": "empty",
                "description": "Does nothing",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }
        )
        type(mock_tool).tool_type = PropertyMock(return_value="tool")

        async def empty_stream(tool_use, invocation_state):
            return
            yield  # noqa: unreachable — makes this an async generator

        mock_tool.stream = empty_stream

        agent = Agent(
            model=MockedModelProvider([_tool_use("empty"), _end_turn("Ok")]),
            tools=[mock_tool],
            callback_handler=None,
        )
        r = await invoke_with_checkpoint(agent, "Go")
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert agent.messages[-1]["content"][0]["toolResult"]["status"] == "error"
        assert "did not return a result" in agent.messages[-1]["content"][0]["toolResult"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tool_receives_agent_in_invocation_state(self):
        """Tools executed in checkpoint mode receive invocation_state['agent']."""
        from strands import tool

        captured = {}

        @tool
        def spy() -> str:
            """Spy tool."""
            return "ok"

        # Patch the tool's stream to capture invocation_state
        original_stream = spy.stream

        async def capturing_stream(tool_use, invocation_state):
            captured["invocation_state"] = invocation_state
            async for event in original_stream(tool_use, invocation_state):
                yield event

        spy.stream = capturing_stream

        agent = Agent(
            model=MockedModelProvider([_tool_use("spy"), _end_turn("Done")]),
            tools=[spy],
            callback_handler=None,
        )
        r = await invoke_with_checkpoint(agent, "Go")
        r = await invoke_with_checkpoint(agent, checkpoint=r)
        assert "agent" in captured["invocation_state"]
        assert captured["invocation_state"]["agent"] is agent
