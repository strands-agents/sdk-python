"""Tests for strands.experimental.checkpoint — Checkpoint serialization."""

import pytest

from strands import Agent, tool
from strands.experimental.checkpoint import CHECKPOINT_SCHEMA_VERSION, Checkpoint
from strands.types.exceptions import CheckpointException
from tests.fixtures.mocked_model_provider import MockedModelProvider


def test_checkpoint_to_dict_from_dict_round_trip():
    checkpoint = Checkpoint(
        position="after_model",
        cycle_index=1,
        snapshot={"messages": []},
        app_data={"workflow_id": "wf-123"},
    )
    data = checkpoint.to_dict()
    restored = Checkpoint.from_dict(data)

    assert restored.position == checkpoint.position
    assert restored.cycle_index == checkpoint.cycle_index
    assert restored.snapshot == checkpoint.snapshot
    assert restored.app_data == checkpoint.app_data
    assert restored.schema_version == CHECKPOINT_SCHEMA_VERSION


def test_checkpoint_init_schema_version_immutable():
    checkpoint = Checkpoint(position="after_tools")
    assert checkpoint.schema_version == CHECKPOINT_SCHEMA_VERSION


def test_checkpoint_init_defaults():
    checkpoint = Checkpoint(position="after_model")
    assert checkpoint.cycle_index == 0
    assert checkpoint.snapshot == {}
    assert checkpoint.app_data == {}


def test_checkpoint_from_dict_schema_version_mismatch_raises():
    data = Checkpoint(position="after_model").to_dict()
    data["schema_version"] = "0.0"
    with pytest.raises(CheckpointException, match="not compatible with current version"):
        Checkpoint.from_dict(data)


def test_checkpoint_from_dict_missing_schema_version_raises():
    data = {"position": "after_model", "cycle_index": 0, "snapshot": {}, "app_data": {}}
    with pytest.raises(CheckpointException, match="not compatible with current version"):
        Checkpoint.from_dict(data)


def test_checkpoint_from_dict_unknown_fields_warns(caplog):
    data = Checkpoint(position="after_tools").to_dict()
    data["unknown_future_field"] = "something"
    restored = Checkpoint.from_dict(data)
    assert restored.position == "after_tools"
    assert "unknown_future_field" in caplog.text


# =========================================================================
# End-to-end integration tests (Part B)
#
# These tests exercise the full pause/resume cycle through agent.invoke_async,
# using real Agent instances (not mocks) and a scripted model provider. They prove:
#
# 1. Checkpoints round-trip through to_dict/from_dict across fresh Agent instances.
# 2. cycle_index is preserved across process-restart-style resumes.
# 3. Completed tool work survives worker loss — tools do not re-execute on resume.
#
# They do NOT cover mid-tool crashes (orchestrator responsibility) or stateful
# model server-side state (documented V0 limitation).
# =========================================================================


def _assistant_tool_use(tool_use_id: str, name: str, input_data: dict) -> dict:
    """Build a scripted assistant message that invokes a single tool."""
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": input_data}}],
    }


def _assistant_text(text: str) -> dict:
    return {"role": "assistant", "content": [{"text": text}]}


@pytest.mark.asyncio
async def test_checkpoint_round_trip_across_cycles() -> None:
    """Fresh Agent pauses, serialize/deserialize, new Agent resumes, runs to completion."""
    call_log: list[str] = []

    @tool
    def noop(step: str) -> str:
        call_log.append(step)
        return f"ran-{step}"

    scripted_model = MockedModelProvider(
        [
            _assistant_tool_use("t1", "noop", {"step": "one"}),
            _assistant_text("done"),
        ]
    )

    agent_a = Agent(model=scripted_model, tools=[noop], checkpointing=True)

    # Cycle 0 — model requests tool, pause at after_model.
    result_after_model = await agent_a.invoke_async("please run a tool")
    assert result_after_model.stop_reason == "checkpoint"
    assert result_after_model.checkpoint is not None
    assert result_after_model.checkpoint.position == "after_model"
    assert result_after_model.checkpoint.cycle_index == 0
    assert call_log == []  # tool has not yet run

    # Serialize/deserialize — simulates crossing a process or activity boundary.
    checkpoint_wire = result_after_model.checkpoint.to_dict()
    resumed_checkpoint = Checkpoint.from_dict(checkpoint_wire)

    # Fresh Agent instance resumes and runs tools, pausing at after_tools.
    agent_b = Agent(model=scripted_model, tools=[noop], checkpointing=True)
    result_after_tools = await agent_b.invoke_async(
        [{"checkpointResume": {"checkpoint": resumed_checkpoint.to_dict()}}]
    )
    assert result_after_tools.stop_reason == "checkpoint"
    assert result_after_tools.checkpoint is not None
    assert result_after_tools.checkpoint.position == "after_tools"
    assert result_after_tools.checkpoint.cycle_index == 0
    assert call_log == ["one"]  # tool ran exactly once

    # Resume once more — model returns end_turn, agent completes.
    agent_c = Agent(model=scripted_model, tools=[noop], checkpointing=True)
    result_done = await agent_c.invoke_async(
        [{"checkpointResume": {"checkpoint": result_after_tools.checkpoint.to_dict()}}]
    )
    assert result_done.stop_reason == "end_turn"
    assert result_done.checkpoint is None
    # Tool still only ran once across the whole durable run.
    assert call_log == ["one"]


@pytest.mark.asyncio
async def test_crash_after_tools_does_not_rerun_completed_tools() -> None:
    """3 tools run, agent is discarded ('crash'), fresh agent resumes, tools do not re-run."""
    calls_alpha: list[str] = []
    calls_beta: list[str] = []
    calls_gamma: list[str] = []

    @tool
    def alpha(payload: str) -> str:
        calls_alpha.append(payload)
        return f"alpha-{payload}"

    @tool
    def beta(payload: str) -> str:
        calls_beta.append(payload)
        return f"beta-{payload}"

    @tool
    def gamma(payload: str) -> str:
        calls_gamma.append(payload)
        return f"gamma-{payload}"

    # One assistant message requests all three tools, then an end_turn response.
    scripted_model = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "t1", "name": "alpha", "input": {"payload": "a"}}},
                    {"toolUse": {"toolUseId": "t2", "name": "beta", "input": {"payload": "b"}}},
                    {"toolUse": {"toolUseId": "t3", "name": "gamma", "input": {"payload": "c"}}},
                ],
            },
            _assistant_text("all done"),
        ]
    )

    # Pre-crash agent: runs through after_model and after_tools.
    pre_crash = Agent(model=scripted_model, tools=[alpha, beta, gamma], checkpointing=True)
    after_model = await pre_crash.invoke_async("run the three tools")
    assert after_model.stop_reason == "checkpoint"
    assert after_model.checkpoint.position == "after_model"

    # Resume to run the tools.
    pre_crash_b = Agent(model=scripted_model, tools=[alpha, beta, gamma], checkpointing=True)
    after_tools = await pre_crash_b.invoke_async(
        [{"checkpointResume": {"checkpoint": after_model.checkpoint.to_dict()}}]
    )
    assert after_tools.stop_reason == "checkpoint"
    assert after_tools.checkpoint.position == "after_tools"
    # Exactly one call each, no double-runs.
    assert calls_alpha == ["a"]
    assert calls_beta == ["b"]
    assert calls_gamma == ["c"]

    # "Crash": discard pre_crash_b entirely. Persist only the serialized checkpoint.
    persisted = after_tools.checkpoint.to_dict()
    del pre_crash, pre_crash_b

    # Post-crash: brand-new agent resumes from the after_tools checkpoint.
    # The next model response is end_turn — no more tool use.
    post_crash = Agent(model=scripted_model, tools=[alpha, beta, gamma], checkpointing=True)
    final = await post_crash.invoke_async([{"checkpointResume": {"checkpoint": persisted}}])

    assert final.stop_reason == "end_turn"
    # No tool re-executed: call counts are unchanged.
    assert calls_alpha == ["a"]
    assert calls_beta == ["b"]
    assert calls_gamma == ["c"]
