"""Tests for deployment state management."""

import json
import os

import pytest

from strands.experimental.deploy._state import DeployState, StateManager


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def state_manager(tmp_dir):
    return StateManager(base_dir=tmp_dir)


class TestStateManager:
    def test_load_returns_none_when_no_state(self, state_manager):
        result = state_manager.load("nonexistent")
        assert result is None

    def test_save_and_load_roundtrip(self, state_manager):
        state = DeployState(
            target="agentcore",
            region="us-west-2",
            agent_runtime_id="abc123",
            agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/abc",
        )
        state_manager.save("my-agent", state)
        loaded = state_manager.load("my-agent")

        assert loaded is not None
        assert loaded["target"] == "agentcore"
        assert loaded["region"] == "us-west-2"
        assert loaded["agent_runtime_id"] == "abc123"
        assert "last_deployed" in loaded
        assert "created_at" in loaded

    def test_save_creates_strands_directory(self, tmp_dir, state_manager):
        state = DeployState(target="agentcore", region="us-east-1")
        state_manager.save("test", state)

        assert os.path.isdir(os.path.join(tmp_dir, ".strands"))
        assert os.path.isfile(os.path.join(tmp_dir, ".strands", "state.json"))

    def test_save_preserves_created_at_on_update(self, state_manager):
        state1 = DeployState(target="agentcore", region="us-west-2")
        state_manager.save("my-agent", state1)
        first_created_at = state_manager.load("my-agent")["created_at"]

        state2 = DeployState(target="agentcore", region="us-west-2", agent_runtime_id="new-id")
        state_manager.save("my-agent", state2)
        second = state_manager.load("my-agent")

        assert second["created_at"] == first_created_at
        assert second["agent_runtime_id"] == "new-id"

    def test_save_multiple_deployments(self, state_manager):
        state_manager.save("agent-a", DeployState(target="agentcore", region="us-east-1"))
        state_manager.save("agent-b", DeployState(target="agentcore", region="eu-west-1"))

        assert state_manager.load("agent-a")["region"] == "us-east-1"
        assert state_manager.load("agent-b")["region"] == "eu-west-1"

    def test_delete_removes_deployment(self, state_manager):
        state_manager.save("my-agent", DeployState(target="agentcore", region="us-west-2"))
        assert state_manager.load("my-agent") is not None

        state_manager.delete("my-agent")
        assert state_manager.load("my-agent") is None

    def test_delete_nonexistent_is_noop(self, state_manager):
        state_manager.delete("nonexistent")  # Should not raise

    def test_state_file_has_version(self, state_manager, tmp_dir):
        state_manager.save("test", DeployState(target="agentcore", region="us-east-1"))
        with open(os.path.join(tmp_dir, ".strands", "state.json")) as f:
            data = json.load(f)
        assert data["version"] == "1"

    def test_corrupted_state_file_raises(self, state_manager, tmp_dir):
        state_dir = os.path.join(tmp_dir, ".strands")
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, "state.json"), "w") as f:
            f.write("not valid json{{{")

        from strands.experimental.deploy._exceptions import DeployStateException

        with pytest.raises(DeployStateException):
            state_manager.load("anything")
