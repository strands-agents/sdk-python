"""Tests for Agent.session_id resolution and validation."""

import re
from unittest.mock import MagicMock

import pytest

from strands import Agent


@pytest.fixture
def mock_model():
    async def stream(*args, **kwargs):
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {"text": ""}}}

    mock = MagicMock()
    mock.stream.side_effect = stream
    mock.stateful = False
    return mock


class TestAgentSessionId:
    def test_uses_explicit_session_id(self, mock_model):
        agent = Agent(model=mock_model, session_id="my-session")
        assert agent.session_id == "my-session"

    def test_inherits_session_id_from_session_manager(self, mock_model):
        sm = MagicMock()
        sm.session_id = "sm-session"
        agent = Agent(model=mock_model, session_manager=sm)
        assert agent.session_id == "sm-session"

    def test_auto_generates_uuid_when_no_session_id_or_session_manager(self, mock_model):
        agent = Agent(model=mock_model)
        uuid4_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
        assert uuid4_pattern.match(agent.session_id)

    def test_generates_unique_session_ids(self, mock_model):
        agent1 = Agent(model=mock_model)
        agent2 = Agent(model=mock_model)
        assert agent1.session_id != agent2.session_id

    def test_throws_on_session_id_conflict_with_session_manager(self, mock_model):
        sm = MagicMock()
        sm.session_id = "sm-session"
        with pytest.raises(ValueError, match="explicit session_id conflicts with session_manager.session_id"):
            Agent(model=mock_model, session_id="different-session", session_manager=sm)

    def test_accepts_matching_session_id_with_session_manager(self, mock_model):
        sm = MagicMock()
        sm.session_id = "same-session"
        agent = Agent(model=mock_model, session_id="same-session", session_manager=sm)
        assert agent.session_id == "same-session"

    def test_validates_session_id_rejects_path_separators(self, mock_model):
        with pytest.raises(ValueError, match="cannot contain path separators"):
            Agent(model=mock_model, session_id="path/separator")

    def test_inherits_none_session_id_from_session_manager_generates_uuid(self, mock_model):
        sm = MagicMock()
        sm.session_id = None
        agent = Agent(model=mock_model, session_manager=sm)
        uuid4_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
        assert uuid4_pattern.match(agent.session_id)
