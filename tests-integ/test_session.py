import pytest

from strands import Agent
from strands.session.file_session_manager import FileSessionManager


@pytest.fixture
def session_manager():
    session_manager = FileSessionManager(session_id="test")
    try:
        yield session_manager
    finally:
        session_manager.delete_session("test")


def test_agent_with_session(session_manager):
    agent = Agent(session_manager=session_manager)
    agent("Hello!")
    assert len(session_manager.list_messages("test", agent.id)) == 2
