"""Integration tests for session management."""

import tempfile
from uuid import uuid4

import boto3
import pytest
from botocore.client import ClientError

from strands import Agent
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.session.file_session_manager import FileSessionManager
from strands.session.s3_session_manager import S3SessionManager

# yellow_img imported from conftest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def bucket_name():
    bucket_name = f"test-strands-session-bucket-{boto3.client('sts').get_caller_identity()['Account']}"
    s3_client = boto3.resource("s3", region_name="us-west-2")
    try:
        s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    except ClientError as e:
        if "BucketAlreadyOwnedByYou" not in str(e):
            raise e
    yield bucket_name


def test_agent_with_file_session(temp_dir):
    # Set up the session manager and add an agent
    test_session_id = str(uuid4())
    # Create a session
    session_manager = FileSessionManager(session_id=test_session_id, storage_dir=temp_dir)
    try:
        agent = Agent(session_manager=session_manager)
        agent("Hello!")
        assert len(session_manager.list_messages(test_session_id, agent.agent_id)) == 2

        # After agent is persisted and run, restore the agent and run it again
        session_manager_2 = FileSessionManager(session_id=test_session_id, storage_dir=temp_dir)
        agent_2 = Agent(session_manager=session_manager_2)
        assert len(agent_2.messages) == 2
        agent_2("Hello!")
        assert len(agent_2.messages) == 4
        assert len(session_manager_2.list_messages(test_session_id, agent_2.agent_id)) == 4
    finally:
        # Delete the session
        session_manager.delete_session(test_session_id)
        assert session_manager.read_session(test_session_id) is None


def test_agent_with_file_session_and_conversation_manager(temp_dir):
    # Set up the session manager and add an agent
    test_session_id = str(uuid4())
    # Create a session
    session_manager = FileSessionManager(session_id=test_session_id, storage_dir=temp_dir)
    try:
        agent = Agent(
            session_manager=session_manager, conversation_manager=SlidingWindowConversationManager(window_size=1)
        )
        agent("Hello!")
        assert len(session_manager.list_messages(test_session_id, agent.agent_id)) == 2
        # Conversation Manager reduced messages
        assert len(agent.messages) == 1

        # After agent is persisted and run, restore the agent and run it again
        session_manager_2 = FileSessionManager(session_id=test_session_id, storage_dir=temp_dir)
        agent_2 = Agent(
            session_manager=session_manager_2, conversation_manager=SlidingWindowConversationManager(window_size=1)
        )
        assert len(agent_2.messages) == 1
        assert agent_2.conversation_manager.removed_message_count == 1
        agent_2("Hello!")
        assert len(agent_2.messages) == 1
        assert len(session_manager_2.list_messages(test_session_id, agent_2.agent_id)) == 4
    finally:
        # Delete the session
        session_manager.delete_session(test_session_id)
        assert session_manager.read_session(test_session_id) is None


def test_agent_with_file_session_with_image(temp_dir, yellow_img):
    test_session_id = str(uuid4())
    # Create a session
    session_manager = FileSessionManager(session_id=test_session_id, storage_dir=temp_dir)
    try:
        agent = Agent(session_manager=session_manager)
        agent([{"image": {"format": "png", "source": {"bytes": yellow_img}}}])
        assert len(session_manager.list_messages(test_session_id, agent.agent_id)) == 2

        # After agent is persisted and run, restore the agent and run it again
        session_manager_2 = FileSessionManager(session_id=test_session_id, storage_dir=temp_dir)
        agent_2 = Agent(session_manager=session_manager_2)
        assert len(agent_2.messages) == 2
        agent_2("Hello!")
        assert len(agent_2.messages) == 4
        assert len(session_manager_2.list_messages(test_session_id, agent_2.agent_id)) == 4
    finally:
        # Delete the session
        session_manager.delete_session(test_session_id)
        assert session_manager.read_session(test_session_id) is None


def test_agent_with_s3_session(bucket_name):
    test_session_id = str(uuid4())
    session_manager = S3SessionManager(session_id=test_session_id, bucket=bucket_name, region_name="us-west-2")
    try:
        agent = Agent(session_manager=session_manager)
        agent("Hello!")
        assert len(session_manager.list_messages(test_session_id, agent.agent_id)) == 2

        # After agent is persisted and run, restore the agent and run it again
        session_manager_2 = S3SessionManager(session_id=test_session_id, bucket=bucket_name, region_name="us-west-2")
        agent_2 = Agent(session_manager=session_manager_2)
        assert len(agent_2.messages) == 2
        agent_2("Hello!")
        assert len(agent_2.messages) == 4
        assert len(session_manager_2.list_messages(test_session_id, agent_2.agent_id)) == 4
    finally:
        session_manager.delete_session(test_session_id)
        assert session_manager.read_session(test_session_id) is None


def test_agent_with_s3_session_with_image(yellow_img, bucket_name):
    test_session_id = str(uuid4())
    session_manager = S3SessionManager(session_id=test_session_id, bucket=bucket_name, region_name="us-west-2")
    try:
        agent = Agent(session_manager=session_manager)
        agent([{"image": {"format": "png", "source": {"bytes": yellow_img}}}])
        assert len(session_manager.list_messages(test_session_id, agent.agent_id)) == 2

        # After agent is persisted and run, restore the agent and run it again
        session_manager_2 = S3SessionManager(session_id=test_session_id, bucket=bucket_name, region_name="us-west-2")
        agent_2 = Agent(session_manager=session_manager_2)
        assert len(agent_2.messages) == 2
        agent_2("Hello!")
        assert len(agent_2.messages) == 4
        assert len(session_manager_2.list_messages(test_session_id, agent_2.agent_id)) == 4
    finally:
        session_manager.delete_session(test_session_id)
        assert session_manager.read_session(test_session_id) is None
