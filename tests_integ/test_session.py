import boto3
import pytest

from strands import Agent
from strands.session.file_session_manager import FileSessionManager
from strands.session.s3_session_manager import S3SessionManager


@pytest.fixture
def file_session_manager():
    session_manager = FileSessionManager(session_id="test")
    try:
        yield session_manager
    finally:
        session_manager.delete_session("test")


@pytest.fixture
def bucket_name():
    s3 = boto3.resource("s3")
    bucket_name = f"strands-session-test-bucket-{boto3.client('sts').get_caller_identity()['Account']}"

    # Check if bucket exists, create if it doesn't
    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
    finally:
        s3.create_bucket(Bucket=bucket_name)

    yield bucket_name


@pytest.fixture
def s3_session_manager(bucket_name):
    session_manager = S3SessionManager(session_id="test", bucket=bucket_name)
    try:
        yield session_manager
    finally:
        session_manager.delete_session("test")


def test_agent_with_file_session(file_session_manager):
    agent = Agent(session_manager=file_session_manager)
    agent("Hello!")
    assert len(file_session_manager.list_messages("test", agent.agent_id)) == 2


def test_agent_with_s3_session(s3_session_manager):
    agent = Agent(session_manager=s3_session_manager)
    agent("Hello!")
    assert len(s3_session_manager.list_messages("test", agent.agent_id)) == 2
