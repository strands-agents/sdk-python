"""Integration tests for session management."""

import tempfile

import boto3
import pytest
from botocore.client import ClientError

from strands import Agent
from strands.session.file_session_manager import FileSessionManager
from strands.session.s3_session_manager import S3SessionManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def file_session_manager(temp_dir):
    """Create a file session manager for testing."""
    session_manager = FileSessionManager(session_id="test", storage_dir=temp_dir)
    yield session_manager
    session_manager.delete_session("test")


@pytest.fixture
def s3_session_manager():
    """Create an S3 session manager for testing."""
    # Create the bucket
    bucket_name = f"test-strands-session-bucket-{boto3.client('sts').get_caller_identity()['Account']}"
    s3_client = boto3.resource("s3", region_name="us-west-2")
    try:
        s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    except ClientError as e:
        if "BucketAlreadyOwnedByYou" not in str(e):
            raise e

    session_manager = S3SessionManager(session_id="test", bucket=bucket_name, region_name="us-west-2")
    yield session_manager

    session_manager.delete_session("test")


def test_agent_with_file_session(file_session_manager):
    agent = Agent(session_manager=file_session_manager)
    agent("Hello!")
    assert len(file_session_manager.list_messages("test", agent.agent_id)) == 2


def test_agent_with_s3_session(s3_session_manager):
    agent = Agent(session_manager=s3_session_manager)
    agent("Hello!")
    assert len(s3_session_manager.list_messages("test", agent.agent_id)) == 2
