"""Tests for DynamoDBSessionManager."""

import time
from decimal import Decimal

import boto3
import pytest
from botocore.config import Config as BotocoreConfig
from moto import mock_aws

from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.session.dynamodb_session_manager import DynamoDBSessionManager, _convert_decimals_to_native_types
from strands.types.content import ContentBlock
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage, SessionType


@pytest.fixture
def mocked_aws():
    """
    Mock all AWS interactions
    Requires you to create your own boto3 clients
    """
    with mock_aws():
        yield


@pytest.fixture(scope="function")
def dynamodb_table(mocked_aws):
    """DynamoDB table for testing."""
    dynamodb = boto3.resource("dynamodb", region_name="us-west-2")
    client = boto3.client("dynamodb", region_name="us-west-2")
    table_name = "test-session-table"

    dynamodb.create_table(
        TableName=table_name,
        KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"}, {"AttributeName": "SK", "KeyType": "RANGE"}],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    time.sleep(1)  # make sure table is ready
    response = client.describe_table(TableName=table_name)
    assert response["Table"]["TableStatus"] == "ACTIVE"

    return table_name


@pytest.fixture
def dynamodb_manager(mocked_aws, dynamodb_table):
    """Create DynamoDBSessionManager with mocked DynamoDB."""
    yield DynamoDBSessionManager(session_id="test", table_name=dynamodb_table, region_name="us-west-2")


@pytest.fixture
def sample_session():
    """Create sample session for testing."""
    return Session(
        session_id="test-session-123",
        session_type=SessionType.AGENT,
    )


@pytest.fixture
def sample_agent():
    """Create sample agent for testing."""
    return SessionAgent(
        agent_id="test-agent-456",
        state={"key": "value"},
        conversation_manager_state=NullConversationManager().get_state(),
    )


@pytest.fixture
def sample_message():
    """Create sample message for testing."""
    return SessionMessage.from_message(
        message={
            "role": "user",
            "content": [ContentBlock(text="test_message")],
        },
        index=0,
    )


def test_init_dynamodb_session_manager(mocked_aws, dynamodb_table):
    session_manager = DynamoDBSessionManager(session_id="test", table_name=dynamodb_table, region_name="us-west-2")
    assert "strands-agents" in session_manager.client.meta.config.user_agent_extra


def test_init_dynamodb_session_manager_with_config(mocked_aws, dynamodb_table):
    session_manager = DynamoDBSessionManager(
        session_id="test", table_name=dynamodb_table, boto_client_config=BotocoreConfig(), region_name="us-west-2"
    )
    assert "strands-agents" in session_manager.client.meta.config.user_agent_extra


def test_init_dynamodb_session_manager_with_existing_user_agent(mocked_aws, dynamodb_table):
    session_manager = DynamoDBSessionManager(
        session_id="test",
        table_name=dynamodb_table,
        boto_client_config=BotocoreConfig(user_agent_extra="test"),
        region_name="us-west-2",
    )
    assert "strands-agents" in session_manager.client.meta.config.user_agent_extra


def test_create_session(dynamodb_manager, sample_session):
    """Test creating a session in DynamoDB."""
    result = dynamodb_manager.create_session(sample_session)
    assert result == sample_session

    # Verify DynamoDB item created
    response = dynamodb_manager.client.get_item(
        TableName=dynamodb_manager.table_name,
        Key={"PK": {"S": f"session_{sample_session.session_id}"}, "SK": {"S": "session"}},
    )
    assert "Item" in response
    assert response["Item"]["entity_type"]["S"] == "SESSION"

    data = dynamodb_manager.deserializer.deserialize(response["Item"]["data"])
    assert data["session_id"] == sample_session.session_id
    assert data["session_type"] == sample_session.session_type


def test_create_session_already_exists(dynamodb_manager, sample_session):
    """Test creating a session that already exists."""
    dynamodb_manager.create_session(sample_session)

    with pytest.raises(SessionException):
        dynamodb_manager.create_session(sample_session)


def test_read_session(dynamodb_manager, sample_session):
    """Test reading a session from DynamoDB."""
    # Create session first
    dynamodb_manager.create_session(sample_session)

    # Read it back
    result = dynamodb_manager.read_session(sample_session.session_id)

    assert result.session_id == sample_session.session_id
    assert result.session_type == sample_session.session_type


def test_read_nonexistent_session(dynamodb_manager):
    """Test reading a session that doesn't exist."""
    result = dynamodb_manager.read_session("nonexistent-session")
    assert result is None


def test_delete_session(dynamodb_manager, sample_session):
    """Test deleting a session from DynamoDB."""
    # Create session first
    dynamodb_manager.create_session(sample_session)

    # Verify session exists
    response = dynamodb_manager.client.get_item(
        TableName=dynamodb_manager.table_name,
        Key={"PK": {"S": f"session_{sample_session.session_id}"}, "SK": {"S": "session"}},
    )
    assert "Item" in response

    # Delete session
    dynamodb_manager.delete_session(sample_session.session_id)

    # Verify deletion
    response = dynamodb_manager.client.get_item(
        TableName=dynamodb_manager.table_name,
        Key={"PK": {"S": f"session_{sample_session.session_id}"}, "SK": {"S": "session"}},
    )
    assert "Item" not in response


def test_create_agent(dynamodb_manager, sample_session, sample_agent):
    """Test creating an agent in DynamoDB."""
    dynamodb_manager.create_session(sample_session)
    dynamodb_manager.create_agent(sample_session.session_id, sample_agent)

    # Verify DynamoDB item created
    response = dynamodb_manager.client.get_item(
        TableName=dynamodb_manager.table_name,
        Key={"PK": {"S": f"session_{sample_session.session_id}"}, "SK": {"S": f"agent_{sample_agent.agent_id}"}},
    )
    assert "Item" in response
    assert response["Item"]["entity_type"]["S"] == "AGENT"

    data = dynamodb_manager.deserializer.deserialize(response["Item"]["data"])
    assert data["agent_id"] == sample_agent.agent_id
    assert data["state"] == sample_agent.state


def test_read_agent(dynamodb_manager, sample_session, sample_agent):
    """Test reading an agent from DynamoDB."""
    # Create session and agent
    dynamodb_manager.create_session(sample_session)
    dynamodb_manager.create_agent(sample_session.session_id, sample_agent)

    # Read agent
    result = dynamodb_manager.read_agent(sample_session.session_id, sample_agent.agent_id)

    assert result.agent_id == sample_agent.agent_id
    assert result.state == sample_agent.state
    assert isinstance(result.conversation_manager_state.get("removed_message_count"), int)


def test_read_nonexistent_agent(dynamodb_manager, sample_session):
    """Test reading an agent that doesn't exist."""
    # Create session
    dynamodb_manager.create_session(sample_session)
    # Read agent
    result = dynamodb_manager.read_agent(sample_session.session_id, "nonexistent-agent")

    assert result is None


def test_update_agent(dynamodb_manager, sample_session, sample_agent):
    """Test updating an agent in DynamoDB."""
    dynamodb_manager.create_session(sample_session)
    dynamodb_manager.create_agent(sample_session.session_id, sample_agent)

    sample_agent.state = {"updated": "value"}
    dynamodb_manager.update_agent(sample_session.session_id, sample_agent)

    result = dynamodb_manager.read_agent(sample_session.session_id, sample_agent.agent_id)
    assert result.state == {"updated": "value"}


def test_update_nonexistent_agent(dynamodb_manager, sample_session, sample_agent):
    """Test updating an agent that doesn't exist."""
    dynamodb_manager.create_session(sample_session)

    with pytest.raises(SessionException):
        dynamodb_manager.update_agent(sample_session.session_id, sample_agent)


def test_create_message(dynamodb_manager, sample_session, sample_agent, sample_message):
    """Test creating a message in DynamoDB."""
    # Create session and agent
    dynamodb_manager.create_session(sample_session)
    dynamodb_manager.create_agent(sample_session.session_id, sample_agent)

    # Create message
    dynamodb_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Verify DynamoDB item created
    response = dynamodb_manager.client.get_item(
        TableName=dynamodb_manager.table_name,
        Key={
            "PK": {"S": f"session_{sample_session.session_id}"},
            "SK": {"S": f"agent_{sample_agent.agent_id}#message_{sample_message.message_id}"},
        },
    )
    assert "Item" in response
    assert response["Item"]["entity_type"]["S"] == "MESSAGE"

    data = dynamodb_manager.deserializer.deserialize(response["Item"]["data"])
    assert data["message_id"] == sample_message.message_id


def test_read_message(dynamodb_manager, sample_session, sample_agent, sample_message):
    """Test reading a message from DynamoDB."""
    dynamodb_manager.create_session(sample_session)
    dynamodb_manager.create_agent(sample_session.session_id, sample_agent)
    dynamodb_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    result = dynamodb_manager.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)
    assert isinstance(result.message_id, int)
    assert result.message_id == sample_message.message_id
    assert result.message["role"] == sample_message.message["role"]
    assert result.message["content"] == sample_message.message["content"]


def test_read_nonexistent_message(dynamodb_manager, sample_session, sample_agent):
    """Test reading a message that doesn't exist."""
    # Create session and agent, no message
    dynamodb_manager.create_session(sample_session)
    dynamodb_manager.create_agent(sample_session.session_id, sample_agent)

    # Read message
    result = dynamodb_manager.read_message(sample_session.session_id, sample_agent.agent_id, 999)

    assert result is None


def test_list_messages_all(dynamodb_manager, sample_session, sample_agent):
    """Test listing all messages from DynamoDB."""
    # Create session and agent
    dynamodb_manager.create_session(sample_session)
    dynamodb_manager.create_agent(sample_session.session_id, sample_agent)

    # Create multiple messages
    messages = []
    for i in range(5):
        message = SessionMessage(
            {
                "role": "user",
                "content": [ContentBlock(text=f"Message {i}")],
            },
            i,
        )
        messages.append(message)
        dynamodb_manager.create_message(sample_session.session_id, sample_agent.agent_id, message)

    # List all messages
    result = dynamodb_manager.list_messages(sample_session.session_id, sample_agent.agent_id)

    assert len(result) == 5
    for msg in result:
        assert isinstance(msg.message_id, int)


def test_list_messages_with_pagination(dynamodb_manager, sample_session, sample_agent):
    """Test listing messages with pagination in DynamoDB."""
    # Create session and agent
    dynamodb_manager.create_session(sample_session)
    dynamodb_manager.create_agent(sample_session.session_id, sample_agent)

    # Create multiple messages
    for index in range(10):
        message = SessionMessage.from_message(
            message={
                "role": "user",
                "content": [ContentBlock(text="test_message")],
            },
            index=index,
        )
        dynamodb_manager.create_message(sample_session.session_id, sample_agent.agent_id, message)

    # List with limit
    result = dynamodb_manager.list_messages(sample_session.session_id, sample_agent.agent_id, limit=3)
    assert len(result) == 3

    # List with offset
    result = dynamodb_manager.list_messages(sample_session.session_id, sample_agent.agent_id, offset=5)
    assert len(result) == 5


def test_update_message(dynamodb_manager, sample_session, sample_agent, sample_message):
    """Test updating a message in DynamoDB."""
    dynamodb_manager.create_session(sample_session)
    dynamodb_manager.create_agent(sample_session.session_id, sample_agent)
    dynamodb_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    sample_message.message["content"] = [ContentBlock(text="Updated content")]
    dynamodb_manager.update_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    result = dynamodb_manager.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)
    assert result.message["content"][0]["text"] == "Updated content"


def test_update_nonexistent_message(dynamodb_manager, sample_session, sample_agent, sample_message):
    """Test updating a message that doesn't exist."""
    dynamodb_manager.create_session(sample_session)
    dynamodb_manager.create_agent(sample_session.session_id, sample_agent)

    with pytest.raises(SessionException):
        dynamodb_manager.update_message(sample_session.session_id, sample_agent.agent_id, sample_message)


@pytest.mark.parametrize(
    "session_id",
    [
        "session_with_underscore",
        "session#with#hash",
        "session_and#both",
    ],
)
def test__get_session_pk_invalid_session_id(session_id, dynamodb_manager):
    with pytest.raises(
        ValueError, match=f"session_id={session_id} | id cannot contain underscore \(_\) or hash \(#\) characters"
    ):
        dynamodb_manager._get_session_pk(session_id)


@pytest.mark.parametrize(
    "agent_id",
    [
        "agent_with_underscore",
        "agent#with#hash",
        "agent_and#both",
    ],
)
def test__get_agent_sk_invalid_agent_id(agent_id, dynamodb_manager):
    with pytest.raises(
        ValueError, match=f"agent_id={agent_id} | id cannot contain underscore \(_\) or hash \(#\) characters"
    ):
        dynamodb_manager._get_agent_sk(agent_id)


@pytest.mark.parametrize(
    "message_id",
    [
        "not_an_int",
        None,
        [],
    ],
)
def test__get_message_sk_invalid_message_id(message_id, dynamodb_manager):
    """Test that message_id that is not an integer raises ValueError."""
    with pytest.raises(ValueError, match=r"message_id=<.*> \| message id must be an integer"):
        dynamodb_manager._get_message_sk("agent1", message_id)


def test_convert_decimals_to_native_types():
    """Test the Decimal conversion utility function."""
    # Test simple Decimal conversion
    assert _convert_decimals_to_native_types(Decimal("10")) == 10
    assert _convert_decimals_to_native_types(Decimal("10.5")) == 10.5
    assert _convert_decimals_to_native_types(Decimal("0")) == 0

    # Test nested dictionary conversion
    data = {
        "limit": Decimal("10"),
        "max_length": Decimal("8000"),
        "temperature": Decimal("0.5"),
        "name": "test",
        "enabled": True,
        "nested": {"count": Decimal("42"), "ratio": Decimal("3.14")},
    }

    result = _convert_decimals_to_native_types(data)

    assert result["limit"] == 10
    assert isinstance(result["limit"], int)
    assert result["max_length"] == 8000
    assert isinstance(result["max_length"], int)
    assert result["temperature"] == 0.5
    assert isinstance(result["temperature"], float)
    assert result["name"] == "test"
    assert result["enabled"] is True
    assert result["nested"]["count"] == 42
    assert isinstance(result["nested"]["count"], int)
    assert result["nested"]["ratio"] == 3.14
    assert isinstance(result["nested"]["ratio"], float)

    # Test list conversion
    list_data = [Decimal("1"), Decimal("2.5"), "string", {"nested": Decimal("100")}]
    result = _convert_decimals_to_native_types(list_data)

    assert result[0] == 1
    assert isinstance(result[0], int)
    assert result[1] == 2.5
    assert isinstance(result[1], float)
    assert result[2] == "string"
    assert result[3]["nested"] == 100
    assert isinstance(result[3]["nested"], int)


# S3 Offloading Tests


@pytest.fixture(scope="function")
def s3_bucket(mocked_aws):
    """S3 bucket name for testing."""
    # Create the bucket
    s3_client = boto3.client("s3", region_name="us-west-2")
    s3_client.create_bucket(Bucket="test-session-bucket", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    return "test-session-bucket"


@pytest.fixture
def dynamodb_manager_with_s3(mocked_aws, dynamodb_table, s3_bucket):
    """Create DynamoDBSessionManager with S3 bucket."""
    yield DynamoDBSessionManager(
        session_id="test",
        table_name=dynamodb_table,
        region_name="us-west-2",
        s3_bucket=s3_bucket,
        max_item_size=1000,  # Small size to trigger S3 offloading
    )


@pytest.fixture
def large_image_message():
    """Create message with large image content."""
    # Create a large image (50KB+ to trigger S3 offloading)
    large_image_data = b"PNG\r\n\x1a\n" + b"\x00" * 50000

    return SessionMessage.from_message(
        message={
            "role": "user",
            "content": [
                ContentBlock(text="Here's a large image:"),
                ContentBlock(image={"format": "png", "source": {"bytes": large_image_data}}),
            ],
        },
        index=0,
    )


def test_create_large_message_s3_offload(
    dynamodb_manager_with_s3, sample_session, sample_agent, large_image_message, s3_bucket
):
    """Test creating a large message that gets offloaded to S3."""
    # Create session and agent
    dynamodb_manager_with_s3.create_session(sample_session)
    dynamodb_manager_with_s3.create_agent(sample_session.session_id, sample_agent)

    # Create large message
    dynamodb_manager_with_s3.create_message(sample_session.session_id, sample_agent.agent_id, large_image_message)

    # Verify DynamoDB contains S3 reference
    response = dynamodb_manager_with_s3.client.get_item(
        TableName=dynamodb_manager_with_s3.table_name,
        Key={
            "PK": {"S": f"session_{sample_session.session_id}"},
            "SK": {"S": f"agent_{sample_agent.agent_id}#message_{large_image_message.message_id}"},
        },
    )
    assert "Item" in response
    data = dynamodb_manager_with_s3.deserializer.deserialize(response["Item"]["data"])

    # Check that message content is an S3 reference
    assert isinstance(data["message"]["content"], dict)
    assert "s3_reference" in data["message"]["content"]
    assert data["message"]["content"]["s3_reference"]["bucket"] == s3_bucket
    assert "message_0.json" in data["message"]["content"]["s3_reference"]["key"]

    # Verify S3 object exists
    import json

    s3 = boto3.client("s3", region_name="us-west-2")
    s3_key = data["message"]["content"]["s3_reference"]["key"]
    s3_response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    s3_data = json.loads(s3_response["Body"].read().decode("utf-8"))

    # Verify S3 contains full message
    assert s3_data["message_id"] == large_image_message.message_id
    assert len(s3_data["message"]["content"]) == 2
    assert s3_data["message"]["content"][0]["text"] == "Here's a large image:"
    assert "image" in s3_data["message"]["content"][1]


def test_read_large_message_from_s3(dynamodb_manager_with_s3, sample_session, sample_agent, large_image_message):
    """Test reading a large message that was offloaded to S3."""
    # Create session, agent, and large message
    dynamodb_manager_with_s3.create_session(sample_session)
    dynamodb_manager_with_s3.create_agent(sample_session.session_id, sample_agent)
    dynamodb_manager_with_s3.create_message(sample_session.session_id, sample_agent.agent_id, large_image_message)

    # Read message back
    result = dynamodb_manager_with_s3.read_message(
        sample_session.session_id, sample_agent.agent_id, large_image_message.message_id
    )

    # Verify full message is loaded from S3
    assert result.message_id == large_image_message.message_id
    assert result.message["role"] == "user"
    assert len(result.message["content"]) == 2
    assert result.message["content"][0]["text"] == "Here's a large image:"
    assert "image" in result.message["content"][1]
    assert result.message["content"][1]["image"]["format"] == "png"
    assert len(result.message["content"][1]["image"]["source"]["bytes"]) > 50000


def test_list_large_messages_from_s3(dynamodb_manager_with_s3, sample_session, sample_agent, large_image_message):
    """Test listing messages that include S3-offloaded messages."""
    # Create session and agent
    dynamodb_manager_with_s3.create_session(sample_session)
    dynamodb_manager_with_s3.create_agent(sample_session.session_id, sample_agent)

    # Create normal message
    normal_message = SessionMessage.from_message(
        message={"role": "user", "content": [ContentBlock(text="small message")]},
        index=1,
    )
    dynamodb_manager_with_s3.create_message(sample_session.session_id, sample_agent.agent_id, large_image_message)
    dynamodb_manager_with_s3.create_message(sample_session.session_id, sample_agent.agent_id, normal_message)

    # List all messages
    result = dynamodb_manager_with_s3.list_messages(sample_session.session_id, sample_agent.agent_id)

    assert len(result) == 2

    # First message (large, from S3)
    assert result[0].message_id == 0
    assert len(result[0].message["content"]) == 2
    assert "image" in result[0].message["content"][1]

    # Second message (normal, from DynamoDB)
    assert result[1].message_id == 1
    assert result[1].message["content"][0]["text"] == "small message"


def test_update_large_message_s3_offload(dynamodb_manager_with_s3, sample_session, sample_agent, large_image_message):
    """Test updating a large message that gets offloaded to S3."""
    # Create session, agent, and message
    dynamodb_manager_with_s3.create_session(sample_session)
    dynamodb_manager_with_s3.create_agent(sample_session.session_id, sample_agent)
    dynamodb_manager_with_s3.create_message(sample_session.session_id, sample_agent.agent_id, large_image_message)

    # Update with another large message
    updated_large_data = b"PNG\r\n\x1a\n" + b"\xff" * 60000  # Different large image
    large_image_message.message["content"] = [
        ContentBlock(text="Updated large image:"),
        ContentBlock(image={"format": "png", "source": {"bytes": updated_large_data}}),
    ]

    dynamodb_manager_with_s3.update_message(sample_session.session_id, sample_agent.agent_id, large_image_message)

    # Read updated message
    result = dynamodb_manager_with_s3.read_message(
        sample_session.session_id, sample_agent.agent_id, large_image_message.message_id
    )

    assert result.message["content"][0]["text"] == "Updated large image:"
    assert len(result.message["content"][1]["image"]["source"]["bytes"]) > 60000


def test_s3_offload_without_bucket_configured(dynamodb_manager, sample_session, sample_agent, large_image_message):
    """Test that large messages without S3 bucket configured still work (with warning)."""
    # Create session and agent
    dynamodb_manager.create_session(sample_session)
    dynamodb_manager.create_agent(sample_session.session_id, sample_agent)

    # Create large message (should log warning but still work)
    dynamodb_manager.create_message(sample_session.session_id, sample_agent.agent_id, large_image_message)

    # Read message back (should work normally)
    result = dynamodb_manager.read_message(
        sample_session.session_id, sample_agent.agent_id, large_image_message.message_id
    )
    assert result.message_id == large_image_message.message_id


def test_s3_prefix_functionality(
    mocked_aws, dynamodb_table, s3_bucket, sample_session, sample_agent, large_image_message
):
    """Test S3 prefix functionality for organizing keys."""
    # Create manager with S3 prefix
    manager_with_prefix = DynamoDBSessionManager(
        session_id="test",
        table_name=dynamodb_table,
        region_name="us-west-2",
        s3_bucket=s3_bucket,
        s3_prefix="my-app/prod",
        max_item_size=1000,
    )

    manager_with_prefix.create_session(sample_session)
    manager_with_prefix.create_agent(sample_session.session_id, sample_agent)
    manager_with_prefix.create_message(sample_session.session_id, sample_agent.agent_id, large_image_message)

    # Verify S3 key uses prefix
    response = manager_with_prefix.client.get_item(
        TableName=manager_with_prefix.table_name,
        Key={
            "PK": {"S": f"session_{sample_session.session_id}"},
            "SK": {"S": f"agent_{sample_agent.agent_id}#message_{large_image_message.message_id}"},
        },
    )
    data = manager_with_prefix.deserializer.deserialize(response["Item"]["data"])
    s3_key = data["message"]["content"]["s3_reference"]["key"]

    # Should start with the prefix
    assert s3_key.startswith("my-app/prod/sessions/")
    assert "message_0.json" in s3_key


def test_delete_session_with_s3_cleanup(
    dynamodb_manager_with_s3, sample_session, sample_agent, large_image_message, s3_bucket
):
    """Test that delete_session cleans up S3 objects for offloaded messages."""
    # Create session, agent, and large message
    dynamodb_manager_with_s3.create_session(sample_session)
    dynamodb_manager_with_s3.create_agent(sample_session.session_id, sample_agent)
    dynamodb_manager_with_s3.create_message(sample_session.session_id, sample_agent.agent_id, large_image_message)

    # Get S3 key from DynamoDB
    response = dynamodb_manager_with_s3.client.get_item(
        TableName=dynamodb_manager_with_s3.table_name,
        Key={
            "PK": {"S": f"session_{sample_session.session_id}"},
            "SK": {"S": f"agent_{sample_agent.agent_id}#message_{large_image_message.message_id}"},
        },
    )
    data = dynamodb_manager_with_s3.deserializer.deserialize(response["Item"]["data"])
    s3_key = data["message"]["content"]["s3_reference"]["key"]

    # Verify S3 object exists
    s3 = boto3.client("s3", region_name="us-west-2")
    s3.head_object(Bucket=s3_bucket, Key=s3_key)  # Should not raise exception

    # Delete session
    dynamodb_manager_with_s3.delete_session(sample_session.session_id)

    # Verify S3 object is deleted
    from botocore.exceptions import ClientError

    with pytest.raises(ClientError) as exc_info:
        s3.head_object(Bucket=s3_bucket, Key=s3_key)
    assert exc_info.value.response["Error"]["Code"] == "404"

    # Verify DynamoDB items are deleted
    assert dynamodb_manager_with_s3.read_session(sample_session.session_id) is None
