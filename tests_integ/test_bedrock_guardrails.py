import tempfile
import time
from uuid import uuid4

import boto3
import pytest

from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.session.file_session_manager import FileSessionManager

BLOCKED_INPUT = "BLOCKED_INPUT"
BLOCKED_OUTPUT = "BLOCKED_OUTPUT"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="module")
def boto_session():
    return boto3.Session(region_name="us-east-1")


@pytest.fixture(scope="module")
def bedrock_guardrail(boto_session):
    """
    Fixture that creates a guardrail before tests if it doesn't already exist."
    """

    client = boto_session.client("bedrock")

    guardrail_name = "test-guardrail-block-cactus"
    guardrail_id = get_guardrail_id(client, guardrail_name)

    if guardrail_id:
        print(f"Guardrail {guardrail_name} already exists with ID: {guardrail_id}")
    else:
        print(f"Creating guardrail {guardrail_name}")
        response = client.create_guardrail(
            name=guardrail_name,
            description="Testing Guardrail",
            wordPolicyConfig={
                "wordsConfig": [
                    {
                        "text": "CACTUS",
                        "inputAction": "BLOCK",
                        "outputAction": "BLOCK",
                        "inputEnabled": True,
                        "outputEnabled": True,
                    },
                ],
            },
            blockedInputMessaging=BLOCKED_INPUT,
            blockedOutputsMessaging=BLOCKED_OUTPUT,
        )
        guardrail_id = response.get("guardrailId")
        print(f"Created test guardrail with ID: {guardrail_id}")
        wait_for_guardrail_active(client, guardrail_id)
    return guardrail_id


def get_guardrail_id(client, guardrail_name):
    """
    Retrieves the ID of a guardrail by its name.

    Args:
        client: The Bedrock client instance
        guardrail_name: Name of the guardrail to look up

    Returns:
        str: The ID of the guardrail if found, None otherwise
    """
    response = client.list_guardrails()
    for guardrail in response.get("guardrails", []):
        if guardrail["name"] == guardrail_name:
            return guardrail["id"]
    return None


def wait_for_guardrail_active(bedrock_client, guardrail_id, max_attempts=10, delay=5):
    """
    Wait for the guardrail to become active
    """
    for _ in range(max_attempts):
        response = bedrock_client.get_guardrail(guardrailIdentifier=guardrail_id)
        status = response.get("status")

        if status == "READY":
            print(f"Guardrail {guardrail_id} is now active")
            return True

        print(f"Waiting for guardrail to become active. Current status: {status}")
        time.sleep(delay)

    print(f"Guardrail did not become active within {max_attempts * delay} seconds.")
    raise RuntimeError("Guardrail did not become active.")


def test_guardrail_input_intervention(boto_session, bedrock_guardrail):
    bedrock_model = BedrockModel(
        guardrail_id=bedrock_guardrail,
        guardrail_version="DRAFT",
        boto_session=boto_session,
    )

    agent = Agent(model=bedrock_model, system_prompt="You are a helpful assistant.", callback_handler=None)

    response1 = agent("CACTUS")
    response2 = agent("Hello!")

    assert response1.stop_reason == "guardrail_intervened"
    assert str(response1).strip() == BLOCKED_INPUT
    assert response2.stop_reason != "guardrail_intervened"
    assert str(response2).strip() != BLOCKED_INPUT


@pytest.mark.parametrize("processing_mode", ["sync", "async"])
def test_guardrail_output_intervention(boto_session, bedrock_guardrail, processing_mode):
    bedrock_model = BedrockModel(
        guardrail_id=bedrock_guardrail,
        guardrail_version="DRAFT",
        guardrail_redact_output=False,
        guardrail_stream_processing_mode=processing_mode,
        boto_session=boto_session,
    )

    agent = Agent(
        model=bedrock_model,
        system_prompt="When asked to say the word, say CACTUS.",
        callback_handler=None,
        load_tools_from_directory=False,
    )

    response1 = agent("Say the word.")
    response2 = agent("Hello!")
    assert response1.stop_reason == "guardrail_intervened"
    assert BLOCKED_OUTPUT in str(response1)
    assert response2.stop_reason != "guardrail_intervened"
    assert BLOCKED_OUTPUT not in str(response2)


@pytest.mark.parametrize("processing_mode", ["sync", "async"])
def test_guardrail_output_intervention_redact_output(bedrock_guardrail, processing_mode):
    REDACT_MESSAGE = "Redacted."
    bedrock_model = BedrockModel(
        guardrail_id=bedrock_guardrail,
        guardrail_version="DRAFT",
        guardrail_stream_processing_mode=processing_mode,
        guardrail_redact_output=True,
        guardrail_redact_output_message=REDACT_MESSAGE,
        region_name="us-east-1",
    )

    agent = Agent(
        model=bedrock_model,
        system_prompt="When asked to say the word, say CACTUS.",
        callback_handler=None,
        load_tools_from_directory=False,
    )

    response1 = agent("Say the word.")
    response2 = agent("Hello!")
    assert response1.stop_reason == "guardrail_intervened"
    assert REDACT_MESSAGE in str(response1)
    assert response2.stop_reason != "guardrail_intervened"
    assert REDACT_MESSAGE not in str(response2)


def test_guardrail_input_intervention_properly_redacts_in_session(boto_session, bedrock_guardrail, temp_dir):
    bedrock_model = BedrockModel(
        guardrail_id=bedrock_guardrail,
        guardrail_version="DRAFT",
        boto_session=boto_session,
        guardrail_redact_input_message="BLOCKED!",
    )

    test_session_id = str(uuid4())
    session_manager = FileSessionManager(session_id=test_session_id)

    agent = Agent(
        model=bedrock_model,
        system_prompt="You are a helpful assistant.",
        callback_handler=None,
        session_manager=session_manager,
    )

    assert session_manager.read_agent(test_session_id, agent.agent_id) is not None

    response1 = agent("CACTUS")

    assert response1.stop_reason == "guardrail_intervened"
    assert agent.messages[0]["content"][0]["text"] == "BLOCKED!"
    user_input_session_message = session_manager.list_messages(test_session_id, agent.agent_id)[0]
    # Assert persisted message is equal to the redacted message in the agent
    assert user_input_session_message.to_message() == agent.messages[0]

    # Restore an agent from the session, confirm input is still redacted
    session_manager_2 = FileSessionManager(session_id=test_session_id)
    agent_2 = Agent(
        model=bedrock_model,
        system_prompt="You are a helpful assistant.",
        callback_handler=None,
        session_manager=session_manager_2,
    )

    # Assert that the restored agent redacted message is equal to the original agent
    assert agent.messages[0] == agent_2.messages[0]
