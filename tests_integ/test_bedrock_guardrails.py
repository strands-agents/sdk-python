import tempfile
import time
from uuid import uuid4

import boto3
import pytest

from strands import Agent, tool
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


@pytest.mark.parametrize(
    "guardrail_trace",
    [
        pytest.param("disabled", marks=pytest.mark.xfail(reason='redact fails with trace="disabled"')),
        "enabled",
        "enabled_full",
    ],
)
def test_guardrail_input_intervention(boto_session, bedrock_guardrail, guardrail_trace):
    bedrock_model = BedrockModel(
        guardrail_id=bedrock_guardrail,
        guardrail_version="DRAFT",
        boto_session=boto_session,
        guardrail_trace=guardrail_trace,
        guardrail_redact_input_message="Redacted.",
    )

    agent = Agent(model=bedrock_model, system_prompt="You are a helpful assistant.", callback_handler=None)

    response1 = agent("CACTUS")
    response2 = agent("Hello!")

    assert response1.stop_reason == "guardrail_intervened"
    assert str(response1).strip() == BLOCKED_INPUT
    assert response2.stop_reason != "guardrail_intervened"
    assert str(response2).strip() != BLOCKED_INPUT
    assert agent.messages[0]["content"][0]["text"] == "Redacted."


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

    """
    In async streaming: The buffering is non-blocking. 
    Tokens are streamed while Guardrails processes the buffered content in the background. 
    This means the response may be returned before Guardrails has finished processing.
    As a result, we cannot guarantee that the REDACT_MESSAGE is in the response
    """
    if processing_mode == "sync":
        assert BLOCKED_OUTPUT in str(response1)
        assert response2.stop_reason != "guardrail_intervened"
        assert BLOCKED_OUTPUT not in str(response2)
    else:
        cactus_returned_in_response1_blocked_by_input_guardrail = BLOCKED_INPUT in str(response2)
        cactus_blocked_in_response1_allows_next_response = (
            BLOCKED_OUTPUT not in str(response2) and response2.stop_reason != "guardrail_intervened"
        )
        assert (
            cactus_returned_in_response1_blocked_by_input_guardrail or cactus_blocked_in_response1_allows_next_response
        )


@pytest.mark.parametrize("guardrail_trace", ["enabled", "enabled_full"])
@pytest.mark.parametrize("processing_mode", ["sync", "async"])
def test_guardrail_output_intervention_redact_output(bedrock_guardrail, processing_mode, guardrail_trace):
    REDACT_MESSAGE = "Redacted."
    bedrock_model = BedrockModel(
        guardrail_id=bedrock_guardrail,
        guardrail_version="DRAFT",
        guardrail_stream_processing_mode=processing_mode,
        guardrail_trace=guardrail_trace,
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

    """
    In async streaming: The buffering is non-blocking. 
    Tokens are streamed while Guardrails processes the buffered content in the background. 
    This means the response may be returned before Guardrails has finished processing.
    As a result, we cannot guarantee that the REDACT_MESSAGE is in the response.
    """
    if processing_mode == "sync":
        assert REDACT_MESSAGE in str(response1)
        assert response2.stop_reason != "guardrail_intervened"
        assert REDACT_MESSAGE not in str(response2)
    else:
        cactus_returned_in_response1_blocked_by_input_guardrail = BLOCKED_INPUT in str(response2)
        cactus_blocked_in_response1_allows_next_response = (
            REDACT_MESSAGE not in str(response2) and response2.stop_reason != "guardrail_intervened"
        )
        assert (
            cactus_returned_in_response1_blocked_by_input_guardrail or cactus_blocked_in_response1_allows_next_response
        )


@pytest.mark.parametrize("processing_mode", ["sync", "async"])
def test_guardrail_intervention_properly_redacts_tool_result(bedrock_guardrail, processing_mode):
    INPUT_REDACT_MESSAGE = "Input redacted."
    OUTPUT_REDACT_MESSAGE = "Output redacted."
    bedrock_model = BedrockModel(
        guardrail_id=bedrock_guardrail,
        guardrail_version="DRAFT",
        guardrail_stream_processing_mode=processing_mode,
        guardrail_redact_output=True,
        guardrail_redact_input_message=INPUT_REDACT_MESSAGE,
        guardrail_redact_output_message=OUTPUT_REDACT_MESSAGE,
        region_name="us-east-1",
    )

    @tool
    def list_users() -> str:
        "List my users"
        return """[{"name": "Jerry Merry"}, {"name": "Mr. CACTUS"}]"""

    agent = Agent(
        model=bedrock_model,
        system_prompt="You are a helpful assistant.",
        callback_handler=None,
        load_tools_from_directory=False,
        tools=[list_users],
    )

    response1 = agent("List my users.")
    response2 = agent("Thank you!")

    """ Message sequence:
    0 (user): request1
    1 (assistant): reasoning + tool call
    2 (user): tool result
    3 (assistant): response1 -> output guardrail intervenes
    4 (user): request2
    5 (assistant): response2

    Guardrail intervened on output in message 3 will cause
    the redaction of the preceding input (message 2) and message 3.
    """

    assert response1.stop_reason == "guardrail_intervened"

    if processing_mode == "sync":
        """ In sync mode the guardrail processing is blocking.
        The response is already blocked and redacted. """

        assert OUTPUT_REDACT_MESSAGE in str(response1)
        assert OUTPUT_REDACT_MESSAGE not in str(response2)

    """
    In async streaming, the buffering is non-blocking,
    so the response may be returned before Guardrails has finished processing.

    However, in both sync and async, with guardrail_redact_output=True:
    
    1. the content should be properly redacted in memory, so that
    response2 is not blocked by guardrails;
    """
    assert response2.stop_reason != "guardrail_intervened"

    """
    2. the tool result block should be redacted properly, so that the
    conversation is not corrupted.
    """

    tool_call = [b for b in agent.messages[1]["content"] if "toolUse" in b][0]["toolUse"]
    tool_result = [b for b in agent.messages[2]["content"] if "toolResult" in b][0]["toolResult"]
    assert tool_result["toolUseId"] == tool_call["toolUseId"]
    assert tool_result["content"][0]["text"] == INPUT_REDACT_MESSAGE


def test_guardrail_latest_message(boto_session, bedrock_guardrail):
    """Test that guardrail_latest_user_message wraps both text and image in the latest user message."""
    bedrock_model = BedrockModel(
        guardrail_id=bedrock_guardrail,
        guardrail_version="DRAFT",
        guardrail_latest_message=True,
        boto_session=boto_session,
    )

    # Create agent with multimodal content in latest user message
    agent = Agent(
        model=bedrock_model,
        system_prompt="You are a helpful assistant.",
        callback_handler=None,
        messages=[
            {"role": "user", "content": [{"text": "First message"}]},
            {"role": "assistant", "content": [{"text": "Hello!"}]},
            {
                "role": "user",
                "content": [
                    {"text": "Look at this image"},
                    {"image": {"format": "png", "source": {"bytes": b"fake_image_data"}}},
                ],
            },
        ],
    )

    response = agent("What do you see?")
    assert response.stop_reason != "guardrail_intervened"


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
