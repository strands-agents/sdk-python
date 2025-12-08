import unittest.mock

import pytest

from strands import Agent, tool


@pytest.fixture
def randint():
    with unittest.mock.patch("strands.tools._caller.random.randint") as mock:
        yield mock


@pytest.fixture
def model():
    return unittest.mock.Mock()


@pytest.fixture
def test_tool():
    @tool(name="test_tool")
    def function(random_string: str) -> str:
        return random_string

    return function


@pytest.fixture
def agent(model, test_tool):
    return Agent(model=model, tools=[test_tool])


def test_agent_tool(randint, agent):
    conversation_manager_spy = unittest.mock.Mock(wraps=agent.conversation_manager)
    agent.conversation_manager = conversation_manager_spy

    randint.return_value = 1

    tru_result = agent.tool.test_tool(random_string="abcdEfghI123")
    exp_result = {
        "content": [
            {
                "text": "abcdEfghI123",
            },
        ],
        "status": "success",
        "toolUseId": "tooluse_test_tool_1",
    }

    assert tru_result == exp_result
    conversation_manager_spy.apply_management.assert_called_with(agent)


@pytest.mark.asyncio
async def test_agent_tool_in_async_context(randint, agent):
    randint.return_value = 123

    tru_result = agent.tool.test_tool(random_string="abcdEfghI123")
    exp_result = {
        "content": [
            {
                "text": "abcdEfghI123",
            },
        ],
        "status": "success",
        "toolUseId": "tooluse_test_tool_123",
    }

    assert tru_result == exp_result


def test_agent_tool_user_message_override(agent):
    agent.tool.test_tool(random_string="abcdEfghI123", user_message_override="test override")

    tru_message = agent.messages[0]
    exp_message = {
        "content": [
            {
                "text": "test override\n",
            },
            {
                "text": (
                    'agent.tool.test_tool direct tool call.\nInput parameters: {"random_string": "abcdEfghI123"}\n'
                ),
            },
        ],
        "role": "user",
    }

    assert tru_message == exp_message


def test_agent_tool_do_not_record_tool(agent):
    agent.record_direct_tool_call = False
    agent.tool.test_tool(random_string="abcdEfghI123", user_message_override="test override")

    tru_messages = agent.messages
    exp_messages = []

    assert tru_messages == exp_messages


def test_agent_tool_do_not_record_tool_with_method_override(agent):
    agent.record_direct_tool_call = True
    agent.tool.test_tool(
        random_string="abcdEfghI123", user_message_override="test override", record_direct_tool_call=False
    )

    tru_messages = agent.messages
    exp_messages = []

    assert tru_messages == exp_messages


def test_agent_tool_tool_does_not_exist(agent):
    with pytest.raises(AttributeError):
        agent.tool.does_not_exist()


def test_agent_tool_no_parameter_conflict(agent, randint):
    @tool(name="system_prompter")
    def function(system_prompt: str) -> str:
        return system_prompt

    agent.tool_registry.register_tool(function)

    randint.return_value = 1

    tru_result = agent.tool.system_prompter(system_prompt="tool prompt")
    exp_result = {"toolUseId": "tooluse_system_prompter_1", "status": "success", "content": [{"text": "tool prompt"}]}
    assert tru_result == exp_result


def test_agent_tool_with_name_normalization(agent, randint):
    tool_name = "system-prompter"

    @tool(name=tool_name)
    def function(system_prompt: str) -> str:
        return system_prompt

    agent.tool_registry.register_tool(function)

    randint.return_value = 1

    tru_result = agent.tool.system_prompter(system_prompt="tool prompt")
    exp_result = {"toolUseId": "tooluse_system_prompter_1", "status": "success", "content": [{"text": "tool prompt"}]}
    assert tru_result == exp_result


def test_agent_tool_with_no_normalized_match(agent, randint):
    randint.return_value = 1

    with pytest.raises(AttributeError) as err:
        agent.tool.system_prompter_1(system_prompt="tool prompt")

    assert str(err.value) == "Tool 'system_prompter_1' not found"


def test_agent_tool_non_serializable_parameter_filtering(agent, randint):
    """Test that non-serializable objects in tool parameters are properly filtered during tool call recording."""
    randint.return_value = 42

    # Create a non-serializable object (Agent instance)
    another_agent = Agent()

    # This should not crash even though we're passing non-serializable objects
    result = agent.tool.test_tool(
        random_string="test_value",
        non_serializable_agent=another_agent,  # This would previously cause JSON serialization error
        user_message_override="Testing non-serializable parameter filtering",
    )

    # Verify the tool executed successfully
    expected_result = {
        "content": [{"text": "test_value"}],
        "status": "success",
        "toolUseId": "tooluse_test_tool_42",
    }
    assert result == expected_result

    # The key test: this should not crash during execution
    # Check that we have messages recorded (exact count may vary)
    assert len(agent.messages) > 0

    # Check user message with filtered parameters - this is the main test for the bug fix
    user_message = agent.messages[0]
    assert user_message["role"] == "user"
    assert len(user_message["content"]) == 2

    # Check override message
    assert user_message["content"][0]["text"] == "Testing non-serializable parameter filtering\n"

    # Check tool call description with filtered parameters - this is where JSON serialization would fail
    tool_call_text = user_message["content"][1]["text"]
    assert "agent.tool.test_tool direct tool call." in tool_call_text
    assert '"random_string": "test_value"' in tool_call_text
    assert '"non_serializable_agent": "<<non-serializable: Agent>>"' not in tool_call_text


def test_agent_tool_no_non_serializable_parameters(agent, randint):
    """Test that normal tool calls with only serializable parameters work unchanged."""
    randint.return_value = 555

    # Call with only serializable parameters
    result = agent.tool.test_tool(random_string="normal_call", user_message_override="Normal tool call test")

    # Verify successful execution
    expected_result = {
        "content": [{"text": "normal_call"}],
        "status": "success",
        "toolUseId": "tooluse_test_tool_555",
    }
    assert result == expected_result

    # Check message recording works normally
    assert len(agent.messages) > 0
    user_message = agent.messages[0]
    tool_call_text = user_message["content"][1]["text"]

    # Verify normal parameter serialization (no filtering needed)
    assert "agent.tool.test_tool direct tool call." in tool_call_text
    assert '"random_string": "normal_call"' in tool_call_text
    # Should not contain any "<<non-serializable:" strings
    assert "<<non-serializable:" not in tool_call_text


def test_agent_tool_record_direct_tool_call_disabled_with_non_serializable(agent, randint):
    """Test that when record_direct_tool_call is disabled, non-serializable parameters don't cause issues."""
    randint.return_value = 777

    # Disable tool call recording
    agent.record_direct_tool_call = False

    # This should work fine even with non-serializable parameters since recording is disabled
    result = agent.tool.test_tool(
        random_string="no_recording", non_serializable_agent=Agent(), user_message_override="This shouldn't be recorded"
    )

    # Verify successful execution
    expected_result = {
        "content": [{"text": "no_recording"}],
        "status": "success",
        "toolUseId": "tooluse_test_tool_777",
    }
    assert result == expected_result

    # Verify no messages were recorded
    assert len(agent.messages) == 0


def test_agent_tool_call_parameter_filtering_integration(randint):
    """Test that tool calls properly filter parameters in message recording."""
    randint.return_value = 42

    @tool
    def test_tool(action: str) -> str:
        """Test tool with single parameter."""
        return action

    agent = Agent(tools=[test_tool])

    # Call tool with extra non-spec parameters
    result = agent.tool.test_tool(
        action="test_value",
        agent=agent,  # Should be filtered out
        extra_param="filtered",  # Should be filtered out
    )

    # Verify tool executed successfully
    assert result["status"] == "success"
    assert result["content"] == [{"text": "test_value"}]

    # Check that only spec parameters are recorded in message history
    assert len(agent.messages) > 0
    user_message = agent.messages[0]
    tool_call_text = user_message["content"][0]["text"]

    # Should only contain the 'action' parameter
    assert '"action": "test_value"' in tool_call_text
    assert '"agent"' not in tool_call_text
    assert '"extra_param"' not in tool_call_text


def test_agent_tool_caller_interrupt():
    @tool(context=True)
    def test_tool(tool_context):
        tool_context.interrupt("test-interrupt")

    agent = Agent(tools=[test_tool])

    exp_message = r"cannot raise interrupt in direct tool call"
    with pytest.raises(RuntimeError, match=exp_message):
        agent.tool.test_tool(agent=agent)

    tru_state = agent._interrupt_state.to_dict()
    exp_state = {
        "activated": False,
        "context": {},
        "interrupts": {},
    }
    assert tru_state == exp_state

    tru_messages = agent.messages
    exp_messages = []
    assert tru_messages == exp_messages


def test_agent_tool_caller_interrupt_activated():
    agent = Agent()
    agent._interrupt_state.activated = True

    exp_message = r"cannot directly call tool during interrupt"
    with pytest.raises(RuntimeError, match=exp_message):
        agent.tool.test_tool()
