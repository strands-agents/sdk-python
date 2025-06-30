import copy
import importlib
import os
import textwrap
import unittest.mock

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.agent import AgentResult
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.handlers.callback_handler import PrintingCallbackHandler, null_callback_handler
from strands.models.bedrock import DEFAULT_BEDROCK_MODEL_ID, BedrockModel
from strands.types.content import Messages
from strands.types.exceptions import ContextWindowOverflowException, EventLoopException


@pytest.fixture
def mock_randint():
    with unittest.mock.patch.object(strands.agent.agent.random, "randint") as mock:
        yield mock


@pytest.fixture
def mock_model(request):
    def converse(*args, **kwargs):
        return mock.mock_converse(*copy.deepcopy(args), **copy.deepcopy(kwargs))

    mock = unittest.mock.Mock(spec=getattr(request, "param", None))
    mock.configure_mock(mock_converse=unittest.mock.MagicMock())
    mock.converse.side_effect = converse

    return mock


@pytest.fixture
def system_prompt(request):
    return request.param if hasattr(request, "param") else "You are a helpful assistant."


@pytest.fixture
def callback_handler():
    return unittest.mock.Mock()


@pytest.fixture
def messages(request):
    return request.param if hasattr(request, "param") else []


@pytest.fixture
def mock_event_loop_cycle():
    with unittest.mock.patch("strands.agent.agent.event_loop_cycle") as mock:
        yield mock


@pytest.fixture
def tool_registry():
    return strands.tools.registry.ToolRegistry()


@pytest.fixture
def tool_decorated():
    @strands.tools.tool(name="tool_decorated")
    def function(random_string: str) -> str:
        return random_string

    return function


@pytest.fixture
def tool_module(tmp_path):
    tool_definition = textwrap.dedent("""
        TOOL_SPEC = {
            "name": "tool_module",
            "description": "tool module",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        }

        def tool_module():
            return
    """)
    tool_path = tmp_path / "tool_module.py"
    tool_path.write_text(tool_definition)

    return str(tool_path)


@pytest.fixture
def tool_imported(tmp_path, monkeypatch):
    tool_definition = textwrap.dedent("""
        TOOL_SPEC = {
            "name": "tool_imported",
            "description": "tool imported",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        }

        def tool_imported():
            return
    """)
    tool_path = tmp_path / "tool_imported.py"
    tool_path.write_text(tool_definition)

    init_path = tmp_path / "__init__.py"
    init_path.touch()

    monkeypatch.syspath_prepend(str(tmp_path))

    dot_path = ".".join(os.path.splitext(tool_path)[0].split(os.sep)[-1:])
    return importlib.import_module(dot_path)


@pytest.fixture
def tool(tool_decorated, tool_registry):
    function_tool = strands.tools.tools.FunctionTool(tool_decorated, tool_name="tool_decorated")
    tool_registry.register_tool(function_tool)

    return function_tool


@pytest.fixture
def tools(request, tool):
    return request.param if hasattr(request, "param") else [tool_decorated]


@pytest.fixture
def agent(
    mock_model,
    system_prompt,
    callback_handler,
    messages,
    tools,
    tool,
    tool_registry,
    tool_decorated,
    request,
):
    agent = Agent(
        model=mock_model,
        system_prompt=system_prompt,
        callback_handler=callback_handler,
        messages=messages,
        tools=tools,
    )

    # Only register the tool directly if tools wasn't parameterized
    if not hasattr(request, "param") or request.param is None:
        # Create a new function tool directly from the decorated function
        function_tool = strands.tools.tools.FunctionTool(tool_decorated, tool_name="tool_decorated")
        agent.tool_registry.register_tool(function_tool)

    return agent


@pytest.fixture
def alist():
    async def gen(items):
        for item in items:
            yield item

    return gen


def test_agent__init__tool_loader_format(tool_decorated, tool_module, tool_imported, tool_registry):
    _ = tool_registry

    agent = Agent(tools=[tool_decorated, tool_module, tool_imported])

    tru_tool_names = sorted(tool_spec["toolSpec"]["name"] for tool_spec in agent.tool_config["tools"])
    exp_tool_names = ["tool_decorated", "tool_imported", "tool_module"]

    assert tru_tool_names == exp_tool_names


def test_agent__init__tool_loader_dict(tool_module, tool_registry):
    _ = tool_registry

    agent = Agent(tools=[{"name": "tool_module", "path": tool_module}])

    tru_tool_names = sorted(tool_spec["toolSpec"]["name"] for tool_spec in agent.tool_config["tools"])
    exp_tool_names = ["tool_module"]

    assert tru_tool_names == exp_tool_names


def test_agent__init__invalid_max_parallel_tools(tool_registry):
    _ = tool_registry

    with pytest.raises(ValueError):
        Agent(max_parallel_tools=0)


def test_agent__init__one_max_parallel_tools_succeeds(tool_registry):
    _ = tool_registry

    Agent(max_parallel_tools=1)


def test_agent__init__with_default_model():
    agent = Agent()

    assert isinstance(agent.model, BedrockModel)
    assert agent.model.config["model_id"] == DEFAULT_BEDROCK_MODEL_ID


def test_agent__init__with_explicit_model(mock_model):
    agent = Agent(model=mock_model)

    assert agent.model == mock_model


def test_agent__init__with_string_model_id():
    agent = Agent(model="nonsense")

    assert isinstance(agent.model, BedrockModel)
    assert agent.model.config["model_id"] == "nonsense"


def test_agent__call__(
    mock_model,
    system_prompt,
    callback_handler,
    agent,
    tool,
    alist,
):
    conversation_manager_spy = unittest.mock.Mock(wraps=agent.conversation_manager)
    agent.conversation_manager = conversation_manager_spy

    mock_model.mock_converse.side_effect = [
        alist(
            [
                {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": tool.tool_spec["name"],
                            },
                        },
                    },
                },
                {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"random_string": "abcdEfghI123"}'}}}},
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        ),
        alist(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    result = agent("test message")

    tru_result = {
        "message": result.message,
        "state": result.state,
        "stop_reason": result.stop_reason,
    }
    exp_result = {
        "message": {"content": [{"text": "test text"}], "role": "assistant"},
        "state": {},
        "stop_reason": "end_turn",
    }

    assert tru_result == exp_result

    mock_model.mock_converse.assert_has_calls(
        [
            unittest.mock.call(
                [
                    {
                        "role": "user",
                        "content": [
                            {"text": "test message"},
                        ],
                    },
                ],
                [tool.tool_spec],
                system_prompt,
            ),
            unittest.mock.call(
                [
                    {
                        "role": "user",
                        "content": [
                            {"text": "test message"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "t1",
                                    "name": tool.tool_spec["name"],
                                    "input": {"random_string": "abcdEfghI123"},
                                },
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "t1",
                                    "status": "success",
                                    "content": [{"text": "abcdEfghI123"}],
                                },
                            },
                        ],
                    },
                ],
                [tool.tool_spec],
                system_prompt,
            ),
        ],
    )

    callback_handler.assert_called()
    conversation_manager_spy.apply_management.assert_called_with(agent)


def test_agent__call__passes_kwargs(mock_model, agent, tool, mock_event_loop_cycle, alist):
    mock_model.mock_converse.side_effect = [
        alist(
            [
                {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": tool.tool_spec["name"],
                            },
                        },
                    },
                },
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        ),
    ]

    override_system_prompt = "Override system prompt"
    override_model = unittest.mock.Mock()
    override_tool_execution_handler = unittest.mock.Mock()
    override_event_loop_metrics = unittest.mock.Mock()
    override_callback_handler = unittest.mock.Mock()
    override_tool_handler = unittest.mock.Mock()
    override_messages = [{"role": "user", "content": [{"text": "override msg"}]}]
    override_tool_config = {"test": "config"}

    async def check_kwargs(**kwargs):
        kwargs_kwargs = kwargs["kwargs"]
        assert kwargs_kwargs["some_value"] == "a_value"
        assert kwargs_kwargs["system_prompt"] == override_system_prompt
        assert kwargs_kwargs["model"] == override_model
        assert kwargs_kwargs["tool_execution_handler"] == override_tool_execution_handler
        assert kwargs_kwargs["event_loop_metrics"] == override_event_loop_metrics
        assert kwargs_kwargs["callback_handler"] == override_callback_handler
        assert kwargs_kwargs["tool_handler"] == override_tool_handler
        assert kwargs_kwargs["messages"] == override_messages
        assert kwargs_kwargs["tool_config"] == override_tool_config
        assert kwargs_kwargs["agent"] == agent

        # Return expected values from event_loop_cycle
        yield {"stop": ("stop", {"role": "assistant", "content": [{"text": "Response"}]}, {}, {})}

    mock_event_loop_cycle.side_effect = check_kwargs

    agent(
        "test message",
        some_value="a_value",
        system_prompt=override_system_prompt,
        model=override_model,
        tool_execution_handler=override_tool_execution_handler,
        event_loop_metrics=override_event_loop_metrics,
        callback_handler=override_callback_handler,
        tool_handler=override_tool_handler,
        messages=override_messages,
        tool_config=override_tool_config,
    )

    mock_event_loop_cycle.assert_called_once()


def test_agent__call__retry_with_reduced_context(mock_model, agent, tool, alist):
    conversation_manager_spy = unittest.mock.Mock(wraps=agent.conversation_manager)
    agent.conversation_manager = conversation_manager_spy

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello!"}]},
        {
            "role": "assistant",
            "content": [{"text": "Hi!"}],
        },
        {"role": "user", "content": [{"text": "Whats your favorite color?"}]},
        {
            "role": "assistant",
            "content": [{"text": "Blue!"}],
        },
    ]
    agent.messages = messages

    mock_model.mock_converse.side_effect = [
        ContextWindowOverflowException(RuntimeError("Input is too long for requested model")),
        alist(
            [
                {
                    "contentBlockStart": {"start": {}},
                },
                {"contentBlockDelta": {"delta": {"text": "Green!"}}},
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        ),
    ]

    agent("And now?")

    expected_messages = [
        {"role": "user", "content": [{"text": "Whats your favorite color?"}]},
        {
            "role": "assistant",
            "content": [{"text": "Blue!"}],
        },
        {
            "role": "user",
            "content": [
                {"text": "And now?"},
            ],
        },
    ]

    mock_model.mock_converse.assert_called_with(
        expected_messages,
        unittest.mock.ANY,
        unittest.mock.ANY,
    )

    conversation_manager_spy.reduce_context.assert_called_once()
    assert conversation_manager_spy.apply_management.call_count == 1


def test_agent__call__always_sliding_window_conversation_manager_doesnt_infinite_loop(mock_model, agent, tool):
    conversation_manager = SlidingWindowConversationManager(window_size=500, should_truncate_results=False)
    conversation_manager_spy = unittest.mock.Mock(wraps=conversation_manager)
    agent.conversation_manager = conversation_manager_spy

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello!"}]},
        {
            "role": "assistant",
            "content": [{"text": "Hi!"}],
        },
        {"role": "user", "content": [{"text": "Whats your favorite color?"}]},
    ] * 1000
    agent.messages = messages

    mock_model.mock_converse.side_effect = ContextWindowOverflowException(
        RuntimeError("Input is too long for requested model")
    )

    with pytest.raises(ContextWindowOverflowException):
        agent("Test!")

    assert conversation_manager_spy.reduce_context.call_count > 0
    assert conversation_manager_spy.apply_management.call_count == 1


def test_agent__call__null_conversation_window_manager__doesnt_infinite_loop(mock_model, agent, tool):
    agent.conversation_manager = NullConversationManager()

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello!"}]},
        {
            "role": "assistant",
            "content": [{"text": "Hi!"}],
        },
        {"role": "user", "content": [{"text": "Whats your favorite color?"}]},
    ] * 1000
    agent.messages = messages

    mock_model.mock_converse.side_effect = ContextWindowOverflowException(
        RuntimeError("Input is too long for requested model")
    )

    with pytest.raises(ContextWindowOverflowException):
        agent("Test!")


def test_agent__call__tool_truncation_doesnt_infinite_loop(mock_model, agent):
    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello!"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "123", "input": {"hello": "world"}, "name": "test"}}],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "123", "content": [{"text": "Some large input!"}], "status": "success"}}
            ],
        },
    ]
    agent.messages = messages

    mock_model.mock_converse.side_effect = ContextWindowOverflowException(
        RuntimeError("Input is too long for requested model")
    )

    with pytest.raises(ContextWindowOverflowException):
        agent("Test!")


def test_agent__call__retry_with_overwritten_tool(mock_model, agent, tool, alist):
    conversation_manager_spy = unittest.mock.Mock(wraps=agent.conversation_manager)
    agent.conversation_manager = conversation_manager_spy

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello!"}]},
        {
            "role": "assistant",
            "content": [{"text": "Hi!"}],
        },
    ]
    agent.messages = messages

    mock_model.mock_converse.side_effect = [
        alist(
            [
                {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": tool.tool_spec["name"],
                            },
                        },
                    },
                },
                {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"random_string": "abcdEfghI123"}'}}}},
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        ),
        # Will truncate the tool result
        ContextWindowOverflowException(RuntimeError("Input is too long for requested model")),
        # Will reduce the context
        ContextWindowOverflowException(RuntimeError("Input is too long for requested model")),
        alist([]),
    ]

    agent("test message")

    expected_messages = [
        {"role": "user", "content": [{"text": "test message"}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "t1", "name": "tool_decorated", "input": {"random_string": "abcdEfghI123"}}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "status": "error",
                        "content": [{"text": "The tool result was too large!"}],
                    }
                }
            ],
        },
    ]

    mock_model.mock_converse.assert_called_with(
        expected_messages,
        unittest.mock.ANY,
        unittest.mock.ANY,
    )

    assert conversation_manager_spy.reduce_context.call_count == 2
    assert conversation_manager_spy.apply_management.call_count == 1


def test_agent__call__invalid_tool_use_event_loop_exception(mock_model, agent, tool, alist):
    mock_model.mock_converse.side_effect = [
        alist(
            [
                {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": tool.tool_spec["name"],
                            },
                        },
                    },
                },
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        ),
        RuntimeError,
    ]

    with pytest.raises(EventLoopException):
        agent("test message")


def test_agent__call__callback(mock_model, agent, callback_handler, alist):
    mock_model.mock_converse.return_value = alist(
        [
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "test"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"value"}'}}}},
            {"contentBlockStop": {}},
            {"contentBlockStart": {"start": {}}},
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "value"}}}},
            {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "value"}}}},
            {"contentBlockStop": {}},
            {"contentBlockStart": {"start": {}}},
            {"contentBlockDelta": {"delta": {"text": "value"}}},
            {"contentBlockStop": {}},
        ]
    )

    agent("test")

    callback_handler.assert_has_calls(
        [
            unittest.mock.call(init_event_loop=True),
            unittest.mock.call(start=True),
            unittest.mock.call(start_event_loop=True),
            unittest.mock.call(
                event={"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "test"}}}}
            ),
            unittest.mock.call(event={"contentBlockDelta": {"delta": {"toolUse": {"input": '{"value"}'}}}}),
            unittest.mock.call(
                agent=agent,
                current_tool_use={"toolUseId": "123", "name": "test", "input": {}},
                delta={"toolUse": {"input": '{"value"}'}},
                event_loop_cycle_id=unittest.mock.ANY,
                event_loop_cycle_span=unittest.mock.ANY,
                event_loop_cycle_trace=unittest.mock.ANY,
                request_state={},
            ),
            unittest.mock.call(event={"contentBlockStop": {}}),
            unittest.mock.call(event={"contentBlockStart": {"start": {}}}),
            unittest.mock.call(event={"contentBlockDelta": {"delta": {"reasoningContent": {"text": "value"}}}}),
            unittest.mock.call(
                agent=agent,
                delta={"reasoningContent": {"text": "value"}},
                event_loop_cycle_id=unittest.mock.ANY,
                event_loop_cycle_span=unittest.mock.ANY,
                event_loop_cycle_trace=unittest.mock.ANY,
                reasoning=True,
                reasoningText="value",
                request_state={},
            ),
            unittest.mock.call(event={"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "value"}}}}),
            unittest.mock.call(
                agent=agent,
                delta={"reasoningContent": {"signature": "value"}},
                event_loop_cycle_id=unittest.mock.ANY,
                event_loop_cycle_span=unittest.mock.ANY,
                event_loop_cycle_trace=unittest.mock.ANY,
                reasoning=True,
                reasoning_signature="value",
                request_state={},
            ),
            unittest.mock.call(event={"contentBlockStop": {}}),
            unittest.mock.call(event={"contentBlockStart": {"start": {}}}),
            unittest.mock.call(event={"contentBlockDelta": {"delta": {"text": "value"}}}),
            unittest.mock.call(
                agent=agent,
                data="value",
                delta={"text": "value"},
                event_loop_cycle_id=unittest.mock.ANY,
                event_loop_cycle_span=unittest.mock.ANY,
                event_loop_cycle_trace=unittest.mock.ANY,
                request_state={},
            ),
            unittest.mock.call(event={"contentBlockStop": {}}),
            unittest.mock.call(
                message={
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "123", "name": "test", "input": {}}},
                        {"reasoningContent": {"reasoningText": {"text": "value", "signature": "value"}}},
                        {"text": "value"},
                    ],
                },
            ),
        ],
    )


def test_agent_tool(mock_randint, agent):
    conversation_manager_spy = unittest.mock.Mock(wraps=agent.conversation_manager)
    agent.conversation_manager = conversation_manager_spy

    mock_randint.return_value = 1

    tru_result = agent.tool.tool_decorated(random_string="abcdEfghI123")
    exp_result = {
        "content": [
            {
                "text": "abcdEfghI123",
            },
        ],
        "status": "success",
        "toolUseId": "tooluse_tool_decorated_1",
    }

    assert tru_result == exp_result
    conversation_manager_spy.apply_management.assert_called_with(agent)


def test_agent_tool_user_message_override(agent):
    agent.tool.tool_decorated(random_string="abcdEfghI123", user_message_override="test override")

    tru_message = agent.messages[0]
    exp_message = {
        "content": [
            {
                "text": "test override\n",
            },
            {
                "text": (
                    'agent.tool.tool_decorated direct tool call.\nInput parameters: {"random_string": "abcdEfghI123"}\n'
                ),
            },
        ],
        "role": "user",
    }

    assert tru_message == exp_message


def test_agent_tool_do_not_record_tool(agent):
    agent.record_direct_tool_call = False
    agent.tool.tool_decorated(random_string="abcdEfghI123", user_message_override="test override")

    tru_messages = agent.messages
    exp_messages = []

    assert tru_messages == exp_messages


def test_agent_tool_do_not_record_tool_with_method_override(agent):
    agent.record_direct_tool_call = True
    agent.tool.tool_decorated(
        random_string="abcdEfghI123", user_message_override="test override", record_direct_tool_call=False
    )

    tru_messages = agent.messages
    exp_messages = []

    assert tru_messages == exp_messages


def test_agent_tool_tool_does_not_exist(agent):
    with pytest.raises(AttributeError):
        agent.tool.does_not_exist()


@pytest.mark.parametrize("tools", [None, [tool_decorated]], indirect=True)
def test_agent_tool_names(tools, agent):
    actual = agent.tool_names
    expected = list(agent.tool_registry.get_all_tools_config().keys())

    assert actual == expected


def test_agent__del__(agent):
    del agent


def test_agent_init_with_no_model_or_model_id():
    agent = Agent()
    assert agent.model is not None
    assert agent.model.get_config().get("model_id") == DEFAULT_BEDROCK_MODEL_ID


def test_agent_tool_no_parameter_conflict(agent, tool_registry, mock_randint):
    agent.tool_handler = unittest.mock.Mock()

    @strands.tools.tool(name="system_prompter")
    def function(system_prompt: str) -> str:
        return system_prompt

    agent.tool_registry.register_tool(function)

    mock_randint.return_value = 1

    agent.tool.system_prompter(system_prompt="tool prompt")

    agent.tool_handler.process.assert_called_with(
        tool={
            "toolUseId": "tooluse_system_prompter_1",
            "name": "system_prompter",
            "input": {"system_prompt": "tool prompt"},
        },
        model=unittest.mock.ANY,
        system_prompt="You are a helpful assistant.",
        messages=unittest.mock.ANY,
        tool_config=unittest.mock.ANY,
        callback_handler=unittest.mock.ANY,
        kwargs={"system_prompt": "tool prompt"},
    )


def test_agent_tool_with_name_normalization(agent, tool_registry, mock_randint):
    agent.tool_handler = unittest.mock.Mock()

    tool_name = "system-prompter"

    @strands.tools.tool(name=tool_name)
    def function(system_prompt: str) -> str:
        return system_prompt

    tool = strands.tools.tools.FunctionTool(function)
    agent.tool_registry.register_tool(tool)

    mock_randint.return_value = 1

    agent.tool.system_prompter(system_prompt="tool prompt")

    # Verify the correct tool was invoked
    assert agent.tool_handler.process.call_count == 1
    tool_call = agent.tool_handler.process.call_args.kwargs.get("tool")

    assert tool_call == {
        # Note that the tool-use uses the "python safe" name
        "toolUseId": "tooluse_system_prompter_1",
        # But the name of the tool is the one in the registry
        "name": tool_name,
        "input": {"system_prompt": "tool prompt"},
    }


def test_agent_tool_with_no_normalized_match(agent, tool_registry, mock_randint):
    agent.tool_handler = unittest.mock.Mock()

    mock_randint.return_value = 1

    with pytest.raises(AttributeError) as err:
        agent.tool.system_prompter_1(system_prompt="tool prompt")

    assert str(err.value) == "Tool 'system_prompter_1' not found"


def test_agent_with_none_callback_handler_prints_nothing():
    agent = Agent()

    assert isinstance(agent.callback_handler, PrintingCallbackHandler)


def test_agent_with_callback_handler_none_uses_null_handler():
    agent = Agent(callback_handler=None)

    assert agent.callback_handler == null_callback_handler


def test_agent_callback_handler_not_provided_creates_new_instances():
    """Test that when callback_handler is not provided, new PrintingCallbackHandler instances are created."""
    # Create two agents without providing callback_handler
    agent1 = Agent()
    agent2 = Agent()

    # Both should have PrintingCallbackHandler instances
    assert isinstance(agent1.callback_handler, PrintingCallbackHandler)
    assert isinstance(agent2.callback_handler, PrintingCallbackHandler)

    # But they should be different object instances
    assert agent1.callback_handler is not agent2.callback_handler


def test_agent_callback_handler_explicit_none_uses_null_handler():
    """Test that when callback_handler is explicitly set to None, null_callback_handler is used."""
    agent = Agent(callback_handler=None)

    # Should use null_callback_handler
    assert agent.callback_handler is null_callback_handler


def test_agent_callback_handler_custom_handler_used():
    """Test that when a custom callback_handler is provided, it is used."""
    custom_handler = unittest.mock.Mock()
    agent = Agent(callback_handler=custom_handler)

    # Should use the provided custom handler
    assert agent.callback_handler is custom_handler


# mock the User(name='Jane Doe', age=30, email='jane@doe.com')
class User(BaseModel):
    """A user of the system."""

    name: str
    age: int
    email: str


def test_agent_method_structured_output(agent, alist):
    # Mock the structured_output method on the model
    expected_user = User(name="Jane Doe", age=30, email="jane@doe.com")
    agent.model.structured_output = unittest.mock.Mock(return_value=alist([{"output": expected_user}]))

    prompt = "Jane Doe is 30 years old and her email is jane@doe.com"

    result = agent.structured_output(User, prompt)
    assert result == expected_user

    # Verify the model's structured_output was called with correct arguments
    agent.model.structured_output.assert_called_once_with(User, [{"role": "user", "content": [{"text": prompt}]}])


@pytest.mark.asyncio
async def test_stream_async_returns_all_events(mock_event_loop_cycle):
    agent = Agent()

    # Define the side effect to simulate callback handler being called multiple times
    async def test_event_loop(*args, **kwargs):
        yield {"callback": {"data": "First chunk"}}
        yield {"callback": {"data": "Second chunk"}}
        yield {"callback": {"data": "Final chunk", "complete": True}}

        # Return expected values from event_loop_cycle
        yield {"stop": ("stop", {"role": "assistant", "content": [{"text": "Response"}]}, {}, {})}

    mock_event_loop_cycle.side_effect = test_event_loop
    mock_callback = unittest.mock.Mock()

    iterator = agent.stream_async("test message", callback_handler=mock_callback)

    tru_events = [e async for e in iterator]
    exp_events = [
        {"init_event_loop": True, "callback_handler": mock_callback},
        {"data": "First chunk"},
        {"data": "Second chunk"},
        {"complete": True, "data": "Final chunk"},
    ]
    assert tru_events == exp_events

    exp_calls = [unittest.mock.call(**event) for event in exp_events]
    mock_callback.assert_has_calls(exp_calls)


@pytest.mark.asyncio
async def test_stream_async_passes_kwargs(agent, mock_model, mock_event_loop_cycle, alist):
    mock_model.mock_converse.side_effect = [
        alist(
            [
                {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "a_tool",
                            },
                        },
                    },
                },
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        ),
    ]

    async def check_kwargs(**kwargs):
        kwargs_kwargs = kwargs["kwargs"]
        assert kwargs_kwargs["some_value"] == "a_value"
        # Return expected values from event_loop_cycle
        yield {"stop": ("stop", {"role": "assistant", "content": [{"text": "Response"}]}, {}, {})}

    mock_event_loop_cycle.side_effect = check_kwargs

    iterator = agent.stream_async("test message", some_value="a_value")
    actual_events = [e async for e in iterator]

    assert actual_events == [{"init_event_loop": True, "some_value": "a_value"}]
    assert mock_event_loop_cycle.call_count == 1


@pytest.mark.asyncio
async def test_stream_async_raises_exceptions(mock_event_loop_cycle):
    mock_event_loop_cycle.side_effect = ValueError("Test exception")

    agent = Agent()
    iterator = agent.stream_async("test message")

    await anext(iterator)
    with pytest.raises(ValueError, match="Test exception"):
        await anext(iterator)


def test_agent_init_with_trace_attributes():
    """Test that trace attributes are properly initialized in the Agent."""
    # Test with valid trace attributes
    valid_attributes = {
        "string_attr": "value",
        "int_attr": 123,
        "float_attr": 45.6,
        "bool_attr": True,
        "list_attr": ["item1", "item2"],
    }

    agent = Agent(trace_attributes=valid_attributes)

    # Check that all valid attributes were copied
    assert agent.trace_attributes == valid_attributes

    # Test with mixed valid and invalid trace attributes
    mixed_attributes = {
        "valid_str": "value",
        "invalid_dict": {"key": "value"},  # Should be filtered out
        "invalid_set": {1, 2, 3},  # Should be filtered out
        "valid_list": [1, 2, 3],  # Should be kept
        "invalid_nested_list": [1, {"nested": "dict"}],  # Should be filtered out
    }

    agent = Agent(trace_attributes=mixed_attributes)

    # Check that only valid attributes were copied
    assert "valid_str" in agent.trace_attributes
    assert "valid_list" in agent.trace_attributes
    assert "invalid_dict" not in agent.trace_attributes
    assert "invalid_set" not in agent.trace_attributes
    assert "invalid_nested_list" not in agent.trace_attributes


@unittest.mock.patch("strands.agent.agent.get_tracer")
def test_agent_init_initializes_tracer(mock_get_tracer):
    """Test that the tracer is initialized when creating an Agent."""
    mock_tracer = unittest.mock.MagicMock()
    mock_get_tracer.return_value = mock_tracer

    agent = Agent()

    # Verify tracer was initialized
    mock_get_tracer.assert_called_once()
    assert agent.tracer == mock_tracer
    assert agent.trace_span is None


@unittest.mock.patch("strands.agent.agent.get_tracer")
def test_agent_call_creates_and_ends_span_on_success(mock_get_tracer, mock_model, alist):
    """Test that __call__ creates and ends a span when the call succeeds."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_agent_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Setup mock model response
    mock_model.mock_converse.side_effect = [
        alist(
            [
                {"contentBlockDelta": {"delta": {"text": "test response"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    # Create agent and make a call
    agent = Agent(model=mock_model)
    result = agent("test prompt")

    # Verify span was created
    mock_tracer.start_agent_span.assert_called_once_with(
        prompt="test prompt",
        model_id=unittest.mock.ANY,
        tools=agent.tool_names,
        system_prompt=agent.system_prompt,
        custom_trace_attributes=agent.trace_attributes,
    )

    # Verify span was ended with the result
    mock_tracer.end_agent_span.assert_called_once_with(span=mock_span, response=result)


@pytest.mark.asyncio
@unittest.mock.patch("strands.agent.agent.get_tracer")
async def test_agent_stream_async_creates_and_ends_span_on_success(mock_get_tracer, mock_event_loop_cycle):
    """Test that stream_async creates and ends a span when the call succeeds."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_agent_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Define the side effect to simulate callback handler being called multiple times
    async def call_callback_handler(*args, **kwargs):
        # Extract the callback handler from kwargs
        callback_handler = kwargs.get("callback_handler")
        # Call the callback handler with different data values
        callback_handler(data="First chunk")
        callback_handler(data="Second chunk")
        callback_handler(data="Final chunk", complete=True)
        # Return expected values from event_loop_cycle
        yield {"stop": ("stop", {"role": "assistant", "content": [{"text": "Agent Response"}]}, {}, {})}

    mock_event_loop_cycle.side_effect = call_callback_handler

    # Create agent and make a call
    agent = Agent(model=mock_model)
    iterator = agent.stream_async("test prompt")
    async for _event in iterator:
        pass  # NoOp

    # Verify span was created
    mock_tracer.start_agent_span.assert_called_once_with(
        prompt="test prompt",
        model_id=unittest.mock.ANY,
        tools=agent.tool_names,
        system_prompt=agent.system_prompt,
        custom_trace_attributes=agent.trace_attributes,
    )

    expected_response = AgentResult(
        stop_reason="stop", message={"role": "assistant", "content": [{"text": "Agent Response"}]}, metrics={}, state={}
    )

    # Verify span was ended with the result
    mock_tracer.end_agent_span.assert_called_once_with(span=mock_span, response=expected_response)


@unittest.mock.patch("strands.agent.agent.get_tracer")
def test_agent_call_creates_and_ends_span_on_exception(mock_get_tracer, mock_model):
    """Test that __call__ creates and ends a span when an exception occurs."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_agent_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Setup mock model to raise an exception
    test_exception = ValueError("Test exception")
    mock_model.mock_converse.side_effect = test_exception

    # Create agent and make a call that will raise an exception
    agent = Agent(model=mock_model)

    # Call the agent and catch the exception
    with pytest.raises(ValueError):
        agent("test prompt")

    # Verify span was created
    mock_tracer.start_agent_span.assert_called_once_with(
        prompt="test prompt",
        model_id=unittest.mock.ANY,
        tools=agent.tool_names,
        system_prompt=agent.system_prompt,
        custom_trace_attributes=agent.trace_attributes,
    )

    # Verify span was ended with the exception
    mock_tracer.end_agent_span.assert_called_once_with(span=mock_span, error=test_exception)


@pytest.mark.asyncio
@unittest.mock.patch("strands.agent.agent.get_tracer")
async def test_agent_stream_async_creates_and_ends_span_on_exception(mock_get_tracer, mock_model):
    """Test that stream_async creates and ends a span when the call succeeds."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_agent_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Define the side effect to simulate callback handler raising an Exception
    test_exception = ValueError("Test exception")
    mock_model.mock_converse.side_effect = test_exception

    # Create agent and make a call
    agent = Agent(model=mock_model)

    # Call the agent and catch the exception
    with pytest.raises(ValueError):
        iterator = agent.stream_async("test prompt")
        async for _event in iterator:
            pass  # NoOp

    # Verify span was created
    mock_tracer.start_agent_span.assert_called_once_with(
        prompt="test prompt",
        model_id=unittest.mock.ANY,
        tools=agent.tool_names,
        system_prompt=agent.system_prompt,
        custom_trace_attributes=agent.trace_attributes,
    )

    # Verify span was ended with the exception
    mock_tracer.end_agent_span.assert_called_once_with(span=mock_span, error=test_exception)


@unittest.mock.patch("strands.agent.agent.get_tracer")
def test_event_loop_cycle_includes_parent_span(mock_get_tracer, mock_event_loop_cycle, mock_model, alist):
    """Test that event_loop_cycle is called with the parent span."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_agent_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Setup mock for event_loop_cycle
    mock_event_loop_cycle.return_value = alist(
        [{"stop": ("stop", {"role": "assistant", "content": [{"text": "Response"}]}, {}, {})}]
    )

    # Create agent and make a call
    agent = Agent(model=mock_model)
    agent("test prompt")

    # Verify event_loop_cycle was called with the span
    mock_event_loop_cycle.assert_called_once()
    kwargs = mock_event_loop_cycle.call_args[1]
    assert "event_loop_parent_span" in kwargs
    assert kwargs["event_loop_parent_span"] == mock_span
