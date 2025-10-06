import copy
import importlib
import json
import os
import textwrap
import unittest.mock
from uuid import uuid4

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.agent import AgentResult
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.agent.state import AgentState
from strands.handlers.callback_handler import PrintingCallbackHandler, null_callback_handler
from strands.models.bedrock import DEFAULT_BEDROCK_MODEL_ID, BedrockModel
from strands.session.repository_session_manager import RepositorySessionManager
from strands.telemetry.tracer import serialize
from strands.types._events import EventLoopStopEvent, ModelStreamEvent
from strands.types.content import Messages
from strands.types.exceptions import ContextWindowOverflowException, EventLoopException
from strands.types.session import Session, SessionAgent, SessionMessage, SessionType
from tests.fixtures.mock_session_repository import MockedSessionRepository
from tests.fixtures.mocked_model_provider import MockedModelProvider

# For unit testing we will use the the us inference
FORMATTED_DEFAULT_MODEL_ID = DEFAULT_BEDROCK_MODEL_ID.format("us")


@pytest.fixture
def mock_randint():
    with unittest.mock.patch.object(strands.agent.agent.random, "randint") as mock:
        yield mock


@pytest.fixture
def mock_model(request):
    async def stream(*args, **kwargs):
        result = mock.mock_stream(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        # If result is already an async generator, yield from it
        if hasattr(result, "__aiter__"):
            async for item in result:
                yield item
        else:
            # If result is a regular generator or iterable, convert to async
            for item in result:
                yield item

    mock = unittest.mock.Mock(spec=getattr(request, "param", None))
    mock.configure_mock(mock_stream=unittest.mock.MagicMock())
    mock.stream.side_effect = stream

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
    tool_registry.register_tool(tool_decorated)
    return tool_decorated


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
        agent.tool_registry.register_tool(tool_decorated)

    return agent


@pytest.fixture
def user():
    class User(BaseModel):
        name: str
        age: int
        email: str

    return User(name="Jane Doe", age=30, email="jane@doe.com")


def test_agent__init__tool_loader_format(tool_decorated, tool_module, tool_imported, tool_registry):
    _ = tool_registry

    agent = Agent(tools=[tool_decorated, tool_module, tool_imported])

    tru_tool_names = sorted(tool_spec["name"] for tool_spec in agent.tool_registry.get_all_tool_specs())
    exp_tool_names = ["tool_decorated", "tool_imported", "tool_module"]

    assert tru_tool_names == exp_tool_names


def test_agent__init__tool_loader_dict(tool_module, tool_registry):
    _ = tool_registry

    agent = Agent(tools=[{"name": "tool_module", "path": tool_module}])

    tru_tool_names = sorted(tool_spec["name"] for tool_spec in agent.tool_registry.get_all_tool_specs())
    exp_tool_names = ["tool_module"]

    assert tru_tool_names == exp_tool_names


def test_agent__init__with_default_model():
    agent = Agent()

    assert isinstance(agent.model, BedrockModel)
    assert agent.model.config["model_id"] == FORMATTED_DEFAULT_MODEL_ID


def test_agent__init__with_explicit_model(mock_model):
    agent = Agent(model=mock_model)

    assert agent.model == mock_model


def test_agent__init__with_string_model_id():
    agent = Agent(model="nonsense")

    assert isinstance(agent.model, BedrockModel)
    assert agent.model.config["model_id"] == "nonsense"


def test_agent__init__nested_tools_flattening(tool_decorated, tool_module, tool_imported, tool_registry):
    _ = tool_registry
    # Nested structure: [tool_decorated, [tool_module, [tool_imported]]]
    agent = Agent(tools=[tool_decorated, [tool_module, [tool_imported]]])
    tru_tool_names = sorted(agent.tool_names)
    exp_tool_names = ["tool_decorated", "tool_imported", "tool_module"]
    assert tru_tool_names == exp_tool_names


def test_agent__init__deeply_nested_tools(tool_decorated, tool_module, tool_imported, tool_registry):
    _ = tool_registry
    # Deeply nested structure
    nested_tools = [[[[tool_decorated]], [[tool_module]], tool_imported]]
    agent = Agent(tools=nested_tools)
    tru_tool_names = sorted(agent.tool_names)
    exp_tool_names = ["tool_decorated", "tool_imported", "tool_module"]
    assert tru_tool_names == exp_tool_names


@pytest.mark.parametrize(
    "agent_id",
    [
        "a/../b",
        "a/b",
    ],
)
def test_agent__init__invalid_id(agent_id):
    with pytest.raises(ValueError, match=f"agent_id={agent_id} | id cannot contain path separators"):
        Agent(agent_id=agent_id)


def test_agent__call__(
    mock_model,
    system_prompt,
    callback_handler,
    agent,
    tool,
    agenerator,
):
    conversation_manager_spy = unittest.mock.Mock(wraps=agent.conversation_manager)
    agent.conversation_manager = conversation_manager_spy

    mock_model.mock_stream.side_effect = [
        agenerator(
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
        agenerator(
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

    mock_model.mock_stream.assert_has_calls(
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


def test_agent__call__passes_invocation_state(mock_model, agent, tool, mock_event_loop_cycle, agenerator):
    mock_model.mock_stream.side_effect = [
        agenerator(
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
    override_event_loop_metrics = unittest.mock.Mock()
    override_callback_handler = unittest.mock.Mock()
    override_tool_handler = unittest.mock.Mock()
    override_messages = [{"role": "user", "content": [{"text": "override msg"}]}]
    override_tool_config = {"test": "config"}

    async def check_invocation_state(**kwargs):
        invocation_state = kwargs["invocation_state"]
        assert invocation_state["some_value"] == "a_value"
        assert invocation_state["system_prompt"] == override_system_prompt
        assert invocation_state["model"] == override_model
        assert invocation_state["event_loop_metrics"] == override_event_loop_metrics
        assert invocation_state["callback_handler"] == override_callback_handler
        assert invocation_state["tool_handler"] == override_tool_handler
        assert invocation_state["messages"] == override_messages
        assert invocation_state["tool_config"] == override_tool_config
        assert invocation_state["agent"] == agent

        # Return expected values from event_loop_cycle
        yield EventLoopStopEvent("stop", {"role": "assistant", "content": [{"text": "Response"}]}, {}, {})

    mock_event_loop_cycle.side_effect = check_invocation_state

    agent(
        "test message",
        some_value="a_value",
        system_prompt=override_system_prompt,
        model=override_model,
        event_loop_metrics=override_event_loop_metrics,
        callback_handler=override_callback_handler,
        tool_handler=override_tool_handler,
        messages=override_messages,
        tool_config=override_tool_config,
    )

    mock_event_loop_cycle.assert_called_once()


def test_agent__call__retry_with_reduced_context(mock_model, agent, tool, agenerator):
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

    mock_model.mock_stream.side_effect = [
        ContextWindowOverflowException(RuntimeError("Input is too long for requested model")),
        agenerator(
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

    mock_model.mock_stream.assert_called_with(
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

    mock_model.mock_stream.side_effect = ContextWindowOverflowException(
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

    mock_model.mock_stream.side_effect = ContextWindowOverflowException(
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

    mock_model.mock_stream.side_effect = ContextWindowOverflowException(
        RuntimeError("Input is too long for requested model")
    )

    with pytest.raises(ContextWindowOverflowException):
        agent("Test!")


def test_agent__call__retry_with_overwritten_tool(mock_model, agent, tool, agenerator):
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

    mock_model.mock_stream.side_effect = [
        agenerator(
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
        agenerator([]),
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

    mock_model.mock_stream.assert_called_with(
        expected_messages,
        unittest.mock.ANY,
        unittest.mock.ANY,
    )

    assert conversation_manager_spy.reduce_context.call_count == 2
    assert conversation_manager_spy.apply_management.call_count == 1


def test_agent__call__invalid_tool_use_event_loop_exception(mock_model, agent, tool, agenerator):
    mock_model.mock_stream.side_effect = [
        agenerator(
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


def test_agent__call__callback(mock_model, agent, callback_handler, agenerator):
    mock_model.mock_stream.return_value = agenerator(
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
    assert callback_handler.call_args_list == [
        unittest.mock.call(init_event_loop=True),
        unittest.mock.call(start=True),
        unittest.mock.call(start_event_loop=True),
        unittest.mock.call(event={"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "test"}}}}),
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
        unittest.mock.call(
            result=AgentResult(
                stop_reason="end_turn",
                message={
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "123", "name": "test", "input": {}}},
                        {"reasoningContent": {"reasoningText": {"text": "value", "signature": "value"}}},
                        {"text": "value"},
                    ],
                },
                metrics=unittest.mock.ANY,
                state={},
            )
        ),
    ]


@pytest.mark.asyncio
async def test_agent__call__in_async_context(mock_model, agent, agenerator):
    mock_model.mock_stream.return_value = agenerator(
        [
            {
                "contentBlockStart": {"start": {}},
            },
            {"contentBlockDelta": {"delta": {"text": "abc"}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )

    result = agent("test")

    tru_message = result.message
    exp_message = {"content": [{"text": "abc"}], "role": "assistant"}
    assert tru_message == exp_message


@pytest.mark.asyncio
async def test_agent_invoke_async(mock_model, agent, agenerator):
    mock_model.mock_stream.return_value = agenerator(
        [
            {
                "contentBlockStart": {"start": {}},
            },
            {"contentBlockDelta": {"delta": {"text": "abc"}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )

    result = await agent.invoke_async("test")

    tru_message = result.message
    exp_message = {"content": [{"text": "abc"}], "role": "assistant"}
    assert tru_message == exp_message


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


@pytest.mark.asyncio
async def test_agent_tool_in_async_context(mock_randint, agent):
    mock_randint.return_value = 123

    tru_result = agent.tool.tool_decorated(random_string="abcdEfghI123")
    exp_result = {
        "content": [
            {
                "text": "abcdEfghI123",
            },
        ],
        "status": "success",
        "toolUseId": "tooluse_tool_decorated_123",
    }

    assert tru_result == exp_result


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
    assert agent.model.get_config().get("model_id") == FORMATTED_DEFAULT_MODEL_ID


def test_agent_tool_no_parameter_conflict(agent, tool_registry, mock_randint, agenerator):
    @strands.tools.tool(name="system_prompter")
    def function(system_prompt: str) -> str:
        return system_prompt

    agent.tool_registry.register_tool(function)

    mock_randint.return_value = 1

    tru_result = agent.tool.system_prompter(system_prompt="tool prompt")
    exp_result = {"toolUseId": "tooluse_system_prompter_1", "status": "success", "content": [{"text": "tool prompt"}]}
    assert tru_result == exp_result


def test_agent_tool_with_name_normalization(agent, tool_registry, mock_randint, agenerator):
    tool_name = "system-prompter"

    @strands.tools.tool(name=tool_name)
    def function(system_prompt: str) -> str:
        return system_prompt

    agent.tool_registry.register_tool(function)

    mock_randint.return_value = 1

    tru_result = agent.tool.system_prompter(system_prompt="tool prompt")
    exp_result = {"toolUseId": "tooluse_system_prompter_1", "status": "success", "content": [{"text": "tool prompt"}]}
    assert tru_result == exp_result


def test_agent_tool_with_no_normalized_match(agent, tool_registry, mock_randint):
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


def test_agent_structured_output(agent, system_prompt, user, agenerator):
    # Setup mock tracer and span
    mock_strands_tracer = unittest.mock.MagicMock()
    mock_otel_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_strands_tracer.tracer = mock_otel_tracer
    mock_otel_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    agent.tracer = mock_strands_tracer

    agent.model.structured_output = unittest.mock.Mock(return_value=agenerator([{"output": user}]))

    prompt = "Jane Doe is 30 years old and her email is jane@doe.com"

    # Store initial message count
    initial_message_count = len(agent.messages)

    tru_result = agent.structured_output(type(user), prompt)
    exp_result = user
    assert tru_result == exp_result

    # Verify conversation history is not polluted
    assert len(agent.messages) == initial_message_count

    # Verify the model was called with temporary messages array
    agent.model.structured_output.assert_called_once_with(
        type(user), [{"role": "user", "content": [{"text": prompt}]}], system_prompt=system_prompt
    )

    mock_span.set_attributes.assert_called_once_with(
        {
            "gen_ai.system": "strands-agents",
            "gen_ai.agent.name": "Strands Agents",
            "gen_ai.agent.id": "default",
            "gen_ai.operation.name": "execute_structured_output",
        }
    )

    # ensure correct otel event messages are emitted
    act_event_names = mock_span.add_event.call_args_list
    exp_event_names = [
        unittest.mock.call(
            "gen_ai.system.message", attributes={"role": "system", "content": serialize([{"text": system_prompt}])}
        ),
        unittest.mock.call(
            "gen_ai.user.message",
            attributes={
                "role": "user",
                "content": '[{"text": "Jane Doe is 30 years old and her email is jane@doe.com"}]',
            },
        ),
        unittest.mock.call("gen_ai.choice", attributes={"message": json.dumps(user.model_dump())}),
    ]

    assert act_event_names == exp_event_names


def test_agent_structured_output_multi_modal_input(agent, system_prompt, user, agenerator):
    # Setup mock tracer and span
    mock_strands_tracer = unittest.mock.MagicMock()
    mock_otel_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_strands_tracer.tracer = mock_otel_tracer
    mock_otel_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    agent.tracer = mock_strands_tracer
    agent.model.structured_output = unittest.mock.Mock(return_value=agenerator([{"output": user}]))

    prompt = [
        {"text": "Please describe the user in this image"},
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": b"\x89PNG\r\n\x1a\n",
                },
            }
        },
    ]

    # Store initial message count
    initial_message_count = len(agent.messages)

    tru_result = agent.structured_output(type(user), prompt)
    exp_result = user
    assert tru_result == exp_result

    # Verify conversation history is not polluted
    assert len(agent.messages) == initial_message_count

    # Verify the model was called with temporary messages array
    agent.model.structured_output.assert_called_once_with(
        type(user), [{"role": "user", "content": prompt}], system_prompt=system_prompt
    )

    mock_span.add_event.assert_called_with(
        "gen_ai.choice",
        attributes={"message": json.dumps(user.model_dump())},
    )


@pytest.mark.asyncio
async def test_agent_structured_output_in_async_context(agent, user, agenerator):
    agent.model.structured_output = unittest.mock.Mock(return_value=agenerator([{"output": user}]))

    prompt = "Jane Doe is 30 years old and her email is jane@doe.com"

    # Store initial message count
    initial_message_count = len(agent.messages)

    tru_result = await agent.structured_output_async(type(user), prompt)
    exp_result = user
    assert tru_result == exp_result

    # Verify conversation history is not polluted
    assert len(agent.messages) == initial_message_count


def test_agent_structured_output_without_prompt(agent, system_prompt, user, agenerator):
    """Test that structured_output works with existing conversation history and no new prompt."""
    agent.model.structured_output = unittest.mock.Mock(return_value=agenerator([{"output": user}]))

    # Add some existing messages to the agent
    existing_messages = [
        {"role": "user", "content": [{"text": "Jane Doe is 30 years old"}]},
        {"role": "assistant", "content": [{"text": "I understand."}]},
    ]
    agent.messages.extend(existing_messages)

    initial_message_count = len(agent.messages)

    tru_result = agent.structured_output(type(user))  # No prompt provided
    exp_result = user
    assert tru_result == exp_result

    # Verify conversation history is unchanged
    assert len(agent.messages) == initial_message_count
    assert agent.messages == existing_messages

    # Verify the model was called with existing messages only
    agent.model.structured_output.assert_called_once_with(type(user), existing_messages, system_prompt=system_prompt)


@pytest.mark.asyncio
async def test_agent_structured_output_async(agent, system_prompt, user, agenerator):
    agent.model.structured_output = unittest.mock.Mock(return_value=agenerator([{"output": user}]))

    prompt = "Jane Doe is 30 years old and her email is jane@doe.com"

    # Store initial message count
    initial_message_count = len(agent.messages)

    tru_result = agent.structured_output(type(user), prompt)
    exp_result = user
    assert tru_result == exp_result

    # Verify conversation history is not polluted
    assert len(agent.messages) == initial_message_count

    # Verify the model was called with temporary messages array
    agent.model.structured_output.assert_called_once_with(
        type(user), [{"role": "user", "content": [{"text": prompt}]}], system_prompt=system_prompt
    )


@pytest.mark.asyncio
async def test_stream_async_returns_all_events(mock_event_loop_cycle, alist):
    agent = Agent()

    # Define the side effect to simulate callback handler being called multiple times
    async def test_event_loop(*args, **kwargs):
        yield ModelStreamEvent({"data": "First chunk"})
        yield ModelStreamEvent({"data": "Second chunk"})
        yield ModelStreamEvent({"data": "Final chunk", "complete": True})

        # Return expected values from event_loop_cycle
        yield EventLoopStopEvent("stop", {"role": "assistant", "content": [{"text": "Response"}]}, {}, {})

    mock_event_loop_cycle.side_effect = test_event_loop
    mock_callback = unittest.mock.Mock()

    stream = agent.stream_async("test message", callback_handler=mock_callback)

    tru_events = await alist(stream)
    exp_events = [
        {"init_event_loop": True, "callback_handler": mock_callback},
        {"data": "First chunk"},
        {"data": "Second chunk"},
        {"complete": True, "data": "Final chunk"},
        {
            "result": AgentResult(
                stop_reason="stop",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics={},
                state={},
            ),
        },
    ]
    assert tru_events == exp_events

    exp_calls = [unittest.mock.call(**event) for event in exp_events]
    mock_callback.assert_has_calls(exp_calls)


@pytest.mark.asyncio
async def test_stream_async_multi_modal_input(mock_model, agent, agenerator, alist):
    mock_model.mock_stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "I see text and an image"}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )

    prompt = [
        {"text": "This is a description of the image:"},
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": b"\x89PNG\r\n\x1a\n",
                },
            }
        },
    ]

    stream = agent.stream_async(prompt)
    await alist(stream)

    tru_message = agent.messages
    exp_message = [
        {"content": prompt, "role": "user"},
        {"content": [{"text": "I see text and an image"}], "role": "assistant"},
    ]
    assert tru_message == exp_message


@pytest.mark.asyncio
async def test_stream_async_passes_invocation_state(agent, mock_model, mock_event_loop_cycle, agenerator, alist):
    mock_model.mock_stream.side_effect = [
        agenerator(
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

    async def check_invocation_state(**kwargs):
        invocation_state = kwargs["invocation_state"]
        assert invocation_state["some_value"] == "a_value"
        # Return expected values from event_loop_cycle
        yield EventLoopStopEvent("stop", {"role": "assistant", "content": [{"text": "Response"}]}, {}, {})

    mock_event_loop_cycle.side_effect = check_invocation_state

    stream = agent.stream_async("test message", some_value="a_value")

    tru_events = await alist(stream)
    exp_events = [
        {"init_event_loop": True, "some_value": "a_value"},
        {
            "result": AgentResult(
                stop_reason="stop",
                message={"role": "assistant", "content": [{"text": "Response"}]},
                metrics={},
                state={},
            ),
        },
    ]
    assert tru_events == exp_events

    assert mock_event_loop_cycle.call_count == 1


@pytest.mark.asyncio
async def test_stream_async_raises_exceptions(mock_event_loop_cycle):
    mock_event_loop_cycle.side_effect = ValueError("Test exception")

    agent = Agent()
    stream = agent.stream_async("test message")

    await anext(stream)
    with pytest.raises(ValueError, match="Test exception"):
        await anext(stream)


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
def test_agent_call_creates_and_ends_span_on_success(mock_get_tracer, mock_model, agenerator):
    """Test that __call__ creates and ends a span when the call succeeds."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_agent_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Setup mock model response
    mock_model.mock_stream.side_effect = [
        agenerator(
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
        messages=[{"content": [{"text": "test prompt"}], "role": "user"}],
        agent_name="Strands Agents",
        model_id=unittest.mock.ANY,
        tools=agent.tool_names,
        system_prompt=agent.system_prompt,
        custom_trace_attributes=agent.trace_attributes,
    )

    # Verify span was ended with the result
    mock_tracer.end_agent_span.assert_called_once_with(span=mock_span, response=result)


@pytest.mark.asyncio
@unittest.mock.patch("strands.agent.agent.get_tracer")
async def test_agent_stream_async_creates_and_ends_span_on_success(mock_get_tracer, mock_event_loop_cycle, alist):
    """Test that stream_async creates and ends a span when the call succeeds."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_agent_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    async def test_event_loop(*args, **kwargs):
        yield EventLoopStopEvent("stop", {"role": "assistant", "content": [{"text": "Agent Response"}]}, {}, {})

    mock_event_loop_cycle.side_effect = test_event_loop

    # Create agent and make a call
    agent = Agent(model=mock_model)
    stream = agent.stream_async("test prompt")
    await alist(stream)

    # Verify span was created
    mock_tracer.start_agent_span.assert_called_once_with(
        messages=[{"content": [{"text": "test prompt"}], "role": "user"}],
        agent_name="Strands Agents",
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
    mock_model.mock_stream.side_effect = test_exception

    # Create agent and make a call that will raise an exception
    agent = Agent(model=mock_model)

    # Call the agent and catch the exception
    with pytest.raises(ValueError):
        agent("test prompt")

    # Verify span was created
    mock_tracer.start_agent_span.assert_called_once_with(
        messages=[{"content": [{"text": "test prompt"}], "role": "user"}],
        agent_name="Strands Agents",
        model_id=unittest.mock.ANY,
        tools=agent.tool_names,
        system_prompt=agent.system_prompt,
        custom_trace_attributes=agent.trace_attributes,
    )

    # Verify span was ended with the exception
    mock_tracer.end_agent_span.assert_called_once_with(span=mock_span, error=test_exception)


@pytest.mark.asyncio
@unittest.mock.patch("strands.agent.agent.get_tracer")
async def test_agent_stream_async_creates_and_ends_span_on_exception(mock_get_tracer, mock_model, alist):
    """Test that stream_async creates and ends a span when the call succeeds."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_agent_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Define the side effect to simulate callback handler raising an Exception
    test_exception = ValueError("Test exception")
    mock_model.mock_stream.side_effect = test_exception

    # Create agent and make a call
    agent = Agent(model=mock_model)

    # Call the agent and catch the exception
    with pytest.raises(ValueError):
        stream = agent.stream_async("test prompt")
        await alist(stream)

    # Verify span was created
    mock_tracer.start_agent_span.assert_called_once_with(
        messages=[{"content": [{"text": "test prompt"}], "role": "user"}],
        agent_name="Strands Agents",
        model_id=unittest.mock.ANY,
        tools=agent.tool_names,
        system_prompt=agent.system_prompt,
        custom_trace_attributes=agent.trace_attributes,
    )

    # Verify span was ended with the exception
    mock_tracer.end_agent_span.assert_called_once_with(span=mock_span, error=test_exception)


def test_agent_init_with_state_object():
    agent = Agent(state=AgentState({"foo": "bar"}))
    assert agent.state.get("foo") == "bar"


def test_non_dict_throws_error():
    with pytest.raises(ValueError, match="state must be an AgentState object or a dict"):
        agent = Agent(state={"object", object()})
        print(agent.state)


def test_non_json_serializable_state_throws_error():
    with pytest.raises(ValueError, match="Value is not JSON serializable"):
        agent = Agent(state={"object": object()})
        print(agent.state)


def test_agent_state_breaks_dict_reference():
    ref_dict = {"hello": "world"}
    agent = Agent(state=ref_dict)

    # Make sure shallow object references do not affect state maintained by AgentState
    ref_dict["hello"] = object()

    # This will fail if AgentState reflects the updated reference
    json.dumps(agent.state.get())


def test_agent_state_breaks_deep_dict_reference():
    ref_dict = {"world": "!"}
    init_dict = {"hello": ref_dict}
    agent = Agent(state=init_dict)
    # Make sure deep reference changes do not affect state mained by AgentState
    ref_dict["world"] = object()

    # This will fail if AgentState reflects the updated reference
    json.dumps(agent.state.get())


def test_agent_state_set_breaks_dict_reference():
    agent = Agent()
    ref_dict = {"hello": "world"}
    # Set should copy the input, and not maintain the reference to the original object
    agent.state.set("hello", ref_dict)
    ref_dict["hello"] = object()

    # This will fail if AgentState reflects the updated reference
    json.dumps(agent.state.get())


def test_agent_state_get_breaks_deep_dict_reference():
    agent = Agent(state={"hello": {"world": "!"}})
    # Get should not return a reference to the internal state
    ref_state = agent.state.get()
    ref_state["hello"]["world"] = object()

    # This will fail if AgentState reflects the updated reference
    json.dumps(agent.state.get())


def test_agent_session_management():
    mock_session_repository = MockedSessionRepository()
    session_manager = RepositorySessionManager(session_id="123", session_repository=mock_session_repository)
    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "hello!"}]}])
    agent = Agent(session_manager=session_manager, model=model)
    agent("Hello!")


def test_agent_restored_from_session_management():
    mock_session_repository = MockedSessionRepository()
    mock_session_repository.create_session(Session(session_id="123", session_type=SessionType.AGENT))
    mock_session_repository.create_agent(
        "123",
        SessionAgent(
            agent_id="default",
            state={"foo": "bar"},
            conversation_manager_state=SlidingWindowConversationManager().get_state(),
        ),
    )
    session_manager = RepositorySessionManager(session_id="123", session_repository=mock_session_repository)

    agent = Agent(session_manager=session_manager)

    assert agent.state.get("foo") == "bar"


def test_agent_restored_from_session_management_with_message():
    mock_session_repository = MockedSessionRepository()
    mock_session_repository.create_session(Session(session_id="123", session_type=SessionType.AGENT))
    mock_session_repository.create_agent(
        "123",
        SessionAgent(
            agent_id="default",
            state={"foo": "bar"},
            conversation_manager_state=SlidingWindowConversationManager().get_state(),
        ),
    )
    mock_session_repository.create_message(
        "123", "default", SessionMessage({"role": "user", "content": [{"text": "Hello!"}]}, 0)
    )
    session_manager = RepositorySessionManager(session_id="123", session_repository=mock_session_repository)

    agent = Agent(session_manager=session_manager)

    assert agent.state.get("foo") == "bar"


def test_agent_redacts_input_on_triggered_guardrail():
    mocked_model = MockedModelProvider(
        [{"redactedUserContent": "BLOCKED!", "redactedAssistantContent": "INPUT BLOCKED!"}]
    )

    agent = Agent(
        model=mocked_model,
        system_prompt="You are a helpful assistant.",
        callback_handler=None,
    )

    response1 = agent("CACTUS")

    assert response1.stop_reason == "guardrail_intervened"
    assert agent.messages[0]["content"][0]["text"] == "BLOCKED!"


def test_agent_restored_from_session_management_with_redacted_input():
    mocked_model = MockedModelProvider(
        [{"redactedUserContent": "BLOCKED!", "redactedAssistantContent": "INPUT BLOCKED!"}]
    )

    test_session_id = str(uuid4())
    mocked_session_repository = MockedSessionRepository()
    session_manager = RepositorySessionManager(session_id=test_session_id, session_repository=mocked_session_repository)

    agent = Agent(
        model=mocked_model,
        system_prompt="You are a helpful assistant.",
        callback_handler=None,
        session_manager=session_manager,
    )

    assert mocked_session_repository.read_agent(test_session_id, agent.agent_id) is not None

    response1 = agent("CACTUS")

    assert response1.stop_reason == "guardrail_intervened"
    assert agent.messages[0]["content"][0]["text"] == "BLOCKED!"
    user_input_session_message = mocked_session_repository.list_messages(test_session_id, agent.agent_id)[0]
    # Assert persisted message is equal to the redacted message in the agent
    assert user_input_session_message.to_message() == agent.messages[0]

    # Restore an agent from the session, confirm input is still redacted
    session_manager_2 = RepositorySessionManager(
        session_id=test_session_id, session_repository=mocked_session_repository
    )
    agent_2 = Agent(
        model=mocked_model,
        system_prompt="You are a helpful assistant.",
        callback_handler=None,
        session_manager=session_manager_2,
    )

    # Assert that the restored agent redacted message is equal to the original agent
    assert agent.messages[0] == agent_2.messages[0]


def test_agent_restored_from_session_management_with_correct_index():
    mock_model_provider = MockedModelProvider(
        [{"role": "assistant", "content": [{"text": "hello!"}]}, {"role": "assistant", "content": [{"text": "world!"}]}]
    )
    mock_session_repository = MockedSessionRepository()
    session_manager = RepositorySessionManager(session_id="test", session_repository=mock_session_repository)
    agent = Agent(session_manager=session_manager, model=mock_model_provider)
    agent("Hello!")

    assert len(mock_session_repository.list_messages("test", agent.agent_id)) == 2

    session_manager_2 = RepositorySessionManager(session_id="test", session_repository=mock_session_repository)
    agent_2 = Agent(session_manager=session_manager_2, model=mock_model_provider)

    assert len(agent_2.messages) == 2
    assert agent_2.messages[1]["content"][0]["text"] == "hello!"

    agent_2("Hello!")

    assert len(agent_2.messages) == 4
    session_messages = mock_session_repository.list_messages("test", agent_2.agent_id)
    assert (len(session_messages)) == 4
    assert session_messages[1].message["content"][0]["text"] == "hello!"
    assert session_messages[3].message["content"][0]["text"] == "world!"


def test_agent_with_session_and_conversation_manager():
    mock_model = MockedModelProvider([{"role": "assistant", "content": [{"text": "hello!"}]}])
    mock_session_repository = MockedSessionRepository()
    session_manager = RepositorySessionManager(session_id="123", session_repository=mock_session_repository)
    conversation_manager = SlidingWindowConversationManager(window_size=1)
    # Create an agent with a mocked model and session repository
    agent = Agent(
        session_manager=session_manager,
        conversation_manager=conversation_manager,
        model=mock_model,
    )

    # Assert session was initialized
    assert mock_session_repository.read_session("123") is not None
    assert mock_session_repository.read_agent("123", agent.agent_id) is not None
    assert len(mock_session_repository.list_messages("123", agent.agent_id)) == 0

    agent("Hello!")

    # After invoking, assert that the messages were persisted
    assert len(mock_session_repository.list_messages("123", agent.agent_id)) == 2
    # Assert conversation manager reduced the messages
    assert len(agent.messages) == 1

    # Initialize another agent using the same session
    session_manager_2 = RepositorySessionManager(session_id="123", session_repository=mock_session_repository)
    conversation_manager_2 = SlidingWindowConversationManager(window_size=1)
    agent_2 = Agent(
        session_manager=session_manager_2,
        conversation_manager=conversation_manager_2,
        model=mock_model,
    )
    # Assert that the second agent was initialized properly, and that the messages of both agents are equal
    assert agent.messages == agent_2.messages
    # Asser the conversation manager was initialized properly
    assert agent.conversation_manager.removed_message_count == agent_2.conversation_manager.removed_message_count


def test_agent_tool_non_serializable_parameter_filtering(agent, mock_randint):
    """Test that non-serializable objects in tool parameters are properly filtered during tool call recording."""
    mock_randint.return_value = 42

    # Create a non-serializable object (Agent instance)
    another_agent = Agent()

    # This should not crash even though we're passing non-serializable objects
    result = agent.tool.tool_decorated(
        random_string="test_value",
        non_serializable_agent=another_agent,  # This would previously cause JSON serialization error
        user_message_override="Testing non-serializable parameter filtering",
    )

    # Verify the tool executed successfully
    expected_result = {
        "content": [{"text": "test_value"}],
        "status": "success",
        "toolUseId": "tooluse_tool_decorated_42",
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
    assert "agent.tool.tool_decorated direct tool call." in tool_call_text
    assert '"random_string": "test_value"' in tool_call_text
    assert '"non_serializable_agent": "<<non-serializable: Agent>>"' not in tool_call_text


def test_agent_tool_no_non_serializable_parameters(agent, mock_randint):
    """Test that normal tool calls with only serializable parameters work unchanged."""
    mock_randint.return_value = 555

    # Call with only serializable parameters
    result = agent.tool.tool_decorated(random_string="normal_call", user_message_override="Normal tool call test")

    # Verify successful execution
    expected_result = {
        "content": [{"text": "normal_call"}],
        "status": "success",
        "toolUseId": "tooluse_tool_decorated_555",
    }
    assert result == expected_result

    # Check message recording works normally
    assert len(agent.messages) > 0
    user_message = agent.messages[0]
    tool_call_text = user_message["content"][1]["text"]

    # Verify normal parameter serialization (no filtering needed)
    assert "agent.tool.tool_decorated direct tool call." in tool_call_text
    assert '"random_string": "normal_call"' in tool_call_text
    # Should not contain any "<<non-serializable:" strings
    assert "<<non-serializable:" not in tool_call_text


def test_agent_tool_record_direct_tool_call_disabled_with_non_serializable(agent, mock_randint):
    """Test that when record_direct_tool_call is disabled, non-serializable parameters don't cause issues."""
    mock_randint.return_value = 777

    # Disable tool call recording
    agent.record_direct_tool_call = False

    # This should work fine even with non-serializable parameters since recording is disabled
    result = agent.tool.tool_decorated(
        random_string="no_recording", non_serializable_agent=Agent(), user_message_override="This shouldn't be recorded"
    )

    # Verify successful execution
    expected_result = {
        "content": [{"text": "no_recording"}],
        "status": "success",
        "toolUseId": "tooluse_tool_decorated_777",
    }
    assert result == expected_result

    # Verify no messages were recorded
    assert len(agent.messages) == 0


def test_agent_empty_invoke():
    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "hello!"}]}])
    agent = Agent(model=model, messages=[{"role": "user", "content": [{"text": "hello!"}]}])
    result = agent()
    assert str(result) == "hello!\n"
    assert len(agent.messages) == 2


def test_agent_empty_list_invoke():
    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "hello!"}]}])
    agent = Agent(model=model, messages=[{"role": "user", "content": [{"text": "hello!"}]}])
    result = agent([])
    assert str(result) == "hello!\n"
    assert len(agent.messages) == 2


def test_agent_with_assistant_role_message():
    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "world!"}]}])
    agent = Agent(model=model)
    assistant_message = [{"role": "assistant", "content": [{"text": "hello..."}]}]
    result = agent(assistant_message)
    assert str(result) == "world!\n"
    assert len(agent.messages) == 2


def test_agent_with_multiple_messages_on_invoke():
    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "world!"}]}])
    agent = Agent(model=model)
    input_messages = [
        {"role": "user", "content": [{"text": "hello"}]},
        {"role": "assistant", "content": [{"text": "..."}]},
    ]
    result = agent(input_messages)
    assert str(result) == "world!\n"
    assert len(agent.messages) == 3


def test_agent_with_invalid_input():
    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "world!"}]}])
    agent = Agent(model=model)
    with pytest.raises(ValueError, match="Input prompt must be of type: `str | list[Contentblock] | Messages | None`."):
        agent({"invalid": "input"})


def test_agent_with_invalid_input_list():
    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "world!"}]}])
    agent = Agent(model=model)
    with pytest.raises(ValueError, match="Input prompt must be of type: `str | list[Contentblock] | Messages | None`."):
        agent([{"invalid": "input"}])


def test_agent_with_list_of_message_and_content_block():
    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "world!"}]}])
    agent = Agent(model=model)
    with pytest.raises(ValueError, match="Input prompt must be of type: `str | list[Contentblock] | Messages | None`."):
        agent([{"role": "user", "content": [{"text": "hello"}]}, {"text", "hello"}])


def test_agent_tool_call_parameter_filtering_integration(mock_randint):
    """Test that tool calls properly filter parameters in message recording."""
    mock_randint.return_value = 42

    @strands.tool
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
