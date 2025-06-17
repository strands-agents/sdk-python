# Python
import unittest.mock

import pytest

from src.strands.models.portkey import PortkeyModel
from src.strands.types.exceptions import ContextWindowOverflowException


@pytest.fixture
def model_config():
    return {
        "api_key": "test_api_key",
        "virtual_key": "test_virtual_key",
        "base_url": "https://test.url",
        "model_id": "test_model_id",
        "provider": "openai",
    }


@pytest.fixture
def portkey_model(model_config):
    return PortkeyModel(**model_config)


def test__init__(portkey_model):
    assert portkey_model.config["api_key"] == "test_api_key"
    assert portkey_model.provider == "openai"


def test_get_config(portkey_model):
    config = portkey_model.get_config()
    assert config["api_key"] == "test_api_key"


def test_format_request_no_tools(portkey_model):
    messages = [{"role": "user", "content": "Hello"}]
    request = portkey_model.format_request(messages)
    assert "tools" not in request


def test_format_request_with_tools(portkey_model):
    messages = [{"role": "user", "content": "Hello"}]
    tool_specs = [{"name": "test_tool", "description": "Test tool", "inputSchema": {"json": {"properties": {}}}}]
    request = portkey_model.format_request(messages, tool_specs)
    assert "tools" in request


def test_format_request_system_prompt(portkey_model):
    messages = [{"role": "user", "content": "Hello"}]
    system_prompt = "Test system prompt"
    request = portkey_model.format_request(messages, system_prompt=system_prompt)
    assert request["messages"][0]["role"] == "system"


def test_allow_tool_use_openai(portkey_model):
    assert portkey_model._allow_tool_use()


def test_allow_tool_use_bedrock():
    model_config = {
        "api_key": "test_api_key",
        "virtual_key": "test_virtual_key",
        "base_url": "https://test.url",
        "model_id": "anthropic_model_id",
        "provider": "bedrock",
    }
    portkey_model = PortkeyModel(**model_config)
    assert portkey_model._allow_tool_use() is True


def test_allow_tool_use_false():
    model_config = {
        "api_key": "test_api_key",
        "virtual_key": "test_virtual_key",
        "base_url": "https://test.url",
        "model_id": "test_model_id",
        "provider": "unknown",
    }
    portkey_model = PortkeyModel(**model_config)
    assert portkey_model._allow_tool_use() is False


def test_stream(portkey_model):
    mock_event = {"choices": [{"delta": {"content": "test"}}]}
    with unittest.mock.patch.object(portkey_model.client.chat.completions, "create", return_value=iter([mock_event])):
        request = {"messages": [{"role": "user", "content": "Hello"}], "model": "test_model_id", "stream": True}
        response = list(portkey_model.stream(request))
        assert response[0]["choices"][0]["delta"]["content"] == "test"


def test_stream_context_window_exception(portkey_model):
    with unittest.mock.patch.object(
        portkey_model.client.chat.completions,
        "create",
        side_effect=ContextWindowOverflowException("Context window exceeded"),
    ):
        request = {"messages": [{"role": "user", "content": "Hello"}], "model": "test_model_id", "stream": True}
        with pytest.raises(ContextWindowOverflowException):
            list(portkey_model.stream(request))


def test_format_chunk_tool_calls(portkey_model):
    event = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "function": {"name": "test_tool", "arguments": "test_args"},
                            "type": "function",
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }
    chunk = portkey_model.format_chunk(event)
    assert "contentBlockStart" in chunk


def test_format_chunk_arguments_chunk(portkey_model):
    event = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "function": {"arguments": "test_args"},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }
    chunk = portkey_model.format_chunk(event)
    assert "contentBlockDelta" in chunk


def test_format_chunk_finish_reason_tool_calls(portkey_model):
    event = {"choices": [{"finish_reason": "tool_calls"}]}
    chunk = portkey_model.format_chunk(event)
    assert "contentBlockStop" in chunk


def test_format_chunk_usage(portkey_model):
    event = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "choices": [{"delta": {"content": None}}],  # Ensure 'content' key exists
    }
    chunk = portkey_model.format_chunk(event)
    assert chunk["metadata"]["usage"]["totalTokens"] == 15


def test_format_message_parts_string(portkey_model):
    parts = portkey_model._format_message_parts("user", "test content")
    assert parts == [{"role": "user", "content": "test content"}]


def test_format_message_parts_list_with_text(portkey_model):
    content = [{"text": "test text"}]
    parts = portkey_model._format_message_parts("assistant", content)
    assert parts == [{"role": "assistant", "content": "test text"}]


def test_format_message_parts_tool_use(portkey_model):
    content = [{"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {}}}]
    parts = portkey_model._format_message_parts("assistant", content)
    assert "tool_calls" in parts[0]


def test_format_message_parts_tool_result(portkey_model):
    portkey_model._current_tool_use_id = "123"
    content = [{"toolResult": {"content": [{"text": "result text"}]}}]
    parts = portkey_model._format_message_parts("assistant", content)
    assert parts[0]["content"] == "result text"


def test_map_tools(portkey_model):
    tool_specs = [
        {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {
                "json": {
                    "properties": {"arg1": {"type": "string"}},
                    "required": ["arg1"],
                }
            },
        }
    ]
    tools = portkey_model._map_tools(tool_specs)
    assert tools[0]["function"]["name"] == "test_tool"
    assert tools[0]["function"]["parameters"]["required"] == ["arg1"]


def test_format_tool_use_part(portkey_model):
    part = {"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {}}}
    formatted = portkey_model._format_tool_use_part(part)
    assert formatted["tool_calls"][0]["function"]["name"] == "test_tool"


def test_format_tool_result_part(portkey_model):
    portkey_model._current_tool_use_id = "123"
    part = {"toolResult": {"content": [{"text": "result text"}]}}
    formatted = portkey_model._format_tool_result_part(part)
    assert formatted["content"] == "result text"


def test_should_terminate_with_tool_use(portkey_model):
    event = {"choices": [{"finish_reason": "tool_calls"}]}
    assert portkey_model._should_terminate_with_tool_use(event) is True


def test_converse(portkey_model):
    mock_event = {"choices": [{"delta": {"content": "test"}}]}
    with unittest.mock.patch.object(portkey_model.client.chat.completions, "create", return_value=iter([mock_event])):
        messages = [{"role": "user", "content": "Hello"}]
        tool_specs = [{"name": "test_tool", "description": "Test tool", "inputSchema": {"json": {"properties": {}}}}]
        system_prompt = "Test system prompt"
        response = list(portkey_model.converse(messages, tool_specs, system_prompt))
        assert response[0]["messageStart"]["role"] == "assistant"
