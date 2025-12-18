import json
import unittest.mock

import pydantic
import pytest

from strands.models.neuronvllm import NeuronVLLMModel
from strands.types.content import Messages


@pytest.fixture
def neuronvllm_client(monkeypatch: pytest.MonkeyPatch) -> unittest.mock.Mock:
    from strands import models

    mock_client_cls = unittest.mock.Mock()
    mock_client = unittest.mock.AsyncMock()
    mock_client.chat.completions.create = unittest.mock.AsyncMock()
    mock_client_cls.return_value = mock_client

    monkeypatch.setattr(models.neuronvllm, "AsyncOpenAI", mock_client_cls)
    return mock_client


@pytest.fixture
def model_id() -> str:
    return "m1"


@pytest.fixture
def model(model_id: str) -> NeuronVLLMModel:
    return NeuronVLLMModel({"model_id": model_id})


@pytest.fixture
def messages() -> Messages:
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt() -> str:
    return "s1"


@pytest.fixture
def test_output_model_cls() -> type[pydantic.BaseModel]:
    class TestOutputModel(pydantic.BaseModel):
        name: str
        age: int

    return TestOutputModel


def test__init__model_configs(model_id: str) -> None:
    model = NeuronVLLMModel({"model_id": model_id, "max_tokens": 1})

    tru_max_tokens = model.get_config().get("max_tokens")
    exp_max_tokens = 1

    assert tru_max_tokens == exp_max_tokens


def test_update_config(model: NeuronVLLMModel, model_id: str) -> None:
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


def test_format_request_default(model: NeuronVLLMModel, messages: Messages, model_id: str) -> None:
    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "stop": None,
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_override(model: NeuronVLLMModel, messages: Messages, model_id: str) -> None:
    model.update_config(model_id=model_id)
    tru_request = model.format_request(messages, tool_specs=None)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "stop": None,
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_system_prompt(
    model: NeuronVLLMModel, messages: Messages, model_id: str, system_prompt: str
) -> None:
    tru_request = model.format_request(messages, system_prompt=system_prompt)
    exp_request = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "test"},
        ],
        "model": model_id,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "stop": None,
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_image(model: NeuronVLLMModel, model_id: str) -> None:
    messages: Messages = [{"role": "user", "content": [{"image": {"source": {"bytes": "base64encodedimage"}}}]}]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "images": ["base64encodedimage"]}],
        "model": model_id,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "stop": None,
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_tool_use(model: NeuronVLLMModel, model_id: str) -> None:
    messages: Messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "calculator", "input": '{"expression": "2+2"}'}}]}
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "2+2"}',
                        }
                    }
                ],
            }
        ],
        "model": model_id,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "stop": None,
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_tool_result(model: NeuronVLLMModel, model_id: str) -> None:
    messages: Messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "calculator",
                        "status": "success",
                        "content": [
                            {"text": "4"},
                            {"image": {"source": {"bytes": b"image"}}},
                            {"json": ["4"]},
                        ],
                    },
                },
                {
                    "text": "see results",
                },
            ],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "tool",
                "content": "4",
            },
            {
                "role": "tool",
                "images": [b"image"],
            },
            {
                "role": "tool",
                "content": '["4"]',
            },
            {
                "role": "user",
                "content": "see results",
            },
        ],
        "model": model_id,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "stop": None,
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_unsupported_type(model: NeuronVLLMModel) -> None:
    messages: Messages = [
        {
            "role": "user",
            "content": [{"unsupported": {}}],
        },
    ]

    with pytest.raises(TypeError, match="Unsupported content type: unsupported"):
        model.format_request(messages)


def test_format_request_with_tool_specs(model: NeuronVLLMModel, messages: Messages, model_id: str) -> None:
    tool_specs = [
        {
            "name": "calculator",
            "description": "Calculate mathematical expressions",
            "inputSchema": {
                "json": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
            },
        }
    ]

    tru_request = model.format_request(messages, tool_specs)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "stop": None,
        "stream": True,
        "functions": [
            {
                "name": "calculator",
                "description": "Calculate mathematical expressions",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            }
        ],
    }

    assert tru_request == exp_request


def test_format_request_with_inference_config(model: NeuronVLLMModel, messages: Messages, model_id: str) -> None:
    inference_config = {
        "max_tokens": 1,
        "stop_sequences": ["stop"],
        "temperature": 1.0,
        "top_p": 1.0,
    }

    model.update_config(**inference_config)
    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "temperature": inference_config["temperature"],
        "top_p": inference_config["top_p"],
        "max_tokens": inference_config["max_tokens"],
        "stop": inference_config["stop_sequences"],
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_additional_args(model: NeuronVLLMModel, messages: Messages, model_id: str) -> None:
    additional_args = {"o1": 1}

    model.update_config(additional_args=additional_args)
    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "stop": None,
        "stream": True,
        "o1": 1,
    }

    assert tru_request == exp_request


def test_format_chunk_message_start(model: NeuronVLLMModel) -> None:
    event = {"chunk_type": "message_start"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStart": {"role": "assistant"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_start_text(model: NeuronVLLMModel) -> None:
    event = {"chunk_type": "content_start", "data_type": "text"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStart": {"start": {}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_start_tool(model: NeuronVLLMModel) -> None:
    mock_function = unittest.mock.Mock()
    mock_function.function.name = "calculator"

    event = {"chunk_type": "content_start", "data_type": "tool", "data": mock_function}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStart": {"start": {"toolUse": {"name": "calculator", "toolUseId": "calculator"}}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_delta_text(model: NeuronVLLMModel) -> None:
    event = {"chunk_type": "content_delta", "data_type": "text", "data": "Hello"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockDelta": {"delta": {"text": "Hello"}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_delta_tool(model: NeuronVLLMModel) -> None:
    event = {
        "chunk_type": "content_delta",
        "data_type": "tool",
        "data": unittest.mock.Mock(function=unittest.mock.Mock(arguments={"expression": "2+2"})),
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({"expression": "2+2"})}}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_stop(model: NeuronVLLMModel) -> None:
    event = {"chunk_type": "content_stop"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStop": {}}

    assert tru_chunk == exp_chunk


def test_format_chunk_message_stop_end_turn(model: NeuronVLLMModel) -> None:
    event = {"chunk_type": "message_stop", "data": "stop"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "end_turn"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_message_stop_tool_use(model: NeuronVLLMModel) -> None:
    event = {"chunk_type": "message_stop", "data": "tool_use"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "tool_use"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_metadata(model: NeuronVLLMModel) -> None:
    event = {
        "chunk_type": "metadata",
        "data": unittest.mock.Mock(),
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "metadata": {
            "usage": {},
            "metrics": {},
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_other(model: NeuronVLLMModel) -> None:
    event = {"chunk_type": "other"}

    with pytest.raises(RuntimeError, match="Unknown chunk_type: other"):
        model.format_chunk(event)


@pytest.mark.asyncio
async def test_stream(
    neuronvllm_client: unittest.mock.Mock,
    model: NeuronVLLMModel,
    agenerator,
    alist,
) -> None:
    mock_chunk = unittest.mock.Mock()
    mock_choice = unittest.mock.Mock()
    mock_delta = unittest.mock.Mock()
    mock_delta.content = "Hello"
    mock_delta.tool_calls = None
    mock_choice.delta = mock_delta
    mock_choice.finish_reason = "stop"
    mock_chunk.choices = [mock_choice]

    neuronvllm_client.chat.completions.create.return_value = agenerator([mock_chunk])

    messages: Messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    assert tru_events == exp_events

    expected_request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "m1",
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "stop": None,
        "stream": True,
    }
    neuronvllm_client.chat.completions.create.assert_awaited_once_with(**expected_request)


@pytest.mark.asyncio
async def test_stream_with_tool_calls(
    neuronvllm_client: unittest.mock.Mock,
    model: NeuronVLLMModel,
    agenerator,
    alist,
) -> None:
    mock_chunk = unittest.mock.Mock()
    mock_choice = unittest.mock.Mock()
    mock_delta = unittest.mock.Mock()

    mock_tool_call = unittest.mock.Mock()
    mock_tool_call.function.name = "calculator"
    mock_tool_call.function.arguments = {"expression": "2+2"}

    mock_delta.content = "I'll calculate that for you"
    mock_delta.tool_calls = [mock_tool_call]
    mock_choice.delta = mock_delta
    mock_choice.finish_reason = "stop"
    mock_chunk.choices = [mock_choice]

    neuronvllm_client.chat.completions.create.return_value = agenerator([mock_chunk])

    messages: Messages = [{"role": "user", "content": [{"text": "Calculate 2+2"}]}]
    response = model.stream(messages)

    tru_events = await alist(response)

    # Basic structural checks: first start, last stop
    assert tru_events[0] == {"messageStart": {"role": "assistant"}}
    assert tru_events[1] == {"contentBlockStart": {"start": {}}}
    assert tru_events[-1] == {"messageStop": {"stopReason": "tool_use"}}

    # One toolUse start with expected name/id
    tool_starts = [
        e
        for e in tru_events
        if e.get("contentBlockStart", {}).get("start", {}).get("toolUse") is not None
    ]
    assert len(tool_starts) == 1
    tool_use = tool_starts[0]["contentBlockStart"]["start"]["toolUse"]
    assert tool_use["name"] == "calculator"
    assert tool_use["toolUseId"] == "calculator"

    # One toolUse delta with expected input
    tool_deltas = [
        e
        for e in tru_events
        if "contentBlockDelta" in e
        and "toolUse" in e["contentBlockDelta"]["delta"]
    ]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["contentBlockDelta"]["delta"]["toolUse"]["input"] == '{"expression": "2+2"}'

    # One text delta with the assistant message
    text_deltas = [
        e
        for e in tru_events
        if "contentBlockDelta" in e
        and "text" in e["contentBlockDelta"]["delta"]
    ]
    assert len(text_deltas) == 1
    assert text_deltas[0]["contentBlockDelta"]["delta"]["text"] == "I'll calculate that for you"

    expected_request = {
        "messages": [{"role": "user", "content": "Calculate 2+2"}],
        "model": "m1",
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "stop": None,
        "stream": True,
    }
    neuronvllm_client.chat.completions.create.assert_awaited_once_with(**expected_request)


@pytest.mark.asyncio
async def test_structured_output(
    neuronvllm_client: unittest.mock.Mock,
    model: NeuronVLLMModel,
    test_output_model_cls: type[pydantic.BaseModel],
    alist,
) -> None:
    messages: Messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    mock_response = unittest.mock.Mock()
    mock_tool_call = unittest.mock.Mock()
    mock_tool_call.function.arguments = '{"name": "John", "age": 30}'
    mock_message = unittest.mock.Mock()
    mock_message.tool_calls = [mock_tool_call]
    mock_choice = unittest.mock.Mock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    neuronvllm_client.chat.completions.create = unittest.mock.AsyncMock(return_value=mock_response)

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    tru_result = events[-1]
    exp_result = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_result == exp_result


