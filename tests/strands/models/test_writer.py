import unittest.mock
from typing import Any, List

import pytest

import strands
from strands.models.writer import WriterModel


@pytest.fixture
def writer_client_cls():
    with unittest.mock.patch.object(strands.models.writer.writerai, "AsyncClient") as mock_client_cls:
        yield mock_client_cls


@pytest.fixture
def writer_client(writer_client_cls):
    return writer_client_cls.return_value


@pytest.fixture
def client_args():
    return {"api_key": "writer_api_key"}


@pytest.fixture
def model_id():
    return "palmyra-x5"


@pytest.fixture
def stream_options():
    return {"include_usage": True}


@pytest.fixture
def model(writer_client, model_id, stream_options, client_args):
    _ = writer_client

    return WriterModel(client_args, model_id=model_id, stream_options=stream_options)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "System prompt"


def test__init__(writer_client_cls, model_id, stream_options, client_args):
    model = WriterModel(client_args=client_args, model_id=model_id, stream_options=stream_options)

    config = model.get_config()
    exp_config = {"stream_options": stream_options, "model_id": model_id}

    assert config == exp_config

    writer_client_cls.assert_called_once_with(api_key=client_args.get("api_key", ""))


def test_update_config(model):
    model.update_config(model_id="palmyra-x4")

    model_id = model.get_config().get("model_id")

    assert model_id == "palmyra-x4"


def test_format_request_basic(model, messages, model_id, stream_options):
    request = model.format_request(messages)

    exp_request = {
        "stream": True,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "model": model_id,
        "stream_options": stream_options,
    }

    assert request == exp_request


def test_format_request_with_params(model, messages, model_id, stream_options):
    model.update_config(temperature=0.19)

    request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "model": model_id,
        "stream_options": stream_options,
        "temperature": 0.19,
        "stream": True,
    }

    assert request == exp_request


def test_format_request_with_system_prompt(model, messages, model_id, stream_options, system_prompt):
    request = model.format_request(messages, system_prompt=system_prompt)

    exp_request = {
        "messages": [
            {"content": "System prompt", "role": "system"},
            {"content": [{"text": "test", "type": "text"}], "role": "user"},
        ],
        "model": model_id,
        "stream_options": stream_options,
        "stream": True,
    }

    assert request == exp_request


def test_format_request_with_tool_use(model, model_id, stream_options):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "c1",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    },
                },
            ],
        },
    ]

    request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {"arguments": '{"expression": "2+2"}', "name": "calculator"},
                        "id": "c1",
                        "type": "function",
                    }
                ],
            },
        ],
        "model": model_id,
        "stream_options": stream_options,
        "stream": True,
    }

    assert request == exp_request


def test_format_request_with_tool_results(model, model_id, stream_options):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "c1",
                        "status": "success",
                        "content": [
                            {"text": "answer is 4"},
                        ],
                    }
                }
            ],
        }
    ]

    request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "tool",
                "content": [{"text": "answer is 4", "type": "text"}],
                "tool_call_id": "c1",
            },
        ],
        "model": model_id,
        "stream_options": stream_options,
        "stream": True,
    }

    assert request == exp_request


def test_format_request_with_image(model, model_id, stream_options):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": b"lovely sunny day"},
                    },
                },
            ],
        },
    ]

    request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "image_url": {
                            "url": "data:image/png;base64,bG92ZWx5IHN1bm55IGRheQ==",
                        },
                        "type": "image_url",
                    },
                ],
            },
        ],
        "model": model_id,
        "stream": True,
        "stream_options": stream_options,
    }

    assert request == exp_request


def test_format_request_with_empty_content(model, model_id, stream_options):
    messages = [
        {
            "role": "user",
            "content": [],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [],
        "model": model_id,
        "stream_options": stream_options,
        "stream": True,
    }

    assert tru_request == exp_request


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        ({"video": {}}, "video"),
        ({"document": {}}, "document"),
        ({"reasoningContent": {}}, "reasoningContent"),
        ({"other": {}}, "other"),
    ],
)
def test_format_request_with_unsupported_type(model, content, content_type):
    messages = [
        {
            "role": "user",
            "content": [content],
        },
    ]

    with pytest.raises(TypeError, match=f"content_type=<{content_type}> | unsupported type"):
        model.format_request(messages)


class AsyncStreamWrapper:
    def __init__(self, items: List[Any]):
        self.items = items

    def __aiter__(self):
        return self._generator()

    async def _generator(self):
        for item in self.items:
            yield item


async def mock_streaming_response(items: List[Any]):
    return AsyncStreamWrapper(items)


@pytest.mark.asyncio
async def test_stream(writer_client, model, model_id):
    mock_tool_call_1_part_1 = unittest.mock.Mock(index=0)
    mock_tool_call_2_part_1 = unittest.mock.Mock(index=1)
    mock_delta_1 = unittest.mock.Mock(
        content="I'll calculate", tool_calls=[mock_tool_call_1_part_1, mock_tool_call_2_part_1]
    )

    mock_tool_call_1_part_2 = unittest.mock.Mock(index=0)
    mock_tool_call_2_part_2 = unittest.mock.Mock(index=1)
    mock_delta_2 = unittest.mock.Mock(
        content="that for you", tool_calls=[mock_tool_call_1_part_2, mock_tool_call_2_part_2]
    )

    mock_delta_3 = unittest.mock.Mock(content="", tool_calls=None)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_1)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_2)])
    mock_event_3 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="tool_calls", delta=mock_delta_3)])
    mock_event_4 = unittest.mock.Mock()

    writer_client.chat.chat.return_value = mock_streaming_response(
        [mock_event_1, mock_event_2, mock_event_3, mock_event_4]
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": "calculate 2+2"}]}]
    response = model.stream(messages, None, None)

    # Consume the response
    [event async for event in response]

    # The events should be formatted through format_chunk, so they should be StreamEvent objects
    expected_request = {
        "model": model_id,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "calculate 2+2"}]}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    writer_client.chat.chat.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_stream_empty(writer_client, model, model_id):
    mock_delta = unittest.mock.Mock(content=None, tool_calls=None)
    mock_usage = unittest.mock.Mock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock()
    mock_event_4 = unittest.mock.Mock(usage=mock_usage)

    writer_client.chat.chat.return_value = mock_streaming_response(
        [mock_event_1, mock_event_2, mock_event_3, mock_event_4]
    )

    messages = [{"role": "user", "content": []}]
    response = model.stream(messages, None, None)

    # Consume the response
    [event async for event in response]

    expected_request = {
        "model": model_id,
        "messages": [],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    writer_client.chat.chat.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_stream_with_empty_choices(writer_client, model, model_id):
    mock_delta = unittest.mock.Mock(content="content", tool_calls=None)
    mock_usage = unittest.mock.Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    mock_event_1 = unittest.mock.Mock(spec=[])
    mock_event_2 = unittest.mock.Mock(choices=[])
    mock_event_3 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_4 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_5 = unittest.mock.Mock(usage=mock_usage)

    writer_client.chat.chat.return_value = mock_streaming_response(
        [mock_event_1, mock_event_2, mock_event_3, mock_event_4, mock_event_5]
    )

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    response = model.stream(messages, None, None)

    # Consume the response
    [event async for event in response]

    expected_request = {
        "model": model_id,
        "messages": [{"role": "user", "content": [{"text": "test", "type": "text"}]}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    writer_client.chat.chat.assert_called_once_with(**expected_request)
