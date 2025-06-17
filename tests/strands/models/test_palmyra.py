import unittest.mock

import pytest

import strands
from strands.models.palmyra import PalmyraModel


@pytest.fixture
def palmyra_client_cls():
    with unittest.mock.patch.object(strands.models.palmyra.writerai, "Client") as mock_client_cls:
        yield mock_client_cls


@pytest.fixture
def palmyra_client(palmyra_client_cls):
    return palmyra_client_cls.return_value


@pytest.fixture
def client_args():
    return {"api_key": "writer_api_key"}


@pytest.fixture
def model_name():
    return "palmyra-x5"


@pytest.fixture
def stream_options():
    return {"include_usage": True}


@pytest.fixture
def model(palmyra_client, model_name, stream_options, client_args):
    _ = palmyra_client

    return PalmyraModel(client_args, model=model_name, stream_options=stream_options)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "System prompt"


def test__init__(palmyra_client_cls, model_name, stream_options, client_args):
    model = PalmyraModel(client_args=client_args, model=model_name, stream_options=stream_options)

    config = model.get_config()
    exp_config = {"stream_options": stream_options, "model": model_name}

    assert config == exp_config

    palmyra_client_cls.assert_called_once_with(api_key=client_args.get("api_key", ""))


def test_update_config(model):
    model.update_config(model="palmyra-x4")

    model_id = model.get_config().get("model")

    assert model_id == "palmyra-x4"


def test_format_request_basic(model, messages, model_name, stream_options):
    request = model.format_request(messages)

    exp_request = {
        "stream": True,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "model": model_name,
        "tools": [],
        "stream_options": stream_options,
    }

    assert request == exp_request


def test_format_request_with_params(model, messages, model_name, stream_options):
    model.update_config(temperature=0.19)

    request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "model": model_name,
        "tools": [],
        "stream_options": stream_options,
        "temperature": 0.19,
        "stream": True,
    }

    assert request == exp_request


def test_format_request_with_system_prompt(model, messages, model_name, stream_options, system_prompt):
    request = model.format_request(messages, system_prompt=system_prompt)

    exp_request = {
        "messages": [
            {"content": "System prompt", "role": "system"},
            {"content": [{"text": "test", "type": "text"}], "role": "user"},
        ],
        "model": model_name,
        "tools": [],
        "stream_options": stream_options,
        "stream": True,
    }

    assert request == exp_request


def test_format_request_with_tool_use(model, model_name, stream_options):
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
        "model": model_name,
        "tools": [],
        "stream_options": stream_options,
        "stream": True,
    }

    assert request == exp_request


def test_format_request_with_tool_resultsasync(model, model_name, stream_options):
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
        "model": model_name,
        "tools": [],
        "stream_options": stream_options,
        "stream": True,
    }

    assert request == exp_request


def test_format_request_with_empty_content(model, model_name, stream_options):
    messages = [
        {
            "role": "user",
            "content": [],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [],
        "model": model_name,
        "tools": [],
        "stream_options": stream_options,
        "stream": True,
    }

    assert tru_request == exp_request


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        ({"image": {}}, "image"),
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


def test_stream(palmyra_client, model, model_name):
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

    palmyra_client.chat.chat.return_value = iter([mock_event_1, mock_event_2, mock_event_3, mock_event_4])

    request = {
        "model": model_name,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "calculate 2+2"}]}],
    }
    response = model.stream(request)

    events = list(response)
    exp_events = [
        {"chunk_type": "message_start"},
        {"chunk_type": "content_block_start", "data_type": "text"},
        {"chunk_type": "content_block_delta", "data_type": "text", "data": "I'll calculate"},
        {"chunk_type": "content_block_delta", "data_type": "text", "data": "that for you"},
        {"chunk_type": "content_block_stop", "data_type": "text"},
        {"chunk_type": "content_block_start", "data_type": "tool", "data": mock_tool_call_1_part_1},
        {"chunk_type": "content_block_delta", "data_type": "tool", "data": mock_tool_call_1_part_2},
        {"chunk_type": "content_block_stop", "data_type": "tool"},
        {"chunk_type": "content_block_start", "data_type": "tool", "data": mock_tool_call_2_part_1},
        {"chunk_type": "content_block_delta", "data_type": "tool", "data": mock_tool_call_2_part_2},
        {"chunk_type": "content_block_stop", "data_type": "tool"},
        {"chunk_type": "message_stop", "data": "tool_calls"},
        {"chunk_type": "metadata", "data": mock_event_4.usage},
    ]

    assert events == exp_events
    palmyra_client.chat.chat(**request)


def test_stream_empty(palmyra_client, model, model_name):
    mock_delta = unittest.mock.Mock(content=None, tool_calls=None)
    mock_usage = unittest.mock.Mock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock()
    mock_event_4 = unittest.mock.Mock(usage=mock_usage)

    palmyra_client.chat.chat.return_value = iter([mock_event_1, mock_event_2, mock_event_3, mock_event_4])

    request = {"model": model_name, "messages": [{"role": "user", "content": []}]}
    response = model.stream(request)

    events = list(response)
    exp_events = [
        {"chunk_type": "message_start"},
        {"chunk_type": "content_block_start", "data_type": "text"},
        {"chunk_type": "content_block_stop", "data_type": "text"},
        {"chunk_type": "message_stop", "data": "stop"},
        {"chunk_type": "metadata", "data": mock_usage},
    ]

    assert events == exp_events
    palmyra_client.chat.chat.assert_called_once_with(**request)


def test_stream_with_empty_choices(palmyra_client, model, model_name):
    mock_delta = unittest.mock.Mock(content="content", tool_calls=None)
    mock_usage = unittest.mock.Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    mock_event_1 = unittest.mock.Mock(spec=[])
    mock_event_2 = unittest.mock.Mock(choices=[])
    mock_event_3 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_4 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_5 = unittest.mock.Mock(usage=mock_usage)

    palmyra_client.chat.chat.return_value = iter([mock_event_1, mock_event_2, mock_event_3, mock_event_4, mock_event_5])

    request = {"model": model_name, "messages": [{"role": "user", "content": ["test"]}]}
    response = model.stream(request)

    events = list(response)
    exp_events = [
        {"chunk_type": "message_start"},
        {"chunk_type": "content_block_start", "data_type": "text"},
        {"chunk_type": "content_block_delta", "data_type": "text", "data": "content"},
        {"chunk_type": "content_block_delta", "data_type": "text", "data": "content"},
        {"chunk_type": "content_block_stop", "data_type": "text"},
        {"chunk_type": "message_stop", "data": "stop"},
        {"chunk_type": "metadata", "data": mock_usage},
    ]

    assert events == exp_events
    palmyra_client.chat.chat.assert_called_once_with(**request)
