import unittest.mock

import pytest

import strands
from strands.models.openai import OpenAIModel


@pytest.fixture
def openai_client_cls():
    with unittest.mock.patch.object(strands.models.openai.openai, "OpenAI") as mock_client_cls:
        yield mock_client_cls


@pytest.fixture
def openai_client(openai_client_cls):
    return openai_client_cls.return_value


@pytest.fixture
def model_id():
    return "m1"


@pytest.fixture
def model(openai_client, model_id):
    _ = openai_client

    return OpenAIModel(model_id=model_id)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "s1"


def test__init__(openai_client_cls, model_id):
    model = OpenAIModel({"api_key": "k1"}, model_id=model_id, params={"max_tokens": 1})

    tru_config = model.get_config()
    exp_config = {"model_id": "m1", "params": {"max_tokens": 1}}

    assert tru_config == exp_config

    openai_client_cls.assert_called_once_with(api_key="k1")


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


def test_stream(openai_client, model):
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

    openai_client.chat.completions.create.return_value = iter([mock_event_1, mock_event_2, mock_event_3, mock_event_4])

    request = {"model": "m1", "messages": [{"role": "user", "content": [{"type": "text", "text": "calculate 2+2"}]}]}
    response = model.stream(request)

    tru_events = list(response)
    exp_events = [
        {"chunk_type": "message_start"},
        {"chunk_type": "content_start", "data_type": "text"},
        {"chunk_type": "content_delta", "data_type": "text", "data": "I'll calculate"},
        {"chunk_type": "content_delta", "data_type": "text", "data": "that for you"},
        {"chunk_type": "content_stop", "data_type": "text"},
        {"chunk_type": "content_start", "data_type": "tool", "data": mock_tool_call_1_part_1},
        {"chunk_type": "content_delta", "data_type": "tool", "data": mock_tool_call_1_part_1},
        {"chunk_type": "content_delta", "data_type": "tool", "data": mock_tool_call_1_part_2},
        {"chunk_type": "content_stop", "data_type": "tool"},
        {"chunk_type": "content_start", "data_type": "tool", "data": mock_tool_call_2_part_1},
        {"chunk_type": "content_delta", "data_type": "tool", "data": mock_tool_call_2_part_1},
        {"chunk_type": "content_delta", "data_type": "tool", "data": mock_tool_call_2_part_2},
        {"chunk_type": "content_stop", "data_type": "tool"},
        {"chunk_type": "message_stop", "data": "tool_calls"},
        {"chunk_type": "metadata", "data": mock_event_4.usage},
    ]

    assert tru_events == exp_events
    openai_client.chat.completions.create.assert_called_once_with(**request)


def test_stream_empty(openai_client, model):
    mock_delta = unittest.mock.Mock(content=None, tool_calls=None)
    mock_usage = unittest.mock.Mock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock()
    mock_event_4 = unittest.mock.Mock(usage=mock_usage)

    openai_client.chat.completions.create.return_value = iter([mock_event_1, mock_event_2, mock_event_3, mock_event_4])

    request = {"model": "m1", "messages": [{"role": "user", "content": []}]}
    response = model.stream(request)

    tru_events = list(response)
    exp_events = [
        {"chunk_type": "message_start"},
        {"chunk_type": "content_start", "data_type": "text"},
        {"chunk_type": "content_stop", "data_type": "text"},
        {"chunk_type": "message_stop", "data": "stop"},
        {"chunk_type": "metadata", "data": mock_usage},
    ]

    assert tru_events == exp_events
    openai_client.chat.completions.create.assert_called_once_with(**request)
