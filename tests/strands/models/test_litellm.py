import unittest.mock

import pydantic
import pytest

import strands
from strands.models.litellm import LiteLLMModel


@pytest.fixture
def litellm_acompletion():
    with unittest.mock.patch.object(strands.models.litellm.litellm, "acompletion") as mock_acompletion:
        yield mock_acompletion


@pytest.fixture
def api_key():
    return "a1"


@pytest.fixture
def model_id():
    return "m1"


@pytest.fixture
def model(litellm_acompletion, api_key, model_id):
    _ = litellm_acompletion

    return LiteLLMModel(client_args={"api_key": api_key}, model_id=model_id)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "s1"


@pytest.fixture
def test_output_model_cls():
    class TestOutputModel(pydantic.BaseModel):
        name: str
        age: int

    return TestOutputModel


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


@pytest.mark.parametrize(
    "content, exp_result",
    [
        # Case 1: Thinking
        (
            {
                "reasoningContent": {
                    "reasoningText": {
                        "signature": "reasoning_signature",
                        "text": "reasoning_text",
                    },
                },
            },
            {
                "signature": "reasoning_signature",
                "thinking": "reasoning_text",
                "type": "thinking",
            },
        ),
        # Case 2: Video
        (
            {
                "video": {
                    "source": {"bytes": "base64encodedvideo"},
                },
            },
            {
                "type": "video_url",
                "video_url": {
                    "detail": "auto",
                    "url": "base64encodedvideo",
                },
            },
        ),
        # Case 3: Text
        (
            {"text": "hello"},
            {"type": "text", "text": "hello"},
        ),
    ],
)
def test_format_request_message_content(content, exp_result):
    tru_result = LiteLLMModel.format_request_message_content(content)
    assert tru_result == exp_result


@pytest.mark.asyncio
async def test_stream(litellm_acompletion, api_key, model_id, model, agenerator, alist):
    mock_tool_call_1_part_1 = unittest.mock.Mock(index=0)
    mock_tool_call_2_part_1 = unittest.mock.Mock(index=1)
    mock_delta_1 = unittest.mock.Mock(
        reasoning_content="",
        content=None,
        tool_calls=None,
    )
    mock_delta_2 = unittest.mock.Mock(
        reasoning_content="\nI'm thinking",
        content=None,
        tool_calls=None,
    )
    mock_delta_3 = unittest.mock.Mock(
        content="I'll calculate", tool_calls=[mock_tool_call_1_part_1, mock_tool_call_2_part_1], reasoning_content=None
    )

    mock_tool_call_1_part_2 = unittest.mock.Mock(index=0)
    mock_tool_call_2_part_2 = unittest.mock.Mock(index=1)
    mock_delta_4 = unittest.mock.Mock(
        content="that for you", tool_calls=[mock_tool_call_1_part_2, mock_tool_call_2_part_2], reasoning_content=None
    )

    mock_delta_5 = unittest.mock.Mock(content="", tool_calls=None, reasoning_content=None)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_1)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_2)])
    mock_event_3 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_3)])
    mock_event_4 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_4)])
    mock_event_5 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="tool_calls", delta=mock_delta_5)])
    mock_event_6 = unittest.mock.Mock()

    litellm_acompletion.side_effect = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3, mock_event_4, mock_event_5, mock_event_6])
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": "calculate 2+2"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "\nI'm thinking"}}}},
        {"contentBlockDelta": {"delta": {"text": "I'll calculate"}}},
        {"contentBlockDelta": {"delta": {"text": "that for you"}}},
        {"contentBlockStop": {}},
        {
            "contentBlockStart": {
                "start": {
                    "toolUse": {"name": mock_tool_call_1_part_1.function.name, "toolUseId": mock_tool_call_1_part_1.id}
                }
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": mock_tool_call_1_part_1.function.arguments}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": mock_tool_call_1_part_2.function.arguments}}}},
        {"contentBlockStop": {}},
        {
            "contentBlockStart": {
                "start": {
                    "toolUse": {"name": mock_tool_call_2_part_1.function.name, "toolUseId": mock_tool_call_2_part_1.id}
                }
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": mock_tool_call_2_part_1.function.arguments}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": mock_tool_call_2_part_2.function.arguments}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": mock_event_6.usage.prompt_tokens,
                    "outputTokens": mock_event_6.usage.completion_tokens,
                    "totalTokens": mock_event_6.usage.total_tokens,
                },
                "metrics": {"latencyMs": 0},
            }
        },
    ]

    assert tru_events == exp_events

    expected_request = {
        "api_key": api_key,
        "model": model_id,
        "messages": [{"role": "user", "content": [{"text": "calculate 2+2", "type": "text"}]}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [],
    }
    litellm_acompletion.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_stream_empty(litellm_acompletion, api_key, model_id, model, agenerator, alist):
    mock_delta = unittest.mock.Mock(content=None, tool_calls=None, reasoning_content=None)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock()
    mock_event_4 = unittest.mock.Mock(usage=None)

    litellm_acompletion.side_effect = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3, mock_event_4])
    )

    messages = [{"role": "user", "content": []}]
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    assert len(tru_events) == len(exp_events)
    expected_request = {
        "api_key": api_key,
        "model": model_id,
        "messages": [],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [],
    }
    litellm_acompletion.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_structured_output(litellm_acompletion, model, test_output_model_cls, alist):
    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    mock_choice = unittest.mock.Mock()
    mock_choice.finish_reason = "tool_calls"
    mock_choice.message.content = '{"name": "John", "age": 30}'
    mock_response = unittest.mock.Mock()
    mock_response.choices = [mock_choice]

    litellm_acompletion.side_effect = unittest.mock.AsyncMock(return_value=mock_response)

    with unittest.mock.patch.object(strands.models.litellm, "supports_response_schema", return_value=True):
        stream = model.structured_output(test_output_model_cls, messages)
        events = await alist(stream)
        tru_result = events[-1]

    exp_result = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_result == exp_result
