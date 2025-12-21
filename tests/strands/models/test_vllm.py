import unittest.mock

import pytest

import strands
from strands.models.vllm import VLLMModel


@pytest.fixture
def openai_client():
    with unittest.mock.patch.object(strands.models.vllm.openai, "AsyncOpenAI") as mock_client_cls:
        mock_client = unittest.mock.AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        yield mock_client


@pytest.mark.asyncio
async def test_vllm_model_injects_return_token_ids_by_default(openai_client, agenerator, alist):
    model = VLLMModel(model_id="m1", params={"max_tokens": 1}, return_token_ids=True)

    mock_delta = unittest.mock.Mock(content="hi", tool_calls=None, reasoning_content=None)
    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock(usage=None)
    openai_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3])
    )

    messages = [{"role": "user", "content": [{"text": "hello"}]}]
    _ = await alist(model.stream(messages))

    called_kwargs = openai_client.chat.completions.create.call_args.kwargs
    assert called_kwargs["extra_body"]["return_token_ids"] is True


@pytest.mark.asyncio
async def test_vllm_model_moves_prompt_token_ids_into_extra_body(openai_client, agenerator, alist):
    model = VLLMModel(model_id="m1", params={"max_tokens": 1})

    mock_delta = unittest.mock.Mock(content="hi", tool_calls=None, reasoning_content=None)
    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock(usage=None)
    openai_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3])
    )

    messages = [{"role": "user", "content": [{"text": "hello"}]}]
    # Force chat token-in to validate prompt_token_ids placement in chat requests.
    _ = await alist(
        model.stream(
            messages,
            prompt_token_ids=[1, 2, 3],
            token_in_endpoint="chat",
            extra_body={"foo": "bar"},
        )
    )

    called_kwargs = openai_client.chat.completions.create.call_args.kwargs
    # prompt_token_ids should *not* be a top-level OpenAI request parameter.
    assert "prompt_token_ids" not in called_kwargs
    assert called_kwargs["extra_body"]["prompt_token_ids"] == [1, 2, 3]
    assert called_kwargs["extra_body"]["foo"] == "bar"


@pytest.mark.asyncio
async def test_vllm_stream_token_ids_uses_completions(openai_client, agenerator, alist):
    model = VLLMModel(model_id="m1", params={}, return_token_ids=True)

    # Mock streaming completion events (text deltas + finish)
    choice1 = unittest.mock.Mock(text="hi", finish_reason=None)
    choice2 = unittest.mock.Mock(text=None, finish_reason="stop")
    ev1 = unittest.mock.Mock(choices=[choice1])
    ev2 = unittest.mock.Mock(choices=[choice2])

    openai_client.completions.create = unittest.mock.AsyncMock(return_value=agenerator([ev1, ev2]))

    # Token-only mode is exercised via the main stream() entrypoint.
    messages = [{"role": "user", "content": [{"text": "ignored"}]}]
    events = await alist(
        model.stream(
            messages,
            prompt_token_ids=[1, 2, 3],
            token_in_endpoint="completions",
            max_tokens=4,
        )
    )

    # Ensure we called the completions endpoint (token-only mode).
    called_kwargs = openai_client.completions.create.call_args.kwargs
    assert called_kwargs["prompt"] == " "
    assert called_kwargs["stream"] is True
    assert called_kwargs["extra_body"]["prompt_token_ids"] == [1, 2, 3]
    assert called_kwargs["extra_body"]["return_token_ids"] is True

    # Basic event shape
    assert events[0] == {"messageStart": {"role": "assistant"}}
    assert any("messageStop" in e for e in events)


@pytest.mark.asyncio
async def test_vllm_stream_chat_token_ids_uses_chat_completions(openai_client, agenerator, alist):
    model = VLLMModel(model_id="m1", params={"max_tokens": 4}, return_token_ids=True)

    mock_delta = unittest.mock.Mock(content="hi", tool_calls=None, reasoning_content=None)
    ev1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    ev2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    ev3 = unittest.mock.Mock(usage=None)

    openai_client.chat.completions.create = unittest.mock.AsyncMock(return_value=agenerator([ev1, ev2, ev3]))

    # Token-in chat mode is exercised via the main stream() entrypoint.
    messages = [{"role": "user", "content": [{"text": "ignored"}]}]
    events = await alist(model.stream(messages, prompt_token_ids=[11, 22, 33], token_in_endpoint="chat"))

    called_kwargs = openai_client.chat.completions.create.call_args.kwargs
    assert called_kwargs["extra_body"]["prompt_token_ids"] == [11, 22, 33]
    assert called_kwargs["extra_body"]["return_token_ids"] is True
    assert called_kwargs["stream"] is True
    assert isinstance(events, list) and events


@pytest.mark.asyncio
async def test_vllm_stream_routes_prompt_token_ids_to_completions(openai_client, agenerator, alist):
    model = VLLMModel(model_id="m1", params={}, return_token_ids=True)

    choice1 = unittest.mock.Mock(text="hi", finish_reason=None)
    choice2 = unittest.mock.Mock(text=None, finish_reason="stop")
    ev1 = unittest.mock.Mock(choices=[choice1])
    ev2 = unittest.mock.Mock(choices=[choice2])
    openai_client.completions.create = unittest.mock.AsyncMock(return_value=agenerator([ev1, ev2]))

    # messages are required by the base interface, but will be ignored in completions token-in mode.
    messages = [{"role": "user", "content": [{"text": "ignored"}]}]
    events = await alist(
        model.stream(
            messages,
            prompt_token_ids=[1, 2, 3],
            token_in_endpoint="completions",
            max_tokens=4,
        )
    )

    called_kwargs = openai_client.completions.create.call_args.kwargs
    assert called_kwargs["extra_body"]["prompt_token_ids"] == [1, 2, 3]
    assert any("messageStop" in e for e in events)


@pytest.mark.asyncio
async def test_vllm_stream_routes_prompt_token_ids_to_chat_when_tools_present(openai_client, agenerator, alist):
    model = VLLMModel(model_id="m1", params={"max_tokens": 4}, return_token_ids=True)

    mock_delta = unittest.mock.Mock(content="hi", tool_calls=None, reasoning_content=None)
    ev1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    ev2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    ev3 = unittest.mock.Mock(usage=None)
    openai_client.chat.completions.create = unittest.mock.AsyncMock(return_value=agenerator([ev1, ev2, ev3]))

    tool_specs = [
        {
            "name": "echo_tool",
            "description": "Echo input text.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                }
            },
        }
    ]

    messages = [{"role": "user", "content": [{"text": "ignored"}]}]
    _ = await alist(model.stream(messages, tool_specs=tool_specs, prompt_token_ids=[9, 9, 9], token_in_endpoint="auto"))

    called_kwargs = openai_client.chat.completions.create.call_args.kwargs
    assert called_kwargs["extra_body"]["prompt_token_ids"] == [9, 9, 9]
