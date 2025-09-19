import unittest.mock

import pydantic
import pytest
from google import genai

import strands
from strands.models.gemini import GeminiModel
from strands.types.exceptions import ModelThrottledException


@pytest.fixture
def gemini_client():
    with unittest.mock.patch.object(strands.models.gemini.genai, "Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.aio = unittest.mock.AsyncMock()
        yield mock_client


@pytest.fixture
def model_id():
    return "m1"


@pytest.fixture
def model(gemini_client, model_id):
    _ = gemini_client

    return GeminiModel(model_id=model_id)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def tool_spec():
    return {
        "description": "description",
        "name": "name",
        "inputSchema": {"json": {"key": "val"}},
    }


@pytest.fixture
def system_prompt():
    return "s1"


@pytest.fixture
def weather_output():
    class Weather(pydantic.BaseModel):
        time: str
        weather: str

    return Weather(time="12:00", weather="sunny")


def test__init__model_configs(gemini_client, model_id):
    _ = gemini_client

    model = GeminiModel(model_id=model_id, params={"temperature": 1})

    tru_temperature = model.get_config().get("params")
    exp_temperature = {"temperature": 1}

    assert tru_temperature == exp_temperature


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


@pytest.mark.asyncio
async def test_stream_request_default(gemini_client, model, messages, model_id):
    await anext(model.stream(messages))

    exp_request = {
        "config": {"tools": [{"function_declarations": []}]},
        "contents": [{"parts": [{"text": "test"}], "role": "user"}],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_request_with_params(gemini_client, model, messages, model_id):
    model.update_config(params={"temperature": 1})

    await anext(model.stream(messages))

    exp_request = {
        "config": {
            "tools": [{"function_declarations": []}],
            "temperature": 1,
        },
        "contents": [{"parts": [{"text": "test"}], "role": "user"}],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_request_with_system_prompt(gemini_client, model, messages, model_id, system_prompt):
    await anext(model.stream(messages, system_prompt=system_prompt))

    exp_request = {
        "config": {"system_instruction": system_prompt, "tools": [{"function_declarations": []}]},
        "contents": [{"parts": [{"text": "test"}], "role": "user"}],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.parametrize(
    ("content", "formatted_part"),
    [
        # # PDF
        (
            {"document": {"format": "pdf", "name": "test doc", "source": {"bytes": b"pdf"}}},
            {"inline_data": {"data": "cGRm", "mime_type": "application/pdf"}},
        ),
        # Plain text
        (
            {"document": {"format": "txt", "name": "test doc", "source": {"bytes": b"txt"}}},
            {"inline_data": {"data": "dHh0", "mime_type": "text/plain"}},
        ),
    ],
)
@pytest.mark.asyncio
async def test_stream_request_with_document(content, formatted_part, gemini_client, model, model_id):
    messages = [
        {
            "role": "user",
            "content": [content],
        },
    ]
    await anext(model.stream(messages))

    exp_request = {
        "config": {
            "tools": [{"function_declarations": []}],
        },
        "contents": [{"parts": [formatted_part], "role": "user"}],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_request_with_image(gemini_client, model, model_id):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpg",
                        "source": {"bytes": b"base64encodedimage"},
                    },
                },
            ],
        },
    ]
    await anext(model.stream(messages))

    exp_request = {
        "config": {
            "tools": [{"function_declarations": []}],
        },
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "data": "YmFzZTY0ZW5jb2RlZGltYWdl",
                            "mime_type": "image/jpeg",
                        },
                    },
                ],
                "role": "user",
            },
        ],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_request_with_reasoning(gemini_client, model, model_id):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "signature": "abc",
                            "text": "reasoning_text",
                        },
                    },
                },
            ],
        },
    ]
    await anext(model.stream(messages))

    exp_request = {
        "config": {
            "tools": [{"function_declarations": []}],
        },
        "contents": [
            {
                "parts": [
                    {
                        "text": "reasoning_text",
                        "thought": True,
                        "thought_signature": "YWJj",
                    },
                ],
                "role": "user",
            },
        ],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_request_with_tool_spec(gemini_client, model, model_id, tool_spec):
    await anext(model.stream([], [tool_spec]))

    exp_request = {
        "config": {
            "tools": [
                {
                    "function_declarations": [
                        {
                            "description": "description",
                            "name": "name",
                            "parameters_json_schema": {"key": "val"},
                        },
                    ],
                },
            ],
        },
        "contents": [],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_request_with_tool_use(gemini_client, model, model_id):
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
    await anext(model.stream(messages))

    exp_request = {
        "config": {
            "tools": [{"function_declarations": []}],
        },
        "contents": [
            {
                "parts": [
                    {
                        "function_call": {
                            "args": {"expression": "2+2"},
                            "id": "c1",
                            "name": "calculator",
                        },
                    },
                ],
                "role": "model",
            },
        ],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_request_with_tool_results(gemini_client, model, model_id):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "c1",
                        "status": "success",
                        "content": [
                            {"text": "see image"},
                            {"json": ["see image"]},
                            {
                                "image": {
                                    "format": "jpg",
                                    "source": {"bytes": b"base64encodedimage"},
                                },
                            },
                        ],
                    }
                }
            ],
        }
    ]
    await anext(model.stream(messages))

    exp_request = {
        "config": {
            "tools": [{"function_declarations": []}],
        },
        "contents": [
            {
                "parts": [
                    {
                        "function_response": {
                            "id": "c1",
                            "name": "c1",
                            "response": {
                                "output": [
                                    {"text": "see image"},
                                    {"json": ["see image"]},
                                    {
                                        "inline_data": {
                                            "data": "YmFzZTY0ZW5jb2RlZGltYWdl",
                                            "mime_type": "image/jpeg",
                                        },
                                    },
                                ],
                            },
                        },
                    },
                ],
                "role": "user",
            },
        ],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_request_with_empty_content(gemini_client, model, model_id):
    messages = [
        {
            "role": "user",
            "content": [],
        },
    ]
    await anext(model.stream(messages))

    exp_request = {
        "config": {
            "tools": [{"function_declarations": []}],
        },
        "contents": [{"parts": [], "role": "user"}],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_request_with_unsupported_type(model):
    messages = [
        {
            "role": "user",
            "content": [{"unsupported": {}}],
        },
    ]

    with pytest.raises(TypeError, match="content_type=<unsupported> | unsupported type"):
        await anext(model.stream(messages))


@pytest.mark.asyncio
async def test_stream_response_text(gemini_client, model, messages, agenerator, alist):
    gemini_client.aio.models.generate_content_stream.return_value = agenerator(
        [
            genai.types.GenerateContentResponse(
                candidates=[
                    genai.types.Candidate(
                        content=genai.types.Content(
                            parts=[genai.types.Part(text="test text")],
                        ),
                        finish_reason="STOP",
                    ),
                ],
                usage_metadata=genai.types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=1,
                    total_token_count=3,
                ),
            ),
        ]
    )

    tru_chunks = await alist(model.stream(messages))
    exp_chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "test text"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}, "metrics": {"latencyMs": 0}}},
    ]
    assert tru_chunks == exp_chunks


@pytest.mark.asyncio
async def test_stream_response_tool_use(gemini_client, model, messages, agenerator, alist):
    gemini_client.aio.models.generate_content_stream.return_value = agenerator(
        [
            genai.types.GenerateContentResponse(
                candidates=[
                    genai.types.Candidate(
                        content=genai.types.Content(
                            parts=[
                                genai.types.Part(
                                    function_call=genai.types.FunctionCall(
                                        args={"expression": "2+2"},
                                        id="c1",
                                        name="calculator",
                                    ),
                                ),
                            ],
                        ),
                        finish_reason="STOP",
                    ),
                ],
                usage_metadata=genai.types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=1,
                    total_token_count=3,
                ),
            ),
        ]
    )

    tru_chunks = await alist(model.stream(messages))
    exp_chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockStart": {"start": {"toolUse": {"name": "calculator", "toolUseId": "calculator"}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"expression": "2+2"}'}}}},
        {"contentBlockStop": {}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}, "metrics": {"latencyMs": 0}}},
    ]
    assert tru_chunks == exp_chunks


@pytest.mark.asyncio
async def test_stream_response_reasoning(gemini_client, model, messages, agenerator, alist):
    gemini_client.aio.models.generate_content_stream.return_value = agenerator(
        [
            genai.types.GenerateContentResponse(
                candidates=[
                    genai.types.Candidate(
                        content=genai.types.Content(
                            parts=[
                                genai.types.Part(
                                    text="test reason",
                                    thought=True,
                                    thought_signature=b"abc",
                                ),
                            ],
                        ),
                        finish_reason="STOP",
                    ),
                ],
                usage_metadata=genai.types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=1,
                    total_token_count=3,
                ),
            ),
        ]
    )

    tru_chunks = await alist(model.stream(messages))
    exp_chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "abc", "text": "test reason"}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}, "metrics": {"latencyMs": 0}}},
    ]
    assert tru_chunks == exp_chunks


@pytest.mark.asyncio
async def test_stream_response_max_tokens(gemini_client, model, messages, agenerator, alist):
    gemini_client.aio.models.generate_content_stream.return_value = agenerator(
        [
            genai.types.GenerateContentResponse(
                candidates=[
                    genai.types.Candidate(
                        content=genai.types.Content(
                            parts=[genai.types.Part(text="test text")],
                        ),
                        finish_reason="MAX_TOKENS",
                    ),
                ],
                usage_metadata=genai.types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=1,
                    total_token_count=3,
                ),
            ),
        ]
    )

    tru_chunks = await alist(model.stream(messages))
    exp_chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "test text"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "max_tokens"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}, "metrics": {"latencyMs": 0}}},
    ]
    assert tru_chunks == exp_chunks


@pytest.mark.asyncio
async def test_stream_response_none_candidates(gemini_client, model, messages, agenerator, alist):
    gemini_client.aio.models.generate_content_stream.return_value = agenerator(
        [
            genai.types.GenerateContentResponse(
                candidates=None,
                usage_metadata=genai.types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=1,
                    total_token_count=3,
                ),
            ),
        ]
    )

    tru_chunks = await alist(model.stream(messages))
    exp_chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}, "metrics": {"latencyMs": 0}}},
    ]
    assert tru_chunks == exp_chunks


@pytest.mark.asyncio
async def test_stream_response_throttled_exception(gemini_client, model, messages):
    gemini_client.aio.models.generate_content_stream.side_effect = genai.errors.ClientError(
        429, {"status": "RESOURCE_EXHAUSTED"}
    )

    with pytest.raises(ModelThrottledException, match="RESOURCE_EXHAUSTED"):
        await anext(model.stream(messages))


@pytest.mark.asyncio
async def test_stream_response_client_exception(gemini_client, model, messages):
    gemini_client.aio.models.generate_content_stream.side_effect = genai.errors.ClientError(500, {"status": "INTERNAL"})

    with pytest.raises(genai.errors.ClientError, match="INTERNAL"):
        await anext(model.stream(messages))


@pytest.mark.asyncio
async def test_structured_output(gemini_client, model, messages, model_id, weather_output):
    gemini_client.aio.models.generate_content.return_value = unittest.mock.Mock(parsed=weather_output.model_dump())

    tru_response = await anext(model.structured_output(type(weather_output), messages))
    exp_response = {"output": weather_output}
    assert tru_response == exp_response

    exp_request = {
        "config": {
            "tools": [{"function_declarations": []}],
            "response_mime_type": "application/json",
            "response_schema": weather_output.model_json_schema(),
        },
        "contents": [{"parts": [{"text": "test"}], "role": "user"}],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content.assert_called_with(**exp_request)
