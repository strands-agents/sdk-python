import json
import logging
import unittest.mock

import pydantic
import pytest
from google import genai

import strands
from strands.models.gemini import GeminiModel
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException


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
        "config": {},
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
        "config": {"system_instruction": system_prompt},
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
        "config": {},
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
        "config": {},
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
                            "signature": "YWJj",  # base64 of "abc"
                            "text": "reasoning_text",
                        },
                    },
                },
            ],
        },
    ]
    await anext(model.stream(messages))

    exp_request = {
        "config": {},
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
    """Test toolUse with thoughtSignature is sent as function_call."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "c1",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                        "thoughtSignature": "YWJj",  # base64 of "abc" - required for Gemini thinking models
                    },
                },
            ],
        },
    ]
    await anext(model.stream(messages))

    exp_request = {
        "config": {},
        "contents": [
            {
                "parts": [
                    {
                        "function_call": {
                            "args": {"expression": "2+2"},
                            "id": "c1",
                            "name": "calculator",
                        },
                        "thought_signature": "YWJj",
                    },
                ],
                "role": "model",
            },
        ],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_request_with_tool_use_no_thought_signature(gemini_client, model, model_id):
    """Test toolUse without thoughtSignature is converted to text for backwards compatibility."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "c1",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                        # No thoughtSignature - simulates old session history
                    },
                },
            ],
        },
    ]
    await anext(model.stream(messages))

    # Even without thoughtSignature, toolUse should be a proper function call
    exp_request = {
        "config": {},
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
        "config": {},
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
async def test_stream_request_with_tool_results_preserving_name(gemini_client, model, model_id):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "t1",
                        "name": "tool_1",
                        "input": {},
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
                        "content": [{"text": "done"}],
                    },
                },
            ],
        },
    ]
    await anext(model.stream(messages))

    exp_request = {
        "config": {},
        "contents": [
            {
                "parts": [
                    {
                        "function_call": {
                            "args": {},
                            "id": "t1",
                            "name": "tool_1",
                        },
                    },
                ],
                "role": "model",
            },
            {
                "parts": [
                    {
                        "function_response": {
                            "id": "t1",
                            "name": "tool_1",
                            "response": {"output": [{"text": "done"}]},
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
        "config": {},
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
        {"contentBlockStart": {"start": {"toolUse": {"name": "calculator", "toolUseId": "c1"}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"expression": "2+2"}'}}}},
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
    # signature is base64 encoded: b"abc" -> "YWJj"
    exp_chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "YWJj", "text": "test reason"}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}, "metrics": {"latencyMs": 0}}},
    ]
    assert tru_chunks == exp_chunks


@pytest.mark.asyncio
async def test_stream_response_reasoning_and_text(gemini_client, model, messages, agenerator, alist):
    """Test that both reasoning and text content are captured in separate blocks."""
    gemini_client.aio.models.generate_content_stream.return_value = agenerator(
        [
            genai.types.GenerateContentResponse(
                candidates=[
                    genai.types.Candidate(
                        content=genai.types.Content(
                            parts=[
                                genai.types.Part(
                                    text="thinking about math",
                                    thought=True,
                                    thought_signature=b"sig1",
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
            genai.types.GenerateContentResponse(
                candidates=[
                    genai.types.Candidate(
                        content=genai.types.Content(
                            parts=[
                                genai.types.Part(
                                    text="2 + 2 = 4",
                                    thought=False,
                                ),
                            ],
                        ),
                        finish_reason="STOP",
                    ),
                ],
                usage_metadata=genai.types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=1,
                    total_token_count=5,
                ),
            ),
        ]
    )

    tru_chunks = await alist(model.stream(messages))
    exp_chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "c2lnMQ==", "text": "thinking about math"}}}},
        {"contentBlockStop": {}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "2 + 2 = 4"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 4, "totalTokens": 5}, "metrics": {"latencyMs": 0}}},
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
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}, "metrics": {"latencyMs": 0}}},
    ]
    assert tru_chunks == exp_chunks


@pytest.mark.asyncio
async def test_stream_response_empty_stream(gemini_client, model, messages, agenerator, alist):
    """Test that empty stream doesn't raise UnboundLocalError.

    When the stream yields no events, the candidate variable must be initialized
    to None to avoid UnboundLocalError when referenced in message_stop chunk.
    """
    gemini_client.aio.models.generate_content_stream.return_value = agenerator([])

    tru_chunks = await alist(model.stream(messages))
    exp_chunks = [
        {"messageStart": {"role": "assistant"}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    assert tru_chunks == exp_chunks


@pytest.mark.asyncio
async def test_stream_response_throttled_exception(gemini_client, model, messages):
    gemini_client.aio.models.generate_content_stream.side_effect = genai.errors.ClientError(
        429, {"message": '{"error": {"status": "RESOURCE_EXHAUSTED"}}'}
    )

    with pytest.raises(ModelThrottledException, match="RESOURCE_EXHAUSTED"):
        await anext(model.stream(messages))


@pytest.mark.asyncio
async def test_stream_response_context_overflow_exception(gemini_client, model, messages):
    gemini_client.aio.models.generate_content_stream.side_effect = genai.errors.ClientError(
        400,
        {
            "message": json.dumps(
                {
                    "error": {
                        "message": "request exceeds the maximum number of tokens (100)",
                        "status": "INVALID_ARGUMENT",
                    },
                }
            ),
        },
    )

    with pytest.raises(ContextWindowOverflowException, match="INVALID_ARGUMENT"):
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
            "response_mime_type": "application/json",
            "response_schema": weather_output.model_json_schema(),
        },
        "contents": [{"parts": [{"text": "test"}], "role": "user"}],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content.assert_called_with(**exp_request)


def test_gemini_tools_validation_rejects_function_declarations(model_id):
    tool_with_function_declarations = genai.types.Tool(
        function_declarations=[
            genai.types.FunctionDeclaration(
                name="test_function",
                description="A test function",
            )
        ]
    )

    with pytest.raises(ValueError, match="gemini_tools should not contain FunctionDeclarations"):
        GeminiModel(model_id=model_id, gemini_tools=[tool_with_function_declarations])


def test_gemini_tools_validation_allows_non_function_tools(model_id):
    tool_with_google_search = genai.types.Tool(google_search=genai.types.GoogleSearch())

    model = GeminiModel(model_id=model_id, gemini_tools=[tool_with_google_search])
    assert "gemini_tools" in model.config


def test_gemini_tools_validation_on_update_config(model):
    tool_with_function_declarations = genai.types.Tool(
        function_declarations=[
            genai.types.FunctionDeclaration(
                name="test_function",
                description="A test function",
            )
        ]
    )

    with pytest.raises(ValueError, match="gemini_tools should not contain FunctionDeclarations"):
        model.update_config(gemini_tools=[tool_with_function_declarations])


@pytest.mark.asyncio
async def test_stream_request_with_gemini_tools(gemini_client, messages, model_id):
    google_search_tool = genai.types.Tool(google_search=genai.types.GoogleSearch())
    model = GeminiModel(model_id=model_id, gemini_tools=[google_search_tool])

    await anext(model.stream(messages))

    # When only gemini_tools are provided (no tool_specs), only gemini_tools are included
    exp_request = {
        "config": {
            "tools": [
                {"google_search": {}},
            ]
        },
        "contents": [{"parts": [{"text": "test"}], "role": "user"}],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_request_with_gemini_tools_and_function_tools(gemini_client, messages, tool_spec, model_id):
    code_execution_tool = genai.types.Tool(code_execution=genai.types.ToolCodeExecution())
    model = GeminiModel(model_id=model_id, gemini_tools=[code_execution_tool])

    await anext(model.stream(messages, tool_specs=[tool_spec]))

    exp_request = {
        "config": {
            "tools": [
                {
                    "function_declarations": [
                        {
                            "description": tool_spec["description"],
                            "name": tool_spec["name"],
                            "parameters_json_schema": tool_spec["inputSchema"]["json"],
                        }
                    ]
                },
                {"code_execution": {}},
            ]
        },
        "contents": [{"parts": [{"text": "test"}], "role": "user"}],
        "model": model_id,
    }
    gemini_client.aio.models.generate_content_stream.assert_called_with(**exp_request)


@pytest.mark.asyncio
async def test_stream_handles_non_json_error(gemini_client, model, messages, caplog, alist):
    error_message = "Invalid API key"
    gemini_client.aio.models.generate_content_stream.side_effect = genai.errors.ClientError(
        error_message, {"message": error_message}
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(genai.errors.ClientError, match=error_message):
            await alist(model.stream(messages))

    assert "Gemini API returned non-JSON error" in caplog.text
    assert f"error_message=<{error_message}>" in caplog.text


@pytest.mark.asyncio
async def test_stream_with_injected_client(model_id, agenerator, alist):
    """Test that stream works with an injected client and doesn't close it."""
    # Create a mock injected client
    mock_injected_client = unittest.mock.Mock()
    mock_injected_client.aio = unittest.mock.AsyncMock()

    mock_injected_client.aio.models.generate_content_stream.return_value = agenerator(
        [
            genai.types.GenerateContentResponse(
                candidates=[
                    genai.types.Candidate(
                        content=genai.types.Content(
                            parts=[genai.types.Part(text="Hello")],
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

    # Create model with injected client
    model = GeminiModel(client=mock_injected_client, model_id=model_id)

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)

    # Verify events were generated
    assert len(tru_events) > 0

    # Verify the injected client was used
    mock_injected_client.aio.models.generate_content_stream.assert_called_once()


@pytest.mark.asyncio
async def test_structured_output_with_injected_client(model_id, weather_output, alist):
    """Test that structured_output works with an injected client and doesn't close it."""
    # Create a mock injected client
    mock_injected_client = unittest.mock.Mock()
    mock_injected_client.aio = unittest.mock.AsyncMock()

    mock_injected_client.aio.models.generate_content.return_value = unittest.mock.Mock(
        parsed=weather_output.model_dump()
    )

    # Create model with injected client
    model = GeminiModel(client=mock_injected_client, model_id=model_id)

    messages = [{"role": "user", "content": [{"text": "Generate weather"}]}]
    stream = model.structured_output(type(weather_output), messages)
    events = await alist(stream)

    # Verify output was generated
    assert len(events) == 1
    assert events[0] == {"output": weather_output}

    # Verify the injected client was used
    mock_injected_client.aio.models.generate_content.assert_called_once()


def test_init_with_both_client_and_client_args_raises_error():
    """Test that providing both client and client_args raises ValueError."""
    mock_client = unittest.mock.Mock()

    with pytest.raises(ValueError, match="Only one of 'client' or 'client_args' should be provided"):
        GeminiModel(client=mock_client, client_args={"api_key": "test"}, model_id="test-model")


def test_format_request_filters_s3_source_image(model, caplog):
    """Test that images with Location sources are filtered out with warning."""
    caplog.set_level(logging.WARNING, logger="strands.models.gemini")

    messages = [
        {
            "role": "user",
            "content": [
                {"text": "look at this image"},
                {
                    "image": {
                        "format": "png",
                        "source": {"location": {"type": "s3", "uri": "s3://my-bucket/image.png"}},
                    },
                },
            ],
        },
    ]

    request = model._format_request(messages, None, None, None)

    # Image with S3 source should be filtered, text should remain
    formatted_content = request["contents"][0]["parts"]
    assert len(formatted_content) == 1
    assert "text" in formatted_content[0]
    assert "Location sources are not supported by Gemini" in caplog.text


def test_format_request_filters_location_source_document(model, caplog):
    """Test that documents with Location sources are filtered out with warning."""
    caplog.set_level(logging.WARNING, logger="strands.models.gemini")

    messages = [
        {
            "role": "user",
            "content": [
                {"text": "analyze this document"},
                {
                    "document": {
                        "format": "pdf",
                        "name": "report.pdf",
                        "source": {"location": {"type": "s3", "uri": "s3://my-bucket/report.pdf"}},
                    },
                },
                {
                    "document": {
                        "format": "pdf",
                        "name": "report.pdf",
                        "source": {"location": {"type": "s3", "uri": "s3://my-bucket/report.pdf"}},
                    },
                },
            ],
        },
    ]

    request = model._format_request(messages, None, None, None)

    # Document with S3 source should be filtered, text should remain
    formatted_content = request["contents"][0]["parts"]
    assert len(formatted_content) == 1
    assert "text" in formatted_content[0]
    assert "Location sources are not supported by Gemini" in caplog.text
