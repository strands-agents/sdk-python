import base64
import unittest.mock
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic
import pytest
from google import genai
from google.genai import types  # Import types for enums

from src.strands.models.gemini import GeminiModel
from src.strands.types.exceptions import ModelThrottledException


@pytest.fixture
def mock_genai_client():
    with patch.object(genai, "Client") as mock_client:
        mock_instance = mock_client.return_value
        yield mock_instance


@pytest.fixture
def mock_model():
    model = MagicMock()
    return model


@pytest.fixture
def model_id():
    return "gemini-pro"


@pytest.fixture
def max_tokens():
    return 100


@pytest.fixture
def model(mock_genai_client, model_id):
    with patch.object(genai, "Client"):
        # Use temperature instead of max_tokens to avoid validation errors
        return GeminiModel(model_id=model_id, params={"temperature": 0.7})


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def agenerator():
    """Create an async generator from a list of items."""

    async def _async_generator(items):
        for item in items:
            yield item

    return _async_generator


@pytest.fixture
def system_prompt():
    return "You are a helpful assistant."


@pytest.fixture
def tool_specs():
    return [
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                    },
                    "required": ["input"],
                },
            },
        },
    ]


@pytest.fixture
def test_output_model_cls():
    class TestOutputModel(pydantic.BaseModel):
        name: str
        age: int

    return TestOutputModel


def test__init__model_configs():
    with patch.object(genai, "Client"):
        model = GeminiModel(model_id="gemini-pro", params={"temperature": 0.7}, api_key="fake-key")

        true_params = model.get_config().get("params")
        exp_params = {"temperature": 0.7}

        assert true_params == exp_params


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    true_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert true_model_id == exp_model_id


def test__format_request_message_content_text(model):
    content = {"text": "Hello, world!"}

    true_result = model._format_request_message_content(content)
    exp_result = {"text": "Hello, world!"}

    assert true_result == exp_result


def test__format_request_message_content_image(model):
    content = {
        "image": {
            "format": "jpg",
            "source": {"bytes": b"testimage"},
        }
    }

    true_result = model._format_request_message_content(content)
    exp_result = {
        "inlineData": {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(b"testimage").decode("utf-8"),
        }
    }

    assert true_result == exp_result


def test__format_request_message_content_document(model):
    content = {
        "document": {
            "format": "pdf",
            "source": {"bytes": b"testdoc"},
        }
    }

    true_result = model._format_request_message_content(content)
    exp_result = {
        "inlineData": {
            "mimeType": "application/pdf",
            "data": base64.b64encode(b"testdoc").decode("utf-8"),
        }
    }

    assert true_result == exp_result


def test__format_request_message_content_unsupported(model):
    content = {"unsupported": {}}

    with pytest.raises(TypeError, match="content_type=<unsupported> | unsupported type"):
        model._format_request_message_content(content)


def test__format_function_call(model):
    tool_use = {
        "name": "calculator",
        "toolUseId": "calc1",
        "input": {"expression": "2+2"},
    }

    true_result = model._format_function_call(tool_use)
    exp_result = {
        "functionCall": {
            "name": "calculator",
            "args": {"expression": "2+2"},
        }
    }

    assert true_result == exp_result


def test__format_function_response(model):
    tool_result = {
        "name": "calculator",
        "toolUseId": "calc1",
        "status": "success",
        "content": [
            {"text": "Result:"},
            {"json": {"result": 4}},
        ],
    }

    true_result = model._format_function_response(tool_result)
    exp_result = {"functionResponse": {"name": "calc1", "response": {"content": 'Result:\n{"result": 4}'}}}

    assert true_result == exp_result


def test_format_request_basic(model, messages, model_id):
    true_request = model._format_request(messages)
    exp_request = {
        "contents": [{"role": "user", "parts": [{"text": "test"}]}],
        "generation_config": {"temperature": 0.7},
        "stream": True,
    }

    assert true_request == exp_request


def test_format_request_with_system_prompt(model, messages, system_prompt):
    true_request = model._format_request(messages, system_prompt=system_prompt)
    exp_request = {
        "contents": [{"role": "user", "parts": [{"text": "test"}]}],
        "generation_config": {"temperature": 0.7},
        "stream": True,
        "system_instruction": {"parts": [{"text": system_prompt}]},
    }

    assert true_request == exp_request


def test_format_request_with_tools(model, messages, tool_specs):
    true_request = model._format_request(messages, tool_specs=tool_specs)
    exp_request = {
        "contents": [{"role": "user", "parts": [{"text": "test"}]}],
        "generation_config": {"temperature": 0.7},
        "stream": True,
        "tools": [
            {
                "function_declarations": [
                    {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {"type": "string"},
                            },
                            "required": ["input"],
                        },
                    }
                ]
            }
        ],
    }

    assert true_request == exp_request


def test_format_request_with_complex_messages(model):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Analyze this image:"},
                {"image": {"format": "jpg", "source": {"bytes": b"image_data"}}},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"name": "analyzer", "toolUseId": "t1", "input": {"detail_level": "high"}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "status": "success",
                        "content": [{"text": "Analysis complete"}, {"json": {"confidence": 0.95}}],
                    }
                },
            ],
        },
    ]

    true_request = model._format_request(messages)

    # Check structure without deep equality
    assert len(true_request["contents"]) == 3
    assert true_request["contents"][0]["role"] == "user"
    assert len(true_request["contents"][0]["parts"]) == 2
    assert true_request["contents"][0]["parts"][0]["text"] == "Analyze this image:"
    assert "inlineData" in true_request["contents"][0]["parts"][1]

    assert true_request["contents"][1]["role"] == "model"
    assert "functionCall" in true_request["contents"][1]["parts"][0]

    assert true_request["contents"][2]["role"] == "user"
    assert "functionResponse" in true_request["contents"][2]["parts"][0]


@pytest.mark.asyncio
async def test_stream_basic(model, messages, agenerator):
    client = model.client

    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace()

    mock_part1 = unittest.mock.MagicMock()
    mock_part1.text = "Hello"
    mock_part1.function_call = None

    mock_part2 = unittest.mock.MagicMock()
    mock_part2.text = " world"
    mock_part2.function_call = None

    mock_part3 = unittest.mock.MagicMock()
    mock_part3.text = "!"
    mock_part3.function_call = None

    mock_content1 = unittest.mock.MagicMock()
    mock_content1.parts = [mock_part1]

    mock_content2 = unittest.mock.MagicMock()
    mock_content2.parts = [mock_part2]

    mock_content3 = unittest.mock.MagicMock()
    mock_content3.parts = [mock_part3]

    mock_candidate1 = unittest.mock.MagicMock()
    mock_candidate1.content = mock_content1
    mock_candidate1.finish_reason = None

    mock_candidate2 = unittest.mock.MagicMock()
    mock_candidate2.content = mock_content2
    mock_candidate2.finish_reason = None

    mock_candidate3 = unittest.mock.MagicMock()
    mock_candidate3.content = mock_content3
    mock_candidate3.finish_reason = types.FinishReason.STOP

    chunks = [
        unittest.mock.MagicMock(candidates=[mock_candidate1]),
        unittest.mock.MagicMock(candidates=[mock_candidate2]),
        unittest.mock.MagicMock(candidates=[mock_candidate3]),
    ]

    async_gen = agenerator(chunks)
    client.aio.models.generate_content_stream = AsyncMock(return_value=async_gen)

    client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=10))

    events = []
    async for event in model.stream(messages):
        events.append(event)

    assert events[0] == {"messageStart": {"role": "assistant"}}
    assert events[1] == {"contentBlockStart": {"start": {}}}
    assert events[2] == {"contentBlockDelta": {"delta": {"text": "Hello"}}}
    assert events[3] == {"contentBlockDelta": {"delta": {"text": " world"}}}
    assert events[4] == {"contentBlockDelta": {"delta": {"text": "!"}}}
    assert events[5] == {"contentBlockStop": {}}
    assert events[6] == {"messageStop": {"stopReason": "end_turn"}}

    assert client.aio.models.count_tokens.await_count >= 1


@pytest.mark.asyncio
@patch("src.strands.models.gemini.genai.types.GenerateContentConfig")
@patch("src.strands.models.gemini.genai.types.ToolConfig")
@patch("src.strands.models.gemini.genai.types.FunctionCallingConfig")
async def test_stream_tool_use(
    mock_function_calling_config, mock_tool_config, mock_config_class, model, messages, tool_specs, agenerator
):
    client = model.client

    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace()

    mock_func_call = unittest.mock.MagicMock()
    mock_func_call.name = "test_tool"
    mock_func_call.args = {"input": "test value"}

    # Create a simple object that has the function_call attribute
    class MockPart:
        def __init__(self):
            self.text = None
            self.function_call = mock_func_call

    mock_part = MockPart()

    mock_content = unittest.mock.MagicMock()
    mock_content.parts = [mock_part]

    mock_candidate = unittest.mock.MagicMock()
    mock_candidate.content = mock_content
    mock_candidate.finish_reason = None  # No finish_reason for function calls initially

    mock_chunk = unittest.mock.MagicMock()
    mock_chunk.candidates = [mock_candidate]

    async_gen = agenerator([mock_chunk])
    client.aio.models.generate_content_stream = AsyncMock(return_value=async_gen)

    client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=10))
    client.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=5))

    mock_function_calling_config.return_value = MagicMock()

    mock_tool_config.return_value = MagicMock()

    events = []
    async for event in model.stream(messages, tool_specs=tool_specs):
        events.append(event)

    assert any(
        "contentBlockStart" in event
        and "toolUse" in event["contentBlockStart"].get("start", {})
        and event["contentBlockStart"]["start"]["toolUse"]["name"] == "test_tool"
        for event in events
    )

    assert any(
        "contentBlockDelta" in event
        and "toolUse" in event["contentBlockDelta"].get("delta", {})
        and "input" in event["contentBlockDelta"]["delta"]["toolUse"]
        for event in events
    )

    assert any("messageStop" in event and event["messageStop"]["stopReason"] == "tool_use" for event in events)


@pytest.mark.asyncio
@patch("src.strands.models.gemini.genai.types.GenerateContentConfig")
async def test_stream_safety_finish_reason(mock_config_class, model, messages):
    """Test handling of safety filtering through finish_reason in structured response."""
    client = model.client

    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace()

    # Mock a candidate with SAFETY finish_reason
    mock_candidate = unittest.mock.MagicMock()
    mock_candidate.finish_reason = types.FinishReason.IMAGE_SAFETY
    mock_candidate.content = None

    mock_chunk = unittest.mock.MagicMock()
    mock_chunk.text = ""
    mock_chunk.candidates = [mock_candidate]
    mock_chunk.finish_reason = None

    async def mock_generator(*args, **kwargs):
        yield mock_chunk

    client.aio.models.generate_content_stream = AsyncMock(return_value=mock_generator())

    client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=10))

    client.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=5))

    events = []
    async for event in model.stream(messages):
        events.append(event)

    assert any(
        "messageStop" in event and event["messageStop"].get("stopReason") == "content_filtered" for event in events
    )


@pytest.mark.asyncio
@patch("src.strands.models.gemini.genai.types.GenerateContentConfig")
async def test_stream_quota_error(mock_config_class, model, messages):
    """Test handling of quota/rate limit errors through HTTP status codes."""
    client = model.client

    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace()

    # Mock a quota exceeded error matching Gemini API response structure
    error_response = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota, please check your plan and billing details.",
            "status": "RESOURCE_EXHAUSTED",
        }
    }
    quota_error = genai.errors.ClientError(
        "You exceeded your current quota, please check your plan and billing details.", response_json=error_response
    )
    # The genai client sets this automatically from the response_json
    quota_error.status = "RESOURCE_EXHAUSTED"

    client.aio.models.generate_content_stream = AsyncMock(side_effect=quota_error)

    client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=10))

    client.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=5))

    exception_raised = False
    exception_message = ""

    try:
        async for _event in model.stream(messages):
            pass  # Consume events until exception is raised
    except ModelThrottledException as e:
        exception_raised = True
        exception_message = str(e)

    assert exception_raised, "ModelThrottledException was not raised"
    assert "rate limit or quota exceeded" in exception_message.lower()


@pytest.mark.asyncio
@patch("src.strands.models.gemini.genai.types.GenerateContentConfig")
async def test_stream_server_error(mock_config_class, model, messages):
    """Test handling of server errors from Gemini."""
    client = model.client

    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace()

    error_response = {"error": {"message": "Internal server error", "code": 500}}
    server_error = genai.errors.ServerError("Internal server error", response_json=error_response)

    client.aio.models.generate_content_stream = AsyncMock(side_effect=server_error)

    client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=10))

    client.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=5))

    exception_raised = False
    exception_message = ""

    try:
        async for _event in model.stream(messages):
            pass  # Consume events until exception is raised
    except ModelThrottledException as e:
        exception_raised = True
        exception_message = str(e)

    assert exception_raised, "ModelThrottledException was not raised"

    assert "Server error" in exception_message
    assert "Internal server error" in exception_message


@pytest.mark.asyncio
@patch("src.strands.models.gemini.genai.types.GenerateContentConfig")
async def test_stream_unknown_api_response_error(mock_config_class, model, messages):
    """Test handling of unparseable responses from Gemini."""
    client = model.client

    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace()

    async def mock_generator(*args, **kwargs):
        yield unittest.mock.MagicMock()
        raise genai.errors.UnknownApiResponseError("Failed to parse response")

    client.aio.models.generate_content_stream = AsyncMock(side_effect=mock_generator)

    client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=10))

    client.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=5))

    with pytest.raises(RuntimeError) as excinfo:
        async for _ in model.stream(messages):
            pass

    assert "Unparseable response from Gemini" in str(excinfo.value)


@pytest.mark.asyncio
@patch("src.strands.models.gemini.genai.types.GenerateContentConfig")
async def test_stream_general_error(mock_config_class, model, messages):
    client = model.client

    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace()

    client.aio.models.generate_content_stream = AsyncMock(side_effect=Exception("Unknown error"))

    client.aio.models.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=10))

    client.count_tokens = AsyncMock(return_value=SimpleNamespace(total_tokens=5))

    with pytest.raises(RuntimeError) as excinfo:
        async for _ in model.stream(messages):
            pass

    assert "Unexpected error streaming from Gemini: Unknown error" in str(excinfo.value)


@pytest.mark.asyncio
@patch("src.strands.models.gemini.genai.types.GenerateContentConfig")
@patch("src.strands.models.gemini.genai.types.ToolConfig")
@patch("src.strands.models.gemini.genai.types.FunctionCallingConfig")
async def test_structured_output_success(
    mock_function_calling_config, mock_tool_config, mock_config_class, model, mock_genai_client, test_output_model_cls
):
    mock_function_calling_config.return_value = MagicMock()

    mock_tool_config.return_value = MagicMock()

    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    with patch.object(model, "stream") as mock_stream:

        async def custom_stream(*args, **kwargs):
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": '{"name": "John", "age": 30}'}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}

        mock_stream.return_value = custom_stream()

        events = []
        async for event in model.structured_output(test_output_model_cls, messages):
            events.append(event)

        assert "output" in events[-1]
        assert isinstance(events[-1]["output"], test_output_model_cls)
        assert events[-1]["output"].name == "John"
        assert events[-1]["output"].age == 30


@pytest.mark.asyncio
@patch("src.strands.models.gemini.genai.types.GenerateContentConfig")
@patch("src.strands.models.gemini.genai.types.ToolConfig")
@patch("src.strands.models.gemini.genai.types.FunctionCallingConfig")
async def test_structured_output_wrong_stop_reason(
    mock_function_calling_config, mock_tool_config, mock_config_class, model, mock_genai_client, test_output_model_cls
):
    mock_function_calling_config.return_value = MagicMock()

    mock_tool_config.return_value = MagicMock()

    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    with patch.object(model, "stream") as mock_stream:

        async def custom_stream(*args, **kwargs):
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": "Some text response"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "max_tokens"}}

        mock_stream.return_value = custom_stream()

        with pytest.raises(ValueError, match='Model returned stop_reason: max_tokens instead of "end_turn"'):
            async for _ in model.structured_output(test_output_model_cls, messages):
                pass


@pytest.mark.asyncio
@patch("src.strands.models.gemini.genai.types.GenerateContentConfig")
@patch("src.strands.models.gemini.genai.types.ToolConfig")
@patch("src.strands.models.gemini.genai.types.FunctionCallingConfig")
async def test_structured_output_missing_data(
    mock_function_calling_config, mock_tool_config, mock_config_class, model, mock_genai_client, test_output_model_cls
):
    mock_function_calling_config.return_value = MagicMock()

    mock_tool_config.return_value = MagicMock()

    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    with patch.object(model, "stream") as mock_stream:

        async def custom_stream(*args, **kwargs):
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": '{"name": "John"}'}}}  # Missing age field
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}

        mock_stream.return_value = custom_stream()

        # Check that ValueError is raised when creating the model
        with pytest.raises(ValueError, match="Failed to create structured output from Gemini response"):
            async for _ in model.structured_output(test_output_model_cls, messages):
                pass
