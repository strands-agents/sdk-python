import unittest.mock

import pydantic
import pytest

import strands
from strands.models.baseten import BasetenModel


@pytest.fixture
def openai_client_cls():
    with unittest.mock.patch.object(strands.models.baseten.openai, "AsyncOpenAI") as mock_client_cls:
        yield mock_client_cls


@pytest.fixture
def openai_client(openai_client_cls):
    return openai_client_cls.return_value


@pytest.fixture
def model_id():
    return "deepseek-ai/DeepSeek-R1-0528"


@pytest.fixture
def model(openai_client, model_id):
    _ = openai_client

    return BasetenModel(model_id=model_id)


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


def test__init__model_apis(openai_client_cls, model_id):
    model = BasetenModel({"api_key": "k1"}, model_id=model_id, params={"max_tokens": 1})

    tru_config = model.get_config()
    exp_config = {"model_id": "deepseek-ai/DeepSeek-R1-0528", "params": {"max_tokens": 1}}

    assert tru_config == exp_config

    openai_client_cls.assert_called_once_with(api_key="k1", base_url="https://inference.baseten.co/v1")


def test__init__dedicated_deployment(openai_client_cls):
    base_url = "https://model-abcd1234.api.baseten.co/environments/production/sync/v1"
    
    model = BasetenModel(
        {"api_key": "k1"}, 
        model_id="abcd1234", 
        base_url=base_url,
        params={"max_tokens": 1}
    )

    tru_config = model.get_config()
    exp_config = {
        "model_id": "abcd1234", 
        "base_url": base_url,
        "params": {"max_tokens": 1}
    }

    assert tru_config == exp_config

    openai_client_cls.assert_called_once_with(api_key="k1", base_url=base_url)


def test__init__base_url_in_client_args(openai_client_cls, model_id):
    custom_base_url = "https://custom.baseten.co/v1"
    model = BasetenModel(
        {"api_key": "k1", "base_url": custom_base_url}, 
        model_id=model_id
    )

    openai_client_cls.assert_called_once_with(api_key="k1", base_url=custom_base_url)


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


def test_format_request_message_content_text():
    """Test formatting text content blocks."""
    content = {"text": "Hello, world!"}
    result = BasetenModel.format_request_message_content(content)
    assert result == {"text": "Hello, world!", "type": "text"}


def test_format_request_message_content_document():
    """Test formatting document content blocks."""
    content = {
        "document": {
            "name": "test.pdf",
            "format": "pdf",
            "source": {"bytes": b"test content"}
        }
    }
    result = BasetenModel.format_request_message_content(content)
    assert result["type"] == "file"
    assert result["file"]["filename"] == "test.pdf"


def test_format_request_message_content_image():
    """Test formatting image content blocks."""
    content = {
        "image": {
            "format": "png",
            "source": {"bytes": b"test image"}
        }
    }
    result = BasetenModel.format_request_message_content(content)
    assert result["type"] == "image_url"
    assert "image_url" in result


def test_format_request_message_content_unsupported():
    """Test handling unsupported content types."""
    content = {"unsupported": "data"}
    with pytest.raises(TypeError, match="unsupported type"):
        BasetenModel.format_request_message_content(content)


def test_format_request_messages_simple():
    """Test formatting simple messages."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    result = BasetenModel.format_request_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == [{"text": "Hello", "type": "text"}]


def test_format_request_messages_with_system_prompt():
    """Test formatting messages with system prompt."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    system_prompt = "You are a helpful assistant."
    result = BasetenModel.format_request_messages(messages, system_prompt)
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[0]["content"] == system_prompt


def test_format_request_messages_with_tool_use():
    """Test formatting messages with tool use."""
    messages = [{
        "role": "assistant", 
        "content": [
            {"text": "I'll help you"},
            {"toolUse": {"name": "calculator", "input": {"a": 1, "b": 2}, "toolUseId": "call_1"}}
        ]
    }]
    result = BasetenModel.format_request_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "tool_calls" in result[0]


def test_format_request_messages_with_tool_result():
    """Test formatting messages with tool result."""
    messages = [{
        "role": "tool", 
        "content": [
            {"toolResult": {"toolUseId": "call_1", "content": [{"json": {"result": 3}}]}}
        ]
    }]
    result = BasetenModel.format_request_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_stream_model_apis(openai_client):
    """Test streaming with Model APIs."""
    model = BasetenModel(
        {"api_key": "k1"}, 
        model_id="deepseek-ai/DeepSeek-R1-0528"
    )
    
    mock_delta = unittest.mock.Mock(content="Hello", tool_calls=None, reasoning_content=None)
    mock_event = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_usage = unittest.mock.Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    async def async_iter():
        yield mock_event
        yield unittest.mock.Mock(usage=mock_usage)

    openai_client.chat.completions.create.return_value = async_iter()

    messages = [{"role": "user", "content": [{"text": "calculate 2+2"}]}]
    response = model.stream(messages)
    tru_events = []
    async for event in response:
        tru_events.append(event)
    
    # Check that the first few events match expected format
    assert len(tru_events) > 0
    assert tru_events[0] == {"messageStart": {"role": "assistant"}}
    assert tru_events[1] == {"contentBlockStart": {"start": {}}}
    
    # Verify the API was called with correct parameters
    openai_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_stream_dedicated_deployment(openai_client):
    """Test streaming with dedicated deployment."""
    base_url = "https://model-abcd1234.api.baseten.co/environments/production/sync/v1"
    
    model = BasetenModel(
        {"api_key": "k1"}, 
        model_id="abcd1234",
        base_url=base_url
    )
    
    mock_delta = unittest.mock.Mock(content="Response", tool_calls=None, reasoning_content=None)
    mock_event = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_usage = unittest.mock.Mock(prompt_tokens=5, completion_tokens=3, total_tokens=8)

    async def async_iter():
        yield mock_event
        yield unittest.mock.Mock(usage=mock_usage)

    openai_client.chat.completions.create.return_value = async_iter()

    messages = [{"role": "user", "content": [{"text": "Test"}]}]
    response = model.stream(messages)
    
    tru_events = []
    async for event in response:
        tru_events.append(event)

    assert len(tru_events) > 0
    assert tru_events[0] == {"messageStart": {"role": "assistant"}}
    openai_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_stream_with_tools(openai_client, model):
    """Test streaming with tool specifications."""
    mock_tool_call = unittest.mock.Mock(index=0)
    mock_delta = unittest.mock.Mock(
        content="I'll calculate", 
        tool_calls=[mock_tool_call], 
        reasoning_content=None
    )
    mock_event = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="tool_calls", delta=mock_delta)])

    async def async_iter():
        yield mock_event
        yield unittest.mock.Mock()

    openai_client.chat.completions.create.return_value = async_iter()

    messages = [{"role": "user", "content": [{"text": "Calculate 2+2"}]}]
    tool_specs = [{
        "name": "calculator",
        "description": "A calculator tool",
        "inputSchema": {"json": {"type": "object", "properties": {"expression": {"type": "string"}}}}
    }]
    
    response = model.stream(messages, tool_specs)
    
    tru_events = []
    async for event in response:
        tru_events.append(event)

    assert len(tru_events) > 0
    # Verify the request included tools
    call_args = openai_client.chat.completions.create.call_args
    assert "tools" in call_args[1]


@pytest.mark.asyncio
async def test_stream_with_system_prompt(openai_client, model):
    """Test streaming with system prompt."""
    mock_delta = unittest.mock.Mock(content="Response", tool_calls=None, reasoning_content=None)
    mock_event = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])

    async def async_iter():
        yield mock_event
        yield unittest.mock.Mock()

    openai_client.chat.completions.create.return_value = async_iter()

    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    system_prompt = "You are a helpful assistant."
    
    response = model.stream(messages, system_prompt=system_prompt)
    
    tru_events = []
    async for event in response:
        tru_events.append(event)

    assert len(tru_events) > 0
    # Verify the request included system prompt
    call_args = openai_client.chat.completions.create.call_args
    messages_arg = call_args[1]["messages"]
    assert messages_arg[0]["role"] == "system"
    assert messages_arg[0]["content"] == system_prompt


@pytest.mark.asyncio
async def test_stream_empty(openai_client, model):
    mock_delta = unittest.mock.Mock(content=None, tool_calls=None, reasoning_content=None)
    mock_usage = unittest.mock.Mock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock()
    mock_event_4 = unittest.mock.Mock(usage=mock_usage)

    # Create async iterator for the response
    async def async_iter():
        for event in [mock_event_1, mock_event_2, mock_event_3, mock_event_4]:
            yield event

    openai_client.chat.completions.create.return_value = async_iter()

    messages = [{"role": "user", "content": []}]
    response = model.stream(messages)

    tru_events = []
    async for event in response:
        tru_events.append(event)

    # Check that we get the expected events
    assert len(tru_events) > 0
    assert tru_events[0] == {"messageStart": {"role": "assistant"}}
    assert tru_events[1] == {"contentBlockStart": {"start": {}}}
    
    openai_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_stream_with_empty_choices(openai_client, model):
    mock_delta = unittest.mock.Mock(content="content", tool_calls=None, reasoning_content=None)
    mock_usage = unittest.mock.Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    # Event with no choices attribute
    mock_event_1 = unittest.mock.Mock(spec=[])

    # Event with empty choices list
    mock_event_2 = unittest.mock.Mock(choices=[])

    # Valid event with content
    mock_event_3 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])

    # Event with finish reason
    mock_event_4 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])

    # Final event with usage info
    mock_event_5 = unittest.mock.Mock(usage=mock_usage)

    # Create async iterator for the response
    async def async_iter():
        for event in [mock_event_1, mock_event_2, mock_event_3, mock_event_4, mock_event_5]:
            yield event

    openai_client.chat.completions.create.return_value = async_iter()

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    response = model.stream(messages)

    tru_events = []
    async for event in response:
        tru_events.append(event)

    # Check that we get the expected events
    assert len(tru_events) > 0
    assert tru_events[0] == {"messageStart": {"role": "assistant"}}
    assert tru_events[1] == {"contentBlockStart": {"start": {}}}
    
    openai_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_structured_output(openai_client, model, test_output_model_cls):
    mock_parsed_response = test_output_model_cls(name="test", age=25)
    mock_choice = unittest.mock.Mock()
    mock_choice.message.parsed = mock_parsed_response
    mock_response = unittest.mock.Mock(choices=[mock_choice])

    openai_client.beta.chat.completions.parse.return_value = mock_response

    prompt = [{"role": "user", "content": [{"text": "test"}]}]
    result = []
    async for event in model.structured_output(test_output_model_cls, prompt):
        result.append(event)

    assert len(result) == 1
    assert result[0]["output"] == mock_parsed_response

    openai_client.beta.chat.completions.parse.assert_called_once()


@pytest.mark.asyncio
async def test_structured_output_multiple_choices(openai_client, model, test_output_model_cls):
    mock_choice_1 = unittest.mock.Mock()
    mock_choice_2 = unittest.mock.Mock()
    mock_response = unittest.mock.Mock(choices=[mock_choice_1, mock_choice_2])

    openai_client.beta.chat.completions.parse.return_value = mock_response

    prompt = [{"role": "user", "content": [{"text": "test"}]}]
    
    with pytest.raises(ValueError, match="Multiple choices found in the Baseten response."):
        async for _ in model.structured_output(test_output_model_cls, prompt):
            pass


@pytest.mark.asyncio
async def test_structured_output_no_valid_parsed(openai_client, model, test_output_model_cls):
    mock_choice = unittest.mock.Mock()
    mock_choice.message.parsed = None
    mock_response = unittest.mock.Mock(choices=[mock_choice])

    openai_client.beta.chat.completions.parse.return_value = mock_response

    prompt = [{"role": "user", "content": [{"text": "test"}]}]
    
    with pytest.raises(ValueError, match="No valid tool use or tool use input was found in the Baseten response."):
        async for _ in model.structured_output(test_output_model_cls, prompt):
            pass 