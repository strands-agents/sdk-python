import time
import uuid

import pydantic
import pytest

import strands
from strands import Agent
from strands.models import BedrockModel
from strands.types.content import ContentBlock


@pytest.fixture
def system_prompt():
    return "You are an AI assistant that uses & instead of ."


@pytest.fixture
def streaming_model():
    return BedrockModel(
        streaming=True,
    )


@pytest.fixture
def non_streaming_model():
    return BedrockModel(
        streaming=False,
    )


@pytest.fixture
def streaming_agent(streaming_model, system_prompt):
    return Agent(
        model=streaming_model,
        system_prompt=system_prompt,
        load_tools_from_directory=False,
    )


@pytest.fixture
def non_streaming_agent(non_streaming_model, system_prompt):
    return Agent(
        model=non_streaming_model,
        system_prompt=system_prompt,
        load_tools_from_directory=False,
    )


@pytest.fixture
def yellow_color():
    class Color(pydantic.BaseModel):
        """Describes a color."""

        name: str

        @pydantic.field_validator("name", mode="after")
        @classmethod
        def lower(_, value):
            return value.lower()

    return Color(name="yellow")


def test_streaming_agent(streaming_agent):
    """Test agent with streaming model."""
    result = streaming_agent("Hello!")

    assert len(str(result)) > 0


def test_non_streaming_agent(non_streaming_agent):
    """Test agent with non-streaming model."""
    result = non_streaming_agent("Hello!")

    assert len(str(result)) > 0


@pytest.mark.asyncio
async def test_streaming_model_events(streaming_model, alist):
    """Test streaming model events."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]

    # Call stream and collect events
    events = await alist(streaming_model.stream(messages))

    # Verify basic structure of events
    assert any("messageStart" in event for event in events)
    assert any("contentBlockDelta" in event for event in events)
    assert any("messageStop" in event for event in events)


@pytest.mark.asyncio
async def test_non_streaming_model_events(non_streaming_model, alist):
    """Test non-streaming model events."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]

    # Call stream and collect events
    events = await alist(non_streaming_model.stream(messages))

    # Verify basic structure of events
    assert any("messageStart" in event for event in events)
    assert any("contentBlockDelta" in event for event in events)
    assert any("messageStop" in event for event in events)


def test_tool_use_streaming(streaming_model):
    """Test tool use with streaming model."""

    tool_was_called = False

    @strands.tool
    def calculator(expression: str) -> float:
        """Calculate the result of a mathematical expression."""

        nonlocal tool_was_called
        tool_was_called = True
        return eval(expression)

    agent = Agent(model=streaming_model, tools=[calculator], load_tools_from_directory=False)
    result = agent("What is 123 + 456?")

    # Print the full message content for debugging
    print("\nFull message content:")
    import json

    print(json.dumps(result.message["content"], indent=2))

    assert tool_was_called


def test_tool_use_non_streaming(non_streaming_model):
    """Test tool use with non-streaming model."""

    tool_was_called = False

    @strands.tool
    def calculator(expression: str) -> float:
        """Calculate the result of a mathematical expression."""

        nonlocal tool_was_called
        tool_was_called = True
        return eval(expression)

    agent = Agent(model=non_streaming_model, tools=[calculator], load_tools_from_directory=False)
    agent("What is 123 + 456?")

    assert tool_was_called


def test_structured_output_streaming(streaming_model):
    """Test structured output with streaming model."""

    class Weather(pydantic.BaseModel):
        time: str
        weather: str

    agent = Agent(model=streaming_model)

    result = agent.structured_output(Weather, "The time is 12:00 and the weather is sunny")
    assert isinstance(result, Weather)
    assert result.time == "12:00"
    assert result.weather == "sunny"


def test_structured_output_non_streaming(non_streaming_model):
    """Test structured output with non-streaming model."""

    class Weather(pydantic.BaseModel):
        time: str
        weather: str

    agent = Agent(model=non_streaming_model)

    result = agent.structured_output(Weather, "The time is 12:00 and the weather is sunny")
    assert isinstance(result, Weather)
    assert result.time == "12:00"
    assert result.weather == "sunny"


def test_invoke_multi_modal_input(streaming_agent, yellow_img):
    content = [
        {"text": "what is in this image"},
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": yellow_img,
                },
            },
        },
    ]
    result = streaming_agent(content)
    text = result.message["content"][0]["text"].lower()

    assert "yellow" in text


def test_document_citations(non_streaming_agent, letter_pdf):
    content: list[ContentBlock] = [
        {
            "document": {
                "name": "letter to shareholders",
                "source": {"bytes": letter_pdf},
                "citations": {"enabled": True},
                "context": "This is a letter to shareholders",
                "format": "pdf",
            },
        },
        {"text": "What does the document say about artificial intelligence? Use citations to back up your answer."},
    ]
    non_streaming_agent(content)

    assert any("citationsContent" in content for content in non_streaming_agent.messages[-1]["content"])

    # Validate message structure is valid in multi-turn
    non_streaming_agent("What is your favorite part?")


def test_document_citations_streaming(streaming_agent, letter_pdf):
    content: list[ContentBlock] = [
        {
            "document": {
                "name": "letter to shareholders",
                "source": {"bytes": letter_pdf},
                "citations": {"enabled": True},
                "context": "This is a letter to shareholders",
                "format": "pdf",
            },
        },
        {"text": "What does the document say about artificial intelligence? Use citations to back up your answer."},
    ]
    streaming_agent(content)

    assert any("citationsContent" in content for content in streaming_agent.messages[-1]["content"])

    # Validate message structure is valid in multi-turn
    streaming_agent("What is your favorite part?")


def test_structured_output_multi_modal_input(streaming_agent, yellow_img, yellow_color):
    content = [
        {"text": "Is this image red, blue, or yellow?"},
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": yellow_img,
                },
            },
        },
    ]
    tru_color = streaming_agent.structured_output(type(yellow_color), content)
    exp_color = yellow_color
    assert tru_color == exp_color


def test_redacted_content_handling():
    """Test redactedContent handling with thinking mode."""
    bedrock_model = BedrockModel(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        additional_request_fields={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 2000,
            }
        },
    )

    agent = Agent(name="test_redact", model=bedrock_model)
    # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#example-working-with-redacted-thinking-blocks
    result = agent(
        "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
    )

    assert "reasoningContent" in result.message["content"][0]
    assert "redactedContent" in result.message["content"][0]["reasoningContent"]
    assert isinstance(result.message["content"][0]["reasoningContent"]["redactedContent"], bytes)


def test_reasoning_content_in_messages_with_thinking_disabled():
    """Test that messages with reasoningContent are accepted when thinking is explicitly disabled."""
    # First, get a real reasoning response with thinking enabled
    thinking_model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        additional_request_fields={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024,
            }
        },
    )
    agent_with_thinking = Agent(model=thinking_model)
    result_with_thinking = agent_with_thinking("What is 2+2?")

    # Verify we got reasoning content
    assert "reasoningContent" in result_with_thinking.message["content"][0]

    # Now create a model with thinking disabled and use the messages from the thinking session
    disabled_model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        additional_request_fields={
            "thinking": {
                "type": "disabled",
            }
        },
    )

    # Use the conversation history that includes reasoning content
    messages = agent_with_thinking.messages

    agent_disabled = Agent(model=disabled_model, messages=messages)
    result = agent_disabled("What about 3+3?")

    assert result.stop_reason == "end_turn"


def test_multi_prompt_system_content():
    """Test multi-prompt system content blocks."""
    system_prompt_content = [
        {"text": "You are a helpful assistant."},
        {"text": "Always be concise."},
        {"text": "End responses with 'Done.'"},
    ]

    agent = Agent(system_prompt=system_prompt_content, load_tools_from_directory=False)
    # just verifying there is no failure
    agent("Hello!")


def test_prompt_caching_with_5m_ttl(streaming_model):
    """Test prompt caching with 5 minute TTL and verify cache metrics.

    This test verifies:
    1. First call creates cache (cacheWriteInputTokens > 0)
    2. Second call reads from cache (cacheReadInputTokens > 0)
    """
    # Use unique identifier to avoid cache conflicts between test runs
    unique_id = str(uuid.uuid4())
    # Minimum 1024 tokens required for caching
    large_context = f"Background information for test {unique_id}: " + ("This is important context. " * 200)

    system_prompt_with_cache = [
        {"text": large_context},
        {"cachePoint": {"type": "default", "ttl": "5m"}},
        {"text": "You are a helpful assistant."},
    ]

    agent = Agent(
        model=streaming_model,
        system_prompt=system_prompt_with_cache,
        load_tools_from_directory=False,
    )

    # First call should create the cache (cache write)
    result1 = agent("What is 2+2?")
    assert len(str(result1)) > 0

    # Verify cache write occurred on first call
    assert result1.metrics.accumulated_usage.get("cacheWriteInputTokens", 0) > 0, (
        "Expected cacheWriteInputTokens > 0 on first call"
    )

    # Second call should use the cached content (cache read)
    result2 = agent("What is 3+3?")
    assert len(str(result2)) > 0

    # Verify cache read occurred on second call
    assert result2.metrics.accumulated_usage.get("cacheReadInputTokens", 0) > 0, (
        "Expected cacheReadInputTokens > 0 on second call"
    )


def test_prompt_caching_with_1h_ttl(non_streaming_model):
    """Test prompt caching with 1 hour TTL and verify cache metrics.

    Uses unique content per test run to avoid cache conflicts with concurrent CI runs.
    Even with 1hr TTL, unique content ensures cache entries don't interfere across tests.
    """
    # Use timestamp to ensure unique content per test run (avoids CI conflicts)
    unique_id = str(int(time.time() * 1000000))  # microsecond timestamp
    # Minimum 1024 tokens required for caching
    large_context = f"Background information for test {unique_id}: " + ("This is important context. " * 200)

    system_prompt_with_cache = [
        {"text": large_context},
        {"cachePoint": {"type": "default", "ttl": "1h"}},
        {"text": "You are a helpful assistant."},
    ]

    agent = Agent(
        model=non_streaming_model,
        system_prompt=system_prompt_with_cache,
        load_tools_from_directory=False,
    )

    # First call should create the cache
    result1 = agent("What is 2+2?")
    assert len(str(result1)) > 0

    # Verify cache write occurred
    assert result1.metrics.accumulated_usage.get("cacheWriteInputTokens", 0) > 0, (
        "Expected cacheWriteInputTokens > 0 on first call with 1h TTL"
    )

    # Second call should use the cached content
    result2 = agent("What is 3+3?")
    assert len(str(result2)) > 0

    # Verify cache read occurred
    assert result2.metrics.accumulated_usage.get("cacheReadInputTokens", 0) > 0, (
        "Expected cacheReadInputTokens > 0 on second call with 1h TTL"
    )


def test_prompt_caching_with_ttl_in_messages(streaming_model):
    """Test prompt caching with TTL in message content and verify cache metrics."""
    agent = Agent(model=streaming_model, load_tools_from_directory=False)

    unique_id = str(uuid.uuid4())
    # Large content block to cache (minimum 1024 tokens)
    large_text = f"Important context for test {unique_id}: " + ("This is critical information. " * 200)

    content_with_cache = [
        {"text": large_text},
        {"cachePoint": {"type": "default", "ttl": "5m"}},
        {"text": "Based on the context above, what is 5+5?"},
    ]

    # First call creates cache
    result1 = agent(content_with_cache)
    assert len(str(result1)) > 0

    # Verify cache write in message content
    assert result1.metrics.accumulated_usage.get("cacheWriteInputTokens", 0) > 0, (
        "Expected cacheWriteInputTokens > 0 when caching message content"
    )

    # Subsequent call should use cache
    result2 = agent("What about 10+10?")
    assert len(str(result2)) > 0

    # Verify cache read on subsequent call
    assert result2.metrics.accumulated_usage.get("cacheReadInputTokens", 0) > 0, (
        "Expected cacheReadInputTokens > 0 on subsequent call"
    )


def test_prompt_caching_backward_compatibility_no_ttl(non_streaming_model):
    """Test that prompt caching works without TTL (backward compatibility).

    Verifies that cache points work correctly when TTL is not specified,
    maintaining backward compatibility with existing code.
    """
    unique_id = str(uuid.uuid4())
    large_context = f"Background information for test {unique_id}: " + ("This is important context. " * 200)

    system_prompt_with_cache = [
        {"text": large_context},
        {"cachePoint": {"type": "default"}},  # No TTL specified
        {"text": "You are a helpful assistant."},
    ]

    agent = Agent(
        model=non_streaming_model,
        system_prompt=system_prompt_with_cache,
        load_tools_from_directory=False,
    )

    result = agent("Hello!")
    assert len(str(result)) > 0

    # Verify cache write occurred even without TTL
    assert result.metrics.accumulated_usage.get("cacheWriteInputTokens", 0) > 0, (
        "Expected cacheWriteInputTokens > 0 even without TTL specified"
    )
