"""Tests for streaming.py with structured output support."""

import unittest.mock

import pytest
from pydantic import BaseModel

import strands.event_loop.streaming
from strands.tools.structured_output.structured_output_tool import StructuredOutputTool
from strands.types._events import TypedEvent


class SampleModel(BaseModel):
    """Sample model for structured output."""

    name: str
    age: int


@pytest.fixture(autouse=True)
def moto_autouse(moto_env, moto_mock_aws):
    _ = moto_env
    _ = moto_mock_aws


@pytest.mark.asyncio
async def test_stream_messages_with_tool_choice(agenerator, alist):
    """Test stream_messages with tool_choice parameter for structured output."""
    mock_model = unittest.mock.MagicMock()
    mock_model.stream.return_value = agenerator(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "test-123", "name": "SampleModel"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"name": "test", "age": 25}'}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                    "metrics": {"latencyMs": 100},
                }
            },
        ]
    )

    # Create a structured output tool and get its spec
    structured_tool = StructuredOutputTool(SampleModel)
    tool_spec = structured_tool.tool_spec
    tool_choice = {"tool": {"name": "SampleModel"}}

    stream = strands.event_loop.streaming.stream_messages(
        mock_model,
        system_prompt="test prompt",
        messages=[{"role": "user", "content": [{"text": "Generate a test model"}]}],
        tool_specs=[tool_spec],
        tool_choice=tool_choice,
    )

    tru_events = await alist(stream)

    # Verify the model.stream was called with tool_choice
    mock_model.stream.assert_called_with(
        [{"role": "user", "content": [{"text": "Generate a test model"}]}],
        [tool_spec],
        "test prompt",
        tool_choice=tool_choice,
    )

    # Verify we get the expected events
    assert len(tru_events) > 0

    # Find the stop event
    stop_event = None
    for event in tru_events:
        if isinstance(event, dict) and "stop" in event:
            stop_event = event
            break

    assert stop_event is not None
    assert stop_event["stop"][0] == "tool_use"

    # Ensure that we're getting typed events
    non_typed_events = [event for event in tru_events if not isinstance(event, TypedEvent)]
    assert non_typed_events == []


@pytest.mark.asyncio
async def test_stream_messages_without_tool_choice(agenerator, alist):
    """Test stream_messages without tool_choice parameter (default behavior)."""
    mock_model = unittest.mock.MagicMock()
    mock_model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "Regular response"}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )

    stream = strands.event_loop.streaming.stream_messages(
        mock_model,
        system_prompt="test prompt",
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        tool_specs=None,
        tool_choice=None,  # Explicitly passing None
    )

    tru_events = await alist(stream)

    # Verify the model.stream was called with tool_choice=None
    mock_model.stream.assert_called_with(
        [{"role": "user", "content": [{"text": "Hello"}]}],
        None,
        "test prompt",
        tool_choice=None,
    )

    # Verify we get the expected events
    assert len(tru_events) > 0

    # Find the stop event
    stop_event = None
    for event in tru_events:
        if isinstance(event, dict) and "stop" in event:
            stop_event = event
            break

    assert stop_event is not None
    assert stop_event["stop"][0] == "end_turn"

    # Ensure that we're getting typed events
    non_typed_events = [event for event in tru_events if not isinstance(event, TypedEvent)]
    assert non_typed_events == []


@pytest.mark.asyncio
async def test_stream_messages_with_forced_structured_output(agenerator, alist):
    """Test stream_messages with forced structured output tool."""
    mock_model = unittest.mock.MagicMock()

    # Simulate a response with tool use
    mock_model.stream.return_value = agenerator(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "SampleModel"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"name": "Alice", "age": 30}'}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30},
                    "metrics": {"latencyMs": 150},
                }
            },
        ]
    )

    # Create a structured output tool and get its spec
    structured_tool = StructuredOutputTool(SampleModel)
    tool_spec = structured_tool.tool_spec

    # Force the use of the structured output tool
    tool_choice = {"tool": {"name": "SampleModel"}}

    stream = strands.event_loop.streaming.stream_messages(
        mock_model,
        system_prompt="Extract user information",
        messages=[{"role": "user", "content": [{"text": "Alice is 30 years old"}]}],
        tool_specs=[tool_spec],
        tool_choice=tool_choice,
    )

    tru_events = await alist(stream)

    # Verify the model.stream was called with the forced tool choice
    mock_model.stream.assert_called_with(
        [{"role": "user", "content": [{"text": "Alice is 30 years old"}]}],
        [tool_spec],
        "Extract user information",
        tool_choice=tool_choice,
    )

    # Verify we get the expected events
    assert len(tru_events) > 0

    # Find the stop event and verify it contains the extracted data
    stop_event = None
    for event in tru_events:
        if isinstance(event, dict) and "stop" in event:
            stop_event = event
            break

    assert stop_event is not None
    stop_reason, message, usage, metrics = stop_event["stop"]

    assert stop_reason == "tool_use"
    assert message["role"] == "assistant"
    assert len(message["content"]) > 0

    # Check that the tool use contains the expected data
    tool_use_content = None
    for content in message["content"]:
        if "toolUse" in content:
            tool_use_content = content["toolUse"]
            break

    assert tool_use_content is not None
    assert tool_use_content["name"] == "SampleModel"
    assert tool_use_content["input"] == {"name": "Alice", "age": 30}

    # Verify usage metrics
    assert usage["inputTokens"] == 20
    assert usage["outputTokens"] == 10
    assert usage["totalTokens"] == 30
    assert metrics["latencyMs"] == 150


@pytest.mark.asyncio
async def test_stream_messages_with_multiple_tools_and_choice(agenerator, alist):
    """Test stream_messages with multiple tools and specific tool choice."""
    mock_model = unittest.mock.MagicMock()

    class PersonModel(BaseModel):
        name: str
        age: int

    class CompanyModel(BaseModel):
        name: str
        employees: int

    # Simulate choosing the PersonModel tool
    mock_model.stream.return_value = agenerator(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "456", "name": "PersonModel"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"name": "Bob", "age": 25}'}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )

    # Create multiple structured output tools
    person_tool = StructuredOutputTool(PersonModel)
    company_tool = StructuredOutputTool(CompanyModel)

    tool_specs = [person_tool.tool_spec, company_tool.tool_spec]
    tool_choice = {"tool": {"name": "PersonModel"}}

    stream = strands.event_loop.streaming.stream_messages(
        mock_model,
        system_prompt="Extract information",
        messages=[{"role": "user", "content": [{"text": "Bob is 25 years old"}]}],
        tool_specs=tool_specs,
        tool_choice=tool_choice,
    )

    tru_events = await alist(stream)

    # Verify the model.stream was called with the correct tool choice
    mock_model.stream.assert_called_with(
        [{"role": "user", "content": [{"text": "Bob is 25 years old"}]}],
        tool_specs,
        "Extract information",
        tool_choice=tool_choice,
    )

    # Verify the correct tool was used
    stop_event = None
    for event in tru_events:
        if isinstance(event, dict) and "stop" in event:
            stop_event = event
            break

    assert stop_event is not None
    _, message, _, _ = stop_event["stop"]

    # Check that PersonModel was used, not CompanyModel
    tool_use_content = None
    for content in message["content"]:
        if "toolUse" in content:
            tool_use_content = content["toolUse"]
            break

    assert tool_use_content is not None
    assert tool_use_content["name"] == "PersonModel"
    assert tool_use_content["input"] == {"name": "Bob", "age": 25}


@pytest.mark.asyncio
async def test_stream_messages_with_auto_tool_choice(agenerator, alist):
    """Test stream_messages with 'auto' tool choice."""
    mock_model = unittest.mock.MagicMock()

    # Simulate model choosing a tool automatically
    mock_model.stream.return_value = agenerator(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "I'll extract the information."}}},
            {"contentBlockStop": {}},
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "789", "name": "SampleModel"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"name": "Charlie", "age": 35}'}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )

    structured_tool = StructuredOutputTool(SampleModel)
    tool_spec = structured_tool.tool_spec
    tool_choice = "auto"  # Let the model decide

    stream = strands.event_loop.streaming.stream_messages(
        mock_model,
        system_prompt="Extract if needed",
        messages=[{"role": "user", "content": [{"text": "Charlie is 35"}]}],
        tool_specs=[tool_spec],
        tool_choice=tool_choice,
    )

    tru_events = await alist(stream)

    # Verify the model.stream was called with 'auto' tool choice
    mock_model.stream.assert_called_with(
        [{"role": "user", "content": [{"text": "Charlie is 35"}]}],
        [tool_spec],
        "Extract if needed",
        tool_choice="auto",
    )

    # Verify we get both text and tool use in the response
    stop_event = None
    for event in tru_events:
        if isinstance(event, dict) and "stop" in event:
            stop_event = event
            break

    assert stop_event is not None
    _, message, _, _ = stop_event["stop"]

    # Check that we have both text and tool use content
    has_text = False
    has_tool_use = False

    for content in message["content"]:
        if "text" in content:
            has_text = True
            assert content["text"] == "I'll extract the information."
        elif "toolUse" in content:
            has_tool_use = True
            assert content["toolUse"]["name"] == "SampleModel"
            assert content["toolUse"]["input"] == {"name": "Charlie", "age": 35}

    assert has_text
    assert has_tool_use
