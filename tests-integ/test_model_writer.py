import os

import pytest

import strands
from strands import Agent
from strands.models.writer import WriterModel

if not os.getenv("WRITER_API_KEY"):
    pytest.skip("WRITER_API_KEY environment variable missing", allow_module_level=True)


@pytest.fixture
def model():
    return WriterModel(
        model="palmyra-x5",
        client_args={"api_key": os.getenv("WRITER_API_KEY", "")},
        stream_options={"include_usage": True},
    )


@pytest.fixture
def system_prompt():
    return "You are a smart assistant, that uses @ instead of all punctuation marks. It is an obligation!"


@pytest.fixture
def tools():
    @strands.tool
    def tool_time(location: str) -> str:
        """Returning time in the specific location.

        Args:
            location: Location to return time at.
        """

        return "12:00"

    @strands.tool
    def tool_weather(location: str, time: str) -> str:
        """Returning weather in the specific location and time.

        Args:
            location: Location to return weather at.
            time: Moment of time to return weather in specified location.
        """

        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def agent(model, tools, system_prompt):
    return Agent(model=model, tools=tools, system_prompt=system_prompt, load_tools_from_directory=False)


def test_agent(agent):
    response = agent("How are you?")

    assert len(response.message) > 0
    assert "@" in response.message.get("content", [])[0].get("text", "")


@pytest.mark.asyncio
async def test_async_streaming_agent(agent):
    response = agent.stream_async("How are you?")

    full_message = ""
    async for event in response:
        if delta_text := event.get("event", {}).get("contentBlockDelta", {}).get("delta", {}).get("text", ""):
            full_message += delta_text

    assert len(full_message) > 0
    assert "@" in full_message


def test_model_events(model):
    messages = [{"role": "user", "content": [{"text": "How are you?"}]}]

    response_events = {key for x in model.converse(messages) for key in x.keys()}

    assert all(
        [
            event_type in response_events
            for event_type in [
                "messageStart",
                "contentBlockStart",
                "contentBlockDelta",
                "contentBlockStop",
                "messageStop",
                "metadata",
            ]
        ]
    )


def test_agent_with_tool_calls(agent, model):
    model.update_config(model="palmyra-x4")
    agent.system_prompt = ""

    response = agent("What is the time and weather in Warsaw?")
    response_message_text = response.message.get("content", [])[0].get("text", "")

    assert len(response.message) > 0
    assert "12" in response_message_text
    assert "sunny" in response_message_text
    assert all(tool in response.metrics.tool_metrics for tool in ["tool_time", "tool_weather"])


@pytest.mark.asyncio
async def test_async_streaming_agent_with_tool_calls(agent, model):
    model.update_config(model="palmyra-x4")
    agent.system_prompt = ""

    response = agent.stream_async("What is the time and weather in Warsaw?")

    full_message = ""
    async for event in response:
        if delta_text := event.get("event", {}).get("contentBlockDelta", {}).get("delta", {}).get("text", ""):
            full_message += delta_text

    assert len(full_message) > 0
    assert "12" in full_message
    assert "sunny" in full_message
