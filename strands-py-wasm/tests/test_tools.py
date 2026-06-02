import pytest

from strands import Agent, BedrockModel, tool


@pytest.fixture
def model():
    return BedrockModel("us.anthropic.claude-haiku-4-5-20251001-v1:0")


@pytest.fixture
def weather_tool():
    @tool
    def get_weather(city: str) -> str:
        """Return the current weather for a city."""
        return f"It is 72F and sunny in {city}."

    return get_weather


@pytest.fixture
def agent(model, weather_tool):
    return Agent(model=model, tools=[weather_tool])


@pytest.mark.asyncio
async def test_decorated_tool_invocation(agent):
    result = await agent.invoke_async("What is the weather in Seattle?")
    assert "72" in str(result)
