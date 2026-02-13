import os

import pydantic
import pytest
from google import genai

import strands
from strands import Agent
from strands.models.gemini import GeminiModel
from tests_integ.models import providers

# these tests only run if we have the gemini api key
pytestmark = providers.gemini.mark


@pytest.fixture
def model():
    return GeminiModel(
        client_args={"api_key": os.getenv("GOOGLE_API_KEY")},
        model_id="gemini-2.5-flash",
        params={"temperature": 0.15},  # Lower temperature for consistent test behavior
    )


@pytest.fixture
def gemini_tool_model():
    return GeminiModel(
        client_args={"api_key": os.getenv("GOOGLE_API_KEY")},
        model_id="gemini-2.5-flash",
        params={"temperature": 0.15},  # Lower temperature for consistent test behavior
        gemini_tools=[genai.types.Tool(code_execution=genai.types.ToolCodeExecution())],
    )


@pytest.fixture
def tools():
    @strands.tool
    def tool_time(city: str) -> str:
        return "12:00"

    @strands.tool
    def tool_weather(city: str) -> str:
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def system_prompt():
    return "You are a helpful AI assistant."


@pytest.fixture
def assistant_agent(model, system_prompt):
    return Agent(model=model, system_prompt=system_prompt)


@pytest.fixture
def tool_agent(model, tools, system_prompt):
    return Agent(model=model, tools=tools, system_prompt=system_prompt)


@pytest.fixture
def weather():
    class Weather(pydantic.BaseModel):
        """Extracts the time and weather from the user's message with the exact strings."""

        time: str
        weather: str

    return Weather(time="12:00", weather="sunny")


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


@pytest.fixture(scope="module")
def test_image_path(request):
    return request.config.rootpath / "tests_integ" / "test_image.png"


def test_agent_invoke(tool_agent):
    result = tool_agent("What is the current time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_invoke_async(tool_agent):
    result = await tool_agent.invoke_async("What is the current time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_stream_async(tool_agent):
    stream = tool_agent.stream_async("What is the current time and weather in New York?")
    async for event in stream:
        _ = event

    result = event["result"]
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


def test_agent_invoke_multiturn(assistant_agent):
    assistant_agent("What color is the sky?")
    assistant_agent("What color is lava?")
    result = assistant_agent("What was the answer to my first question?")
    text = result.message["content"][0]["text"].lower()

    assert "blue" in text


def test_agent_invoke_image_input(assistant_agent, yellow_img):
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
    result = assistant_agent(content)
    text = result.message["content"][0]["text"].lower()

    assert "yellow" in text


def test_agent_invoke_document_input(assistant_agent, letter_pdf):
    content = [
        {"text": "summarize this document"},
        {"document": {"format": "pdf", "source": {"bytes": letter_pdf}}},
    ]
    result = assistant_agent(content)
    text = result.message["content"][0]["text"].lower()

    assert "shareholder" in text


def test_agent_structured_output(assistant_agent, weather):
    tru_weather = assistant_agent.structured_output(type(weather), "The time is 12:00 and the weather is sunny")
    exp_weather = weather
    assert tru_weather == exp_weather


@pytest.mark.asyncio
async def test_agent_structured_output_async(assistant_agent, weather):
    tru_weather = await assistant_agent.structured_output_async(
        type(weather), "The time is 12:00 and the weather is sunny"
    )
    exp_weather = weather
    assert tru_weather == exp_weather


def test_agent_structured_output_image_input(assistant_agent, yellow_img, yellow_color):
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
    tru_color = assistant_agent.structured_output(type(yellow_color), content)
    exp_color = yellow_color
    assert tru_color == exp_color


def test_agent_with_gemini_code_execution_tool(gemini_tool_model):
    system_prompt = "Generate and run code for all calculations"
    agent = Agent(model=gemini_tool_model, system_prompt=system_prompt)
    # sample prompt taken from https://ai.google.dev/gemini-api/docs/code-execution
    result_turn1 = agent(
        "What is the sum of the first 50 prime numbers? Generate and run code for the calculation, "
        "and make sure you get all 50."
    )

    # NOTE: We don't verify tool history because built-in tools are currently represented in message history
    assert "5117" in str(result_turn1)

    result_turn2 = agent("Summarize that into a single number")
    assert "5117" in str(result_turn2)


def test_agent_with_reasoning_content(model, assistant_agent):
    """Test that reasoning content is captured in message history."""

    model.update_config(
        params={
            "thinking_config": {
                "thinking_budget": 1024,
                "include_thoughts": True,
            },
        },
    )

    result = assistant_agent("Think about what 2+2 is")
    assert "reasoningContent" in result.message["content"][0]
    assert result.message["content"][0]["reasoningContent"]["reasoningText"]["text"]
