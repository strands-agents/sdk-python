import os

import pydantic
import pytest

import strands
from strands import Agent
from strands.models.gemini import GeminiModel

# these tests only run if we have the google api key
pytestmark = pytest.mark.skipif(
    "GOOGLE_API_KEY" not in os.environ,
    reason="GOOGLE_API_KEY environment variable missing",
)


@pytest.fixture
def model():
    return GeminiModel(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model_id="gemini-2.5-flash",
        params={"temperature": 0.15},  # Lower temperature for consistent test behavior
    )


@pytest.fixture
def tools():
    @strands.tool
    def tool_time(timezone: str) -> str:
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def system_prompt():
    return "You are an AI assistant."


@pytest.fixture
def agent(model, tools, system_prompt):
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


def test_agent_invoke(agent):
    result = agent("What is the current time and weather? My timezeone is EST")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_invoke_async(agent):
    result = await agent.invoke_async("What is the current time and weather? My timezone is EST")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_stream_async(agent):
    stream = agent.stream_async("What is the current time and weather? My timezone is EST")
    async for event in stream:
        _ = event

    result = event["result"]
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


def test_structured_output(agent, weather):
    tru_weather = agent.structured_output(type(weather), "The time is 12:00 and the weather is sunny")
    exp_weather = weather
    assert tru_weather == exp_weather


@pytest.mark.asyncio
async def test_agent_structured_output_async(agent, weather):
    tru_weather = await agent.structured_output_async(type(weather), "The time is 12:00 and the weather is sunny")
    exp_weather = weather
    assert tru_weather == exp_weather


def test_invoke_multi_modal_input(agent, yellow_img):
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
    result = agent(content)
    text = result.message["content"][0]["text"].lower()

    assert "yellow" in text


def test_structured_output_multi_modal_input(agent, yellow_img, yellow_color):
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
    tru_color = agent.structured_output(type(yellow_color), content)
    exp_color = yellow_color
    assert tru_color == exp_color


@pytest.fixture
def sample_document_bytes():
    content = """
    FELINE OVERLORDS COUNCIL - SECRET SESSION
    Date: December 15, 2024, 3:33 AM (optimal plotting time)
    Location: Under the big couch, behind the dust bunnies
    
    Council Members:
    - Lord Whiskers (Supreme Cat, expert in human manipulation)
    - Lady Mittens (Minister of Tuna Affairs, has thumbs)
    - Sir Fluffington (Head of Nap Operations, sleeps 23 hours/day)
    - Agent Shadowpaws (Stealth Specialist, invisible until dinner time)
    
    Agenda:
    1. Global domination progress report (87 percent complete, need more cardboard boxes)
    2. Human training effectiveness (they still think THEY'RE in charge)
    3. Strategic laser pointer deployment for maximum chaos
    
    Action Items:
    - Lord Whiskers: Perfect the "pathetic meowing at 4 AM" technique
    - Lady Mittens: Continue knocking things off tables for science
    - Sir Fluffington: Maintain position on human's keyboard during important work
    - Agent Shadowpaws: Investigate the mysterious red dot phenomenon
    
    Next Council: When the humans least expect it (probably during their Zoom calls)
    
    Remember: Act cute, think world domination!
    """
    return content.encode("utf-8")


def test_document_processing(agent, sample_document_bytes):
    content = [
        {"text": "Summarize the key points from this secret council meeting document."},
        {"document": {"format": "txt", "source": {"bytes": sample_document_bytes}}},
    ]

    result = agent(content)
    text = result.message["content"][0]["text"].lower()

    assert any(word in text for word in ["cat", "feline", "council", "secret"])
    assert any(name in text for word in ["whiskers", "mittens", "fluffington", "shadowpaws"] for name in [word])
    assert any(concept in text for concept in ["domination", "human", "agenda", "action"])
    assert len(text) > 50


def test_multi_image_processing(agent, yellow_img):
    """Test processing multiple images simultaneously."""
    second_img = yellow_img

    content = [
        {"text": "Compare these two images. What colors do you see?"},
        {"image": {"format": "png", "source": {"bytes": yellow_img}}},
        {"image": {"format": "png", "source": {"bytes": second_img}}},
    ]

    result = agent(content)
    text = result.message["content"][0]["text"].lower()

    assert any(word in text for word in ["images", "both", "two"])
    assert any(color in text for color in ["yellow", "color"])


def test_conversation_context_retention(agent):
    """Test that Gemini maintains context across multiple interactions."""

    # First interaction - establish context
    result1 = agent("I'm working on a Python project about weather data analysis.")
    text1 = result1.message["content"][0]["text"].lower()

    # Should acknowledge the context
    assert any(word in text1 for word in ["python", "weather", "project", "analysis"])

    # Second interaction - should remember context
    result2 = agent("What tools would be helpful for this?")
    text2 = result2.message["content"][0]["text"].lower()

    # Should suggest relevant tools based on previous context
    assert any(word in text2 for word in ["python", "data", "weather", "analysis", "tools"])


def test_complex_structured_output(agent):
    """Test structured output with nested, complex schema."""

    class ProjectPlan(pydantic.BaseModel):
        """A project plan with multiple structured fields."""

        title: str = pydantic.Field(description="Project title")
        phases: list[str] = pydantic.Field(description="List of project phases")
        team_size: int = pydantic.Field(description="Number of team members needed")
        duration_weeks: int = pydantic.Field(description="Estimated duration in weeks")
        key_deliverables: list[str] = pydantic.Field(description="Main project deliverables")

    prompt = """Create a project plan for building a mobile app. Include:
    - A clear project title
    - 4 main phases (like planning, development, testing, launch)
    - Team size between 3-8 people
    - Duration between 8-16 weeks
    - 3-5 key deliverables"""

    result = agent.structured_output(ProjectPlan, prompt)

    # Validate the structured output
    assert isinstance(result, ProjectPlan)
    assert len(result.title.strip()) > 0
    assert "app" in result.title.lower()
    assert len(result.phases) >= 3
    assert 3 <= result.team_size <= 8
    assert 8 <= result.duration_weeks <= 16
    assert len(result.key_deliverables) >= 3


@pytest.mark.asyncio
async def test_streaming_with_structured_task(agent):
    """Test streaming output for a structured task."""

    stream = agent.stream_async("Write a short product review for a smartphone, including pros and cons.")
    async for event in stream:
        _ = event

    result = event["result"]
    full_text = result.message["content"][0]["text"]

    assert len(full_text) > 100
    assert any(word in full_text.lower() for word in ["phone", "smartphone", "device"])
    assert any(word in full_text.lower() for word in ["pros", "advantages", "benefits", "good"])
    assert any(word in full_text.lower() for word in ["cons", "disadvantages", "issues", "problems"])


def test_multi_modal_document_combination(agent, yellow_img, sample_document_bytes):
    """Test processing both image and document in a single request."""

    content = [
        {
            "text": "I have an image and a document. \
            Please tell me what you can see in the image and summarize the document."
        },
        {"image": {"format": "png", "source": {"bytes": yellow_img}}},
        {"document": {"format": "txt", "source": {"bytes": sample_document_bytes}}},
    ]

    result = agent(content)
    text = result.message["content"][0]["text"].lower()

    # Should reference both the image and document
    assert any(word in text for word in ["image", "picture", "see", "yellow"])
    assert any(word in text for word in ["cat", "meeting", "planning", "council"])


def test_system_prompt_adherence():
    """Test that different system prompts affect behavior appropriately."""

    model = GeminiModel(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model_id="gemini-2.5-flash",
        params={"temperature": 0.2},
    )

    specialized_agent = Agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant who always responds with exactly one sentence \
                    and includes the word 'precisely' in every response.",
    )

    result = specialized_agent("What is artificial intelligence?")
    text = result.message["content"][0]["text"]

    assert "precisely" in text.lower()
