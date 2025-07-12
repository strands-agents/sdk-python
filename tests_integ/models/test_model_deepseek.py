import os

import pytest
from pydantic import BaseModel, Field

import strands
from strands import Agent
from strands.models.deepseek import DeepSeekModel

# these tests only run if we have the deepseek api key
pytestmark = pytest.mark.skipif(
    "DEEPSEEK_API_KEY" not in os.environ,
    reason="DEEPSEEK_API_KEY environment variable missing",
)


@pytest.fixture()
def base_model():
    return DeepSeekModel(
        api_key=os.getenv("DEEPSEEK_API_KEY"), model_id="deepseek-chat", params={"max_tokens": 2000, "temperature": 0.7}
    )


@pytest.fixture()
def tools():
    @strands.tool
    def tool_time() -> str:
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture()
def base_agent(base_model, tools):
    return Agent(model=base_model, tools=tools)


class PersonInfo(BaseModel):
    """Extract person information from text."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job or profession")


def test_basic_conversation(base_agent):
    result = base_agent("Hello, how are you today?")
    assert "content" in result.message
    assert len(result.message["content"]) > 0


def test_tool_usage(base_agent):
    result = base_agent("What is the time and weather?")
    content = result.message.get("content", [])

    # Check for tool calls
    tool_calls = [block for block in content if "toolUse" in block]
    if tool_calls:
        tool_names = [tool["toolUse"]["name"] for tool in tool_calls]
        assert "tool_time" in tool_names or "tool_weather" in tool_names

    assert result.message is not None


def test_structured_output_person(base_agent):
    result = base_agent.structured_output(
        PersonInfo, "John Smith is a 30-year-old software engineer working at a tech startup."
    )
    assert result.name == "John Smith"
    assert result.age == 30
    assert "engineer" in result.occupation.lower()


def test_calculator_integration():
    """Test calculator tool integration like ds_test.py"""
    try:
        from strands_tools import calculator
    except ImportError:
        pytest.skip("strands_tools not available")

    model = DeepSeekModel(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        model_id="deepseek-chat",
        params={"max_tokens": 2000, "temperature": 0.7},
    )

    agent = Agent(model=model, tools=[calculator])
    result = agent("What is 42 ^ 9")

    # Verify response exists
    assert result.message is not None
    content = result.message.get("content", [])
    assert len(content) > 0

    # Check for tool calls
    tool_calls = [block for block in content if "toolUse" in block]
    if tool_calls:
        for tool_call in tool_calls:
            assert tool_call["toolUse"]["name"] == "calculator"
            assert "input" in tool_call["toolUse"]


def test_multi_tool_workflow():
    """Test multi-tool workflow like ds_test.py"""
    try:
        from strands_tools import calculator, file_read, shell
    except ImportError:
        pytest.skip("strands_tools not available")

    model = DeepSeekModel(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        model_id="deepseek-chat",
        params={"max_tokens": 2000, "temperature": 0.7},
    )

    agent = Agent(model=model, tools=[calculator, file_read, shell])

    # Test 1: Calculator
    result1 = agent("What is 42 ^ 9")
    assert result1.message is not None
    content1 = result1.message.get("content", [])
    tool_calls1 = [block for block in content1 if "toolUse" in block]
    if tool_calls1:
        assert any(tool["toolUse"]["name"] == "calculator" for tool in tool_calls1)

    # Test 2: File operations
    result2 = agent("Show me the contents of a single file in this directory")
    assert result2.message is not None
    content2 = result2.message.get("content", [])
    tool_calls2 = [block for block in content2 if "toolUse" in block]
    if tool_calls2:
        tool_names = [tool["toolUse"]["name"] for tool in tool_calls2]
        assert any(name in ["file_read", "shell"] for name in tool_names)


def test_config_update(base_model):
    original_config = base_model.get_config()
    assert original_config["model_id"] == "deepseek-chat"

    base_model.update_config(model_id="deepseek-reasoner", params={"temperature": 0.5})
    updated_config = base_model.get_config()
    assert updated_config["model_id"] == "deepseek-reasoner"
    assert updated_config["params"]["temperature"] == 0.5


@pytest.mark.asyncio
async def test_streaming(base_model):
    agent = Agent(model=base_model)
    events = []
    async for event in agent.stream_async("Tell me a short fact about robots"):
        events.append(event)

    assert len(events) > 0
    assert "result" in events[-1]


@pytest.mark.asyncio
async def test_async_structured_output(base_agent):
    result = await base_agent.structured_output_async(
        PersonInfo, "Alice Johnson is a 25-year-old teacher at the local school."
    )
    assert result.name == "Alice Johnson"
    assert result.age == 25
    assert "teacher" in result.occupation.lower()
