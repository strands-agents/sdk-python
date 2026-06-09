import pytest

from strands import Agent, BedrockModel, http_request, notebook, tool, types


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


@pytest.fixture
def http_request_agent(model):
    return Agent(model=model, vended_tools=[http_request])


@pytest.fixture
def notebook_agent(model):
    return Agent(model=model, vended_tools=[notebook])


@pytest.mark.asyncio
async def test_decorated_tool_invocation(agent):
    result = await agent.invoke_async("What is the weather in Seattle?")
    assert "72" in str(result)


@pytest.mark.asyncio
async def test_http_request_tool_invocation(http_request_agent):
    tools_called: list[str] = []
    last_message: types.Message | None = None
    async for event in http_request_agent.stream_async(
        "Use the http_request tool to GET https://httpbin.org/base64/c3RyYW5kcw== "
        "and tell me the exact decoded text from the response body."
    ):
        if isinstance(event, types.StreamEvent.BeforeToolCall):
            tools_called.append(event.value.tool_use.name)
        elif isinstance(event, types.StreamEvent.MessageAdded):
            last_message = event.value.message

    assert "http_request" in tools_called, f"tools called: {tools_called}"

    assert last_message is not None
    text_blocks = [b.payload.text for b in last_message.content if b.tag == "text"]
    # /base64/c3RyYW5kcw== returns the literal body "strands".
    assert any("strands" in t.lower() for t in text_blocks)


@pytest.mark.asyncio
async def test_notebook_tool_invocation(notebook_agent):
    tools_called: list[str] = []
    last_message: types.Message | None = None
    async for event in notebook_agent.stream_async(
        "Use the notebook tool to create a notebook named 'todo' with the line 'buy milk'. "
        "Then read it back and tell me what's in it."
    ):
        if isinstance(event, types.StreamEvent.BeforeToolCall):
            tools_called.append(event.value.tool_use.name)
        elif isinstance(event, types.StreamEvent.MessageAdded):
            last_message = event.value.message

    assert "notebook" in tools_called, f"tools called: {tools_called}"

    notebooks = (await notebook_agent.get_app_state()).get("notebooks")
    assert notebooks == {"default": "", "todo": "buy milk"}, f"got: {notebooks!r}"

    assert last_message is not None
    text_blocks = [b.payload.text for b in last_message.content if b.tag == "text"]
    assert any("milk" in t.lower() for t in text_blocks)
