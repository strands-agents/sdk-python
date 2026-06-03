import pytest

from strands import Agent, BedrockModel, types


@pytest.fixture
def model():
    return BedrockModel("us.anthropic.claude-haiku-4-5-20251001-v1:0")


@pytest.fixture
def agent(model):
    return Agent(model=model)


@pytest.mark.asyncio
async def test_stream_async_hello_world(agent):
    async for event in agent.stream_async("Say hello world"):
        assert isinstance(event, types.StreamEvent)

    assert isinstance(event, types.StreamEvent.Stop)


def test_app_state(agent):
    assert agent.get_app_state() == {}
    agent.set_app_state({"foo": "bar", "count": 1})
    assert agent.get_app_state() == {"foo": "bar", "count": 1}
