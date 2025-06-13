import pytest
from pydantic import BaseModel

from strands import Agent
from strands.models.ollama import OllamaModel


@pytest.fixture
def model():
    return OllamaModel(host="http://localhost:11434", model_id="llama3.1:8b")


@pytest.fixture
def agent(model):
    return Agent(model=model)


def test_agent(agent):
    result = agent("Say 'hello world' with no other text")
    assert result.message["content"][0]["text"].lower() == "hello world"


def test_structured_output(agent):
    class Weather(BaseModel):
        """Extract the time and weather from the response with the exact strings."""

        time: str
        weather: str

    result = agent.structured_output(Weather, "The time is 12:00 and the weather is sunny")
    assert isinstance(result, Weather)
    assert result.time == "12:00"
    assert result.weather == "sunny"
