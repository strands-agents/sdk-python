import pytest
import requests
from pydantic import BaseModel

from strands import Agent
from strands.models.ollama import OllamaModel


@pytest.fixture
def model():
    return OllamaModel(host="http://localhost:11434", model_id="llama3.3:70b")


@pytest.fixture
def agent(model):
    return Agent(model=model)


@pytest.mark.skipif(
    not requests.get("http://localhost:11434/api/health").ok,
    reason="Local Ollama endpoint not available at localhost:11434",
)
def test_agent(agent):
    result = agent("Say 'hello world' with no other text")
    assert isinstance(result, str)


@pytest.mark.skipif(
    not requests.get("http://localhost:11434/api/health").ok,
    reason="Local Ollama endpoint not available at localhost:11434",
)
def test_structured_output(agent):
    class Weather(BaseModel):
        """Extract the time and weather from the response with the exact strings."""

        time: str
        weather: str

    result = agent.structured_output(Weather, "The time is 12:00 and the weather is sunny")
    assert isinstance(result, Weather)
    assert result.time == "12:00"
    assert result.weather == "sunny"
