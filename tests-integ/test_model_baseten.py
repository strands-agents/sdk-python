import os

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.models.baseten import BasetenModel


@pytest.fixture
def model_model_apis():
    """Test with Model APIs using DeepSeek R1 model."""
    return BasetenModel(
        model_id="deepseek-ai/DeepSeek-V3-0324",
        client_args={
            "api_key": os.getenv("BASETEN_API_KEY"),
        },
    )


@pytest.fixture
def model_dedicated_deployment():
    """Test with dedicated deployment -- change this to your deployment ID when testing."""
    base_url = "https://model-232k7g23.api.baseten.co/environments/production/sync/v1"
    
    return BasetenModel(
        base_url=base_url,
        client_args={
            "api_key": os.getenv("BASETEN_API_KEY"),
        },
    )


@pytest.fixture
def tools():
    @strands.tool
    def tool_time() -> str:
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def agent_model_apis(model_model_apis, tools):
    return Agent(model=model_model_apis, tools=tools)


@pytest.fixture
def agent_dedicated(model_dedicated_deployment, tools):
    return Agent(model=model_dedicated_deployment, tools=tools)


@pytest.mark.skipif(
    "BASETEN_API_KEY" not in os.environ,
    reason="BASETEN_API_KEY environment variable missing",
)
def test_agent_model_apis(agent_model_apis):
    result = agent_model_apis("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.skipif(
    "BASETEN_API_KEY" not in os.environ or "BASETEN_DEPLOYMENT_ID" not in os.environ,
    reason="BASETEN_API_KEY or BASETEN_DEPLOYMENT_ID environment variable missing",
)
def test_agent_dedicated_deployment(agent_dedicated):
    result = agent_dedicated("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.skipif(
    "BASETEN_API_KEY" not in os.environ,
    reason="BASETEN_API_KEY environment variable missing",
)
def test_structured_output_model_apis(model_model_apis):
    class Weather(BaseModel):
        """Extracts the time and weather from the user's message with the exact strings."""

        time: str
        weather: str

    agent = Agent(model=model_model_apis)

    result = agent.structured_output(Weather, "The time is 12:00 and the weather is sunny")
    assert isinstance(result, Weather)
    assert result.time == "12:00"
    assert result.weather == "sunny"


@pytest.mark.skipif(
    "BASETEN_API_KEY" not in os.environ or "BASETEN_DEPLOYMENT_ID" not in os.environ,
    reason="BASETEN_API_KEY or BASETEN_DEPLOYMENT_ID environment variable missing",
)
def test_structured_output_dedicated_deployment(model_dedicated_deployment):
    class Weather(BaseModel):
        """Extracts the time and weather from the user's message with the exact strings."""

        time: str
        weather: str

    agent = Agent(model=model_dedicated_deployment)

    result = agent.structured_output(Weather, "The time is 12:00 and the weather is sunny")
    assert isinstance(result, Weather)
    assert result.time == "12:00"
    assert result.weather == "sunny"


@pytest.mark.skipif(
    "BASETEN_API_KEY" not in os.environ,
    reason="BASETEN_API_KEY environment variable missing",
)
def test_llama_model_model_apis():
    """Test with Llama 4 Maverick model on Model APIs."""
    model = BasetenModel(
        model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        client_args={
            "api_key": os.getenv("BASETEN_API_KEY"),
        },
    )
    
    agent = Agent(model=model)
    result = agent("Hello, how are you?")
    
    assert result.message["content"][0]["text"] is not None
    assert len(result.message["content"][0]["text"]) > 0


@pytest.mark.skipif(
    "BASETEN_API_KEY" not in os.environ,
    reason="BASETEN_API_KEY environment variable missing",
)
def test_deepseek_r1_model_apis():
    """Test with DeepSeek R1 model on Model APIs."""
    model = BasetenModel(
        model_id="deepseek-ai/DeepSeek-R1-0528",
        client_args={
            "api_key": os.getenv("BASETEN_API_KEY"),
        },
    )
    
    agent = Agent(model=model)
    result = agent("What is 2 + 2?")
    
    assert result.message["content"][0]["text"] is not None
    assert len(result.message["content"][0]["text"]) > 0


@pytest.mark.skipif(
    "BASETEN_API_KEY" not in os.environ,
    reason="BASETEN_API_KEY environment variable missing",
)
def test_llama_scout_model_apis():
    """Test with Llama 4 Scout model on Model APIs."""
    model = BasetenModel(
        model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        client_args={
            "api_key": os.getenv("BASETEN_API_KEY"),
        },
    )
    
    agent = Agent(model=model)
    result = agent("Explain quantum computing in simple terms.")
    
    assert result.message["content"][0]["text"] is not None
    assert len(result.message["content"][0]["text"]) > 0 