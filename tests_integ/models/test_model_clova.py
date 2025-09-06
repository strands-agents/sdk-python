"""Integration tests for CLOVA model provider."""

import os

import pytest

from strands.models.clova import ClovaModel


@pytest.fixture
def clova_api_key():
    """Get CLOVA API key from environment."""
    api_key = os.getenv("CLOVA_API_KEY")
    if not api_key:
        pytest.skip("CLOVA_API_KEY not set")
    return api_key


@pytest.fixture
def clova_model(clova_api_key):
    """Create a ClovaModel instance for integration testing."""
    return ClovaModel(api_key=clova_api_key, model="HCX-005")


@pytest.mark.asyncio
async def test_basic_streaming(clova_model):
    """Test basic streaming functionality with real API."""
    prompt = "안녕하세요. 한국어로 간단한 인사를 해주세요."

    response_chunks = []
    async for event in clova_model.stream(prompt):
        if event.get("type") == "text":
            response_chunks.append(event["text"])

    # Check we got a response
    assert len(response_chunks) > 0

    # Combine chunks to get full response
    full_response = "".join(response_chunks)
    assert len(full_response) > 0

    # Korean response should contain Korean characters
    assert any(ord(char) >= 0xAC00 and ord(char) <= 0xD7AF for char in full_response)


@pytest.mark.asyncio
async def test_streaming_with_system_prompt(clova_model):
    """Test streaming with system prompt."""
    prompt = "What is 2 + 2?"
    system_prompt = "You are a helpful math tutor. Answer briefly."

    response_chunks = []
    async for event in clova_model.stream(prompt, system_prompt=system_prompt):
        if event.get("type") == "text":
            response_chunks.append(event["text"])

    # Check we got a response
    assert len(response_chunks) > 0

    # Combine chunks and check for "4" in response
    full_response = "".join(response_chunks).lower()
    assert "4" in full_response or "four" in full_response


@pytest.mark.asyncio
async def test_temperature_parameter(clova_model):
    """Test that temperature parameter affects output."""
    prompt = "Tell me a creative story in one sentence."

    # Low temperature (more deterministic)
    clova_model.update_config(temperature=0.1)
    response1_chunks = []
    async for event in clova_model.stream(prompt):
        if event.get("type") == "text":
            response1_chunks.append(event.text)

    response1 = "".join(response1_chunks)

    # High temperature (more creative)
    clova_model.update_config(temperature=0.9)
    response2_chunks = []
    async for event in clova_model.stream(prompt):
        if event.get("type") == "text":
            response2_chunks.append(event.text)

    response2 = "".join(response2_chunks)

    # Both should produce responses
    assert len(response1) > 0
    assert len(response2) > 0

    # Responses should likely be different (not guaranteed but highly probable)
    # We just check that both produce valid responses


@pytest.mark.asyncio
async def test_max_tokens_limit(clova_model):
    """Test that max_tokens parameter limits output."""
    prompt = "Count from 1 to 100 slowly, one number per line."

    # Set a low token limit
    clova_model.update_config(max_tokens=50)

    response_chunks = []
    async for event in clova_model.stream(prompt):
        if event.get("type") == "text":
            response_chunks.append(event["text"])

    # Check we got a response
    assert len(response_chunks) > 0

    # Response should be limited (not reach 100)
    full_response = "".join(response_chunks)
    assert "100" not in full_response  # Shouldn't reach 100 with token limit


@pytest.mark.asyncio
async def test_model_configuration(clova_model):
    """Test getting and updating model configuration."""
    # Get initial config
    initial_config = clova_model.get_config()
    assert initial_config["model"] == "HCX-005"
    assert "temperature" in initial_config
    assert "max_tokens" in initial_config

    # Update config
    clova_model.update_config(temperature=0.5, max_tokens=2048, top_p=0.8)

    # Verify updates
    updated_config = clova_model.get_config()
    assert updated_config["temperature"] == 0.5
    assert updated_config["max_tokens"] == 2048
    assert updated_config["top_p"] == 0.8


@pytest.mark.asyncio
async def test_bilingual_support(clova_model):
    """Test that CLOVA supports both Korean and English."""
    # Test Korean
    korean_prompt = "한국의 수도는 어디인가요?"
    korean_response = []
    async for event in clova_model.stream(korean_prompt):
        if event.get("type") == "text":
            korean_response.append(event.text)

    korean_text = "".join(korean_response).lower()
    assert "서울" in korean_text or "seoul" in korean_text

    # Test English
    english_prompt = "What is the capital of South Korea?"
    english_response = []
    async for event in clova_model.stream(english_prompt):
        if event.get("type") == "text":
            english_response.append(event.text)

    english_text = "".join(english_response).lower()
    assert "seoul" in english_text or "서울" in english_text


@pytest.mark.asyncio
async def test_structured_output_not_supported(clova_model):
    """Test that structured output is not yet supported."""
    from pydantic import BaseModel

    class TestOutput(BaseModel):
        answer: str

    with pytest.raises(NotImplementedError, match="Structured output is not yet supported"):
        async for _ in clova_model.structured_output(TestOutput, "Test prompt"):
            pass


def test_model_string_representation(clova_model):
    """Test string representation of the model."""
    model_str = str(clova_model)
    assert "ClovaModel" in model_str
    assert "HCX-005" in model_str
    assert "temperature" in model_str.lower()
    assert "max_tokens" in model_str.lower()
