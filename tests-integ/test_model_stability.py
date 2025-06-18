import os

import pytest

from strands import Agent
from strands.models.stability import OutputFormat, StabilityAiImageModel


@pytest.fixture
def model_id(request):
    return request.param


@pytest.fixture
def model(model_id):
    return StabilityAiImageModel(
        api_key=os.getenv("STABILITY_API_KEY"),  # Use the API key loaded from .env
        model_id=model_id,
        aspect_ratio="16:9",
        output_format=OutputFormat.PNG,
    )


@pytest.fixture
def agent(model):
    return Agent(model=model)


@pytest.mark.skipif(
    "STABILITY_API_KEY" not in os.environ,
    reason="STABILITY_API_KEY environment variable missing",
)
@pytest.mark.parametrize(
    "model_id",
    [
        "stability.stable-image-core-v1:1",
        "stability.stable-image-ultra-v1:1",
        "stability.sd3-5-large-v1:0",
    ],
    indirect=True,
)
def test_agent(agent):
    result = agent("dark high contrast render of a psychedelic tree of life illuminating dust in a mystical cave.")

    # Initialize variables
    image_data = None
    image_format = None

    # Find image content
    for content in result.message.get("content", []):
        if isinstance(content, dict) and "image" in content:
            image_data = content["image"]["source"]["bytes"]
            image_format = content["image"]["format"]
            break

    # Verify we found an image
    assert image_data is not None, "No image data found in the response"
    assert image_format is not None, "No image format found in the response"

    # Verify image data is not empty
    assert len(image_data) > 0, "Image data should not be empty"

    # Verify image format is PNG
    assert image_format == "png", f"Expected image format to be 'png', got '{image_format}'"
