import base64
import io
import unittest.mock

import pytest
from PIL import Image

import strands

# Import the StabilityAI exceptions
from strands.models._stabilityaiclient import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NetworkError,
    OutputFormat,
    PayloadTooLargeError,
    RateLimitError,
    StabilityAiError,
    StylePreset,
    ValidationError,
)
from strands.models._stabilityaiclient import (
    ContentModerationError as StabilityContentModerationError,
)
from strands.models.stability import StabilityAiImageModel
from strands.types.exceptions import (
    ContentModerationException,
    EventLoopException,
    ModelAuthenticationException,
    ModelServiceException,
    ModelThrottledException,
    ModelValidationException,
)


@pytest.fixture
def stability_client_cls():
    with unittest.mock.patch.object(strands.models.stability, "StabilityAiClient") as mock_client_cls:
        yield mock_client_cls


@pytest.fixture
def stability_client(stability_client_cls):
    return stability_client_cls.return_value


@pytest.fixture
def model_id():
    return "stability.stable-image-ultra-v1:1"


@pytest.fixture
def model(stability_client, model_id):
    _ = stability_client
    return StabilityAiImageModel(api_key="test_key", model_id=model_id)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "a beautiful sunset over mountains"}]}]


def test__init__(stability_client_cls, model_id):
    model = StabilityAiImageModel(
        api_key="test_key",
        model_id=model_id,
        aspect_ratio="16:9",
        output_format=OutputFormat.JPEG,
        style_preset=StylePreset.PHOTOGRAPHIC,
    )

    tru_config = model.get_config()
    exp_config = {
        "model_id": model_id,
        "aspect_ratio": "16:9",
        "output_format": OutputFormat.JPEG,
        "style_preset": StylePreset.PHOTOGRAPHIC,
    }

    assert tru_config == exp_config
    stability_client_cls.assert_called_once_with(api_key="test_key")


def test__init__with_string_enums(stability_client_cls, model_id):
    model = StabilityAiImageModel(
        api_key="test_key",
        model_id=model_id,
        output_format="jpeg",
        style_preset="photographic",
    )

    tru_config = model.get_config()
    exp_config = {"model_id": model_id, "output_format": OutputFormat.JPEG, "style_preset": StylePreset.PHOTOGRAPHIC}

    assert tru_config == exp_config


def test__init__with_invalid_output_format():
    with pytest.raises(ValueError) as exc_info:
        StabilityAiImageModel(
            api_key="test_key",
            model_id="stability.stable-image-core-v1:1",
            output_format="invalid",
        )
    assert "output_format must be one of:" in str(exc_info.value)


def test__init__with_invalid_style_preset():
    with pytest.raises(ValueError) as exc_info:
        StabilityAiImageModel(
            api_key="test_key",
            model_id="stability.stable-image-core-v1:1",
            style_preset="invalid",
        )
    assert "style_preset must be one of:" in str(exc_info.value)


def test_update_config(model, model_id):
    model.update_config(
        model_id=model_id,
        aspect_ratio="16:9",
        output_format=OutputFormat.JPEG,
    )

    tru_config = model.get_config()
    exp_config = {"model_id": model_id, "aspect_ratio": "16:9", "output_format": OutputFormat.JPEG}

    assert tru_config == exp_config


def test_format_request(model, messages):
    request = model.format_request(messages)

    exp_request = {
        "prompt": "a beautiful sunset over mountains",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "mode": "text-to-image",
        "style_preset": "photographic",
    }

    assert request == exp_request


def test_format_request_with_optional_params(model, messages):
    model.update_config(
        seed=12345,
        image="base64_encoded_image",
        strength=0.5,
    )
    request = model.format_request(messages)

    exp_request = {
        "prompt": "a beautiful sunset over mountains",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "style_preset": "photographic",
        "seed": 12345,
        "image": "base64_encoded_image",
        "mode": "text-to-image",
        "strength": 0.5,
    }

    assert request == exp_request


def test_format_request_with_cfg_scale_sd35(stability_client, messages):
    """Test that cfg_scale is included in request for SD3.5 model."""
    model = StabilityAiImageModel(
        api_key="test_key",
        model_id="stability.sd3-5-large-v1:0",
        cfg_scale=8,
    )

    request = model.format_request(messages)

    exp_request = {
        "prompt": "a beautiful sunset over mountains",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "mode": "text-to-image",
        "style_preset": "photographic",
        "cfg_scale": 8,
    }

    assert request == exp_request


def test_format_request_with_cfg_scale_non_sd35(stability_client, messages):
    """Test that cfg_scale is NOT included in request for non-SD3.5 models."""
    model = StabilityAiImageModel(
        api_key="test_key",
        model_id="stability.stable-image-core-v1:1",
        cfg_scale=8,  # This should be ignored
    )

    request = model.format_request(messages)

    exp_request = {
        "prompt": "a beautiful sunset over mountains",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "mode": "text-to-image",
        "style_preset": "photographic",
        # Note: cfg_scale is not passed in
    }

    assert request == exp_request
    assert "cfg_scale" not in request


def test_update_config_change_model_id(model, messages):
    """Test updating config to change model_id."""
    # Initial model uses stability.stable-image-ultra-v1:1 from fixture
    initial_config = model.get_config()
    assert initial_config["model_id"] == "stability.stable-image-ultra-v1:1"

    # Update to different model
    model.update_config(
        model_id="stability.stable-image-core-v1:1",
        aspect_ratio="16:9",
    )

    updated_config = model.get_config()
    exp_config = {
        "model_id": "stability.stable-image-core-v1:1",
        "aspect_ratio": "16:9",
        "output_format": OutputFormat.PNG,
    }

    assert updated_config == exp_config

    # Verify the model uses the new model_id in requests
    request = model.format_request(messages)
    assert request["aspect_ratio"] == "16:9"


def test_stream_image_to_image_mode(model, stability_client):
    """Test successful image-to-image generation"""
    # Create a 64x64 white PNG image
    white_image = Image.new("RGB", (64, 64), color="white")

    # Convert to PNG bytes
    img_buffer = io.BytesIO()
    white_image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    # Base64 encode the image
    input_image_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # Mock response with a different image
    mock_response = {
        "image": base64.b64encode(b"fake_transformed_image_data").decode("utf-8"),
        "finish_reason": "SUCCESS",
    }
    stability_client.generate_image_json.return_value = mock_response

    request = {
        "prompt": "transform this image into a sunset scene",
        "image": input_image_base64,
        "mode": "image-to-image",
        "strength": 0.75,
        "aspect_ratio": "1:1",
        "output_format": "png",
        "style_preset": "photographic",
    }

    events = list(model.stream(request))

    # Verify the stream events
    assert len(events) == 5
    assert events[0] == {"chunk_type": "message_start"}
    assert events[1] == {"chunk_type": "content_start", "data_type": "text"}
    assert events[2]["chunk_type"] == "content_block_delta"
    assert events[2]["data_type"] == "image"
    assert events[2]["data"] == mock_response["image"]
    assert events[3] == {"chunk_type": "content_stop", "data_type": "text"}
    assert events[4] == {"chunk_type": "message_stop", "data": "SUCCESS"}

    # Verify the client was called with the correct parameters including image-to-image mode
    stability_client.generate_image_json.assert_called_once_with(model.config["model_id"], **request)

    # Verify the request included the base64 image and image-to-image mode
    call_args = stability_client.generate_image_json.call_args[1]
    assert call_args["mode"] == "image-to-image"
    assert call_args["image"] == input_image_base64
    assert call_args["strength"] == 0.75


def test_format_request_no_user_message():
    model = StabilityAiImageModel(api_key="test_key", model_id="stability.stable-image-core-v1:1")
    messages = [{"role": "assistant", "content": [{"text": "test"}]}]

    with pytest.raises(ValueError) as exc_info:
        model.format_request(messages)
    assert "No user message found in the conversation" in str(exc_info.value)


def test_format_chunk_message_start():
    model = StabilityAiImageModel(api_key="test_key", model_id="stability.stable-image-core-v1:1")
    event = {"chunk_type": "message_start"}

    chunk = model.format_chunk(event)
    assert chunk == {"messageStart": {"role": "assistant"}}


def test_format_chunk_content_start():
    model = StabilityAiImageModel(api_key="test_key", model_id="stability.stable-image-core-v1:1")
    event = {"chunk_type": "content_start"}

    chunk = model.format_chunk(event)
    assert chunk == {"contentBlockStart": {"start": {}}}


def test_format_chunk_content_block_delta():
    model = StabilityAiImageModel(api_key="test_key", model_id="stability.stable-image-core-v1:1")
    raw_image_data = b"raw_image_data"
    base64_encoded_data = base64.b64encode(raw_image_data)
    event = {"chunk_type": "content_block_delta", "data": base64_encoded_data}

    chunk = model.format_chunk(event)
    assert chunk == {"contentBlockDelta": {"delta": {"image": {"format": "png", "source": {"bytes": raw_image_data}}}}}


def test_format_chunk_content_stop():
    model = StabilityAiImageModel(api_key="test_key", model_id="stability.stable-image-core-v1:1")
    event = {"chunk_type": "content_stop"}

    chunk = model.format_chunk(event)
    assert chunk == {"contentBlockStop": {}}


def test_format_chunk_message_stop():
    model = StabilityAiImageModel(api_key="test_key", model_id="stability.stable-image-core-v1:1")
    event = {"chunk_type": "message_stop", "data": "stop"}

    chunk = model.format_chunk(event)
    assert chunk == {"messageStop": {"stopReason": "stop"}}


def test_format_chunk_unknown_type():
    model = StabilityAiImageModel(api_key="test_key", model_id="stability.stable-image-core-v1:1")
    event = {"chunk_type": "unknown"}

    with pytest.raises(RuntimeError) as exc_info:
        model.format_chunk(event)
    assert "unknown type" in str(exc_info.value)


# Test exceptions for stream method


def test_stream_authentication_error(model, stability_client):
    """Test that AuthenticationError is converted to ModelAuthenticationException."""
    stability_client.generate_image_json.side_effect = AuthenticationError(
        "Invalid API key", response_data={"error": "unauthorized"}
    )

    request = {
        "prompt": "test prompt",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "style_preset": "photographic",
        "mode": "text-to-image",
    }

    with pytest.raises(ModelAuthenticationException) as exc_info:
        list(model.stream(request))

    assert "Invalid API key" in str(exc_info.value)


def test_stream_content_moderation_error(model, stability_client):
    """Test that ContentModerationError is converted to ContentModerationException."""
    stability_client.generate_image_json.side_effect = StabilityContentModerationError(
        "Content flagged by moderation", response_data={"error": "content_policy_violation"}
    )

    request = {
        "prompt": "an unclothed woman on the beach",
        "seed": 7,
        "aspect_ratio": "1:1",
        "output_format": "png",
        "style_preset": "photographic",
        "mode": "text-to-image",
    }

    with pytest.raises(ContentModerationException) as exc_info:
        list(model.stream(request))

    assert "Content flagged by moderation" in str(exc_info.value)


def test_stream_validation_error(model, stability_client):
    """Test that ValidationError is converted to ModelValidationException."""
    stability_client.generate_image_json.side_effect = ValidationError(
        "Prompt exceeds maximum length", response_data={"error": "validation_error", "field": "prompt"}
    )

    request = {
        "prompt": "a" * 10001,
        "aspect_ratio": "1:1",
        "output_format": "png",
        "style_preset": "photographic",
        "mode": "text-to-image",
    }

    with pytest.raises(ModelValidationException) as exc_info:
        list(model.stream(request))

    assert "Prompt exceeds maximum length" in str(exc_info.value)


def test_stream_bad_request_error(model, stability_client):
    """Test that BadRequestError is converted to ModelValidationException."""
    stability_client.generate_image_json.side_effect = BadRequestError(
        "Invalid aspect ratio", response_data={"error": "bad_request"}
    )

    request = {
        "prompt": "test prompt",
        "aspect_ratio": "invalid",
        "output_format": "png",
        "style_preset": "photographic",
        "mode": "text-to-image",
    }

    with pytest.raises(ModelValidationException) as exc_info:
        list(model.stream(request))

    assert "Invalid aspect ratio" in str(exc_info.value)


def test_stream_payload_too_large_error(model, stability_client):
    """Test that PayloadTooLargeError is converted to ModelValidationException."""
    stability_client.generate_image_json.side_effect = PayloadTooLargeError("Request size exceeds 10MB limit")

    request = {
        "prompt": "test prompt",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "style_preset": "photographic",
        "mode": "text-to-image",
    }

    with pytest.raises(ModelValidationException) as exc_info:
        list(model.stream(request))

    assert "Request size exceeds 10MB limit" in str(exc_info.value)


def test_stream_rate_limit_error(model, stability_client):
    """Test that RateLimitError is converted to ModelThrottledException."""
    stability_client.generate_image_json.side_effect = RateLimitError(
        "Rate limit exceeded. Please retry after 60 seconds.", response_data={"retry_after": 60}
    )

    request = {
        "prompt": "test prompt",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "style_preset": "photographic",
        "mode": "text-to-image",
    }

    with pytest.raises(ModelThrottledException) as exc_info:
        list(model.stream(request))

    assert "Rate limit exceeded" in str(exc_info.value)


def test_stream_internal_server_error(model, stability_client):
    """Test that InternalServerError is converted to ModelServiceException with is_transient=True."""
    stability_client.generate_image_json.side_effect = InternalServerError("Service temporarily unavailable")

    request = {
        "prompt": "test prompt",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "style_preset": "photographic",
        "mode": "text-to-image",
    }

    with pytest.raises(ModelServiceException) as exc_info:
        list(model.stream(request))

    assert "Service temporarily unavailable" in str(exc_info.value)
    assert exc_info.value.is_transient is True


def test_stream_network_error(model, stability_client):
    """Test that NetworkError is converted to EventLoopException."""
    original_error = ConnectionError("Connection timed out")
    stability_client.generate_image_json.side_effect = NetworkError(
        "Network request failed", original_error=original_error
    )

    request = {
        "prompt": "test prompt",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "style_preset": "photographic",
        "mode": "text-to-image",
    }

    with pytest.raises(EventLoopException) as exc_info:
        list(model.stream(request))

    assert exc_info.value.original_exception == original_error
    assert exc_info.value.request_state == request


def test_stream_generic_stability_error(model, stability_client):
    """Test that generic StabilityAiError is converted to ModelServiceException with is_transient=False."""
    stability_client.generate_image_json.side_effect = StabilityAiError("Unexpected error occurred", status_code=418)

    request = {
        "prompt": "test prompt",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "style_preset": "photographic",
        "mode": "text-to-image",
    }

    with pytest.raises(ModelServiceException) as exc_info:
        list(model.stream(request))

    assert "Unexpected error occurred" in str(exc_info.value)
    assert exc_info.value.is_transient is False


def test_stream_success(model, stability_client):
    """Test successful image generation stream."""
    mock_response = {"image": base64.b64encode(b"fake_image_data").decode("utf-8"), "finish_reason": "SUCCESS"}
    stability_client.generate_image_json.return_value = mock_response

    request = {
        "prompt": "a beautiful sunset",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "style_preset": "photographic",
        "mode": "text-to-image",
    }

    events = list(model.stream(request))

    assert len(events) == 5
    assert events[0] == {"chunk_type": "message_start"}
    assert events[1] == {"chunk_type": "content_start", "data_type": "text"}
    assert events[2]["chunk_type"] == "content_block_delta"
    assert events[2]["data_type"] == "image"
    assert events[2]["data"] == mock_response["image"]
    assert events[3] == {"chunk_type": "content_stop", "data_type": "text"}
    assert events[4] == {"chunk_type": "message_stop", "data": "SUCCESS"}

    # Verify the client was called with the correct parameters
    stability_client.generate_image_json.assert_called_once_with(model.config["model_id"], **request)


def test_stream_with_invalid_api_key_string():
    """Test that invalid API key string raises ModelAuthenticationException."""
    model = StabilityAiImageModel(
        api_key="12345",  # Invalid API key (valid str format but not authorized)
        model_id="stability.stable-image-core-v1:1",
    )

    # Mock the client to raise AuthenticationError when called
    with unittest.mock.patch.object(model.client, "generate_image_json") as mock_generate:
        mock_generate.side_effect = AuthenticationError("Invalid API key", response_data={"error": "unauthorized"})

        request = {
            "prompt": "test prompt",
            "aspect_ratio": "1:1",
            "output_format": "png",
            "style_preset": "photographic",
            "mode": "text-to-image",
        }

        with pytest.raises(ModelAuthenticationException) as exc_info:
            list(model.stream(request))

        assert "Invalid API key" in str(exc_info.value)
