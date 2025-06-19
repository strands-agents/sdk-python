import base64
import unittest.mock

import pytest

import strands
from strands.models.stability import OutputFormat, StabilityAiImageModel, StylePreset


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
