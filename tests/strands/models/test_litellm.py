import unittest.mock

import pytest

import strands
from strands.models.litellm import LiteLLMModel


@pytest.fixture
def litellm_client_cls():
    with unittest.mock.patch.object(strands.models.litellm.litellm, "LiteLLM") as mock_client_cls:
        yield mock_client_cls


@pytest.fixture
def litellm_client(litellm_client_cls):
    return litellm_client_cls.return_value


@pytest.fixture
def model_id():
    return "m1"


@pytest.fixture
def model(litellm_client, model_id):
    _ = litellm_client

    return LiteLLMModel(model_id=model_id)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "s1"


def test__init__(litellm_client_cls, model_id):
    model = LiteLLMModel({"api_key": "k1"}, model_id=model_id, params={"max_tokens": 1})

    tru_config = model.get_config()
    exp_config = {"model_id": "m1", "params": {"max_tokens": 1}}

    assert tru_config == exp_config

    litellm_client_cls.assert_called_once_with(api_key="k1")


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


@pytest.mark.parametrize(
    "content, exp_result",
    [
        (
            {
                "reasoningContent": {
                    "reasoningText": {
                        "signature": "reasoning_signature",
                        "text": "reasoning_text",
                    },
                },
            },
            {
                "signature": "reasoning_signature",
                "thinking": "reasoning_text",
                "type": "thinking",
            },
        ),
        (
            {
                "video": {
                    "source": {"bytes": "base64encodedvideo"},
                },
            },
            {
                "type": "video_url",
                "video_url": {
                    "detail": "auto",
                    "url": "base64encodedvideo",
                },
            },
        ),
        (
            {"text": "hello"},
            {"type": "text", "text": "hello"},
        ),
    ],
)
def test_format_request_message_content(content, exp_result):
    tru_result = LiteLLMModel.format_request_message_content(content)
    assert tru_result == exp_result
