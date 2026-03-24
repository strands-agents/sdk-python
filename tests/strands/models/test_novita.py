import os
import unittest.mock

import pytest

import strands
from strands.models.novita import NovitaModel, NOVITA_API_KEY_ENV_VAR, NOVITA_BASE_URL, NOVITA_DEFAULT_MODEL_ID


@pytest.fixture
def novita_client():
    with unittest.mock.patch.object(strands.models.novita.OpenAIModel, "__init__", return_value=None) as mock_init:
        yield mock_init


@pytest.fixture
def model_id():
    return "moonshotai/kimi-k2.5"


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "You are a helpful assistant."


def test__init__with_default_config(monkeypatch, novita_client):
    """Test initialization with default configuration."""
    monkeypatch.setenv(NOVITA_API_KEY_ENV_VAR, "test-api-key")

    NovitaModel()

    # Verify that OpenAIModel.__init__ was called with Novita-specific client_args
    call_args = novita_client.call_args
    assert call_args is not None

    # Check client_args contains Novita base_url and api_key
    client_args = call_args.kwargs.get("client_args", {})
    assert client_args.get("base_url") == NOVITA_BASE_URL
    assert client_args.get("api_key") == "test-api-key"


def test__init__with_custom_api_key(novita_client):
    """Test initialization with custom API key."""
    NovitaModel(api_key="custom-api-key")

    call_args = novita_client.call_args
    client_args = call_args.kwargs.get("client_args", {})
    assert client_args.get("api_key") == "custom-api-key"


def test__init__with_custom_base_url(novita_client):
    """Test initialization with custom base URL."""
    NovitaModel(api_key="test-key", base_url="https://custom.novita.ai/v1")

    call_args = novita_client.call_args
    client_args = call_args.kwargs.get("client_args", {})
    assert client_args.get("base_url") == "https://custom.novita.ai/v1"


def test__init__with_model_id(novita_client):
    """Test initialization with custom model ID."""
    NovitaModel(api_key="test-key", model_id="zai-org/glm-5")

    call_args = novita_client.call_args
    model_config = {k: v for k, v in call_args.kwargs.items() if k not in ["client", "client_args"]}
    assert model_config.get("model_id") == "zai-org/glm-5"


def test__init__default_model_id(novita_client):
    """Test that default model ID is set when not provided."""
    NovitaModel(api_key="test-key")

    call_args = novita_client.call_args
    model_config = {k: v for k, v in call_args.kwargs.items() if k not in ["client", "client_args"]}
    assert model_config.get("model_id") == NOVITA_DEFAULT_MODEL_ID


def test__init__with_params(novita_client):
    """Test initialization with model parameters."""
    NovitaModel(api_key="test-key", params={"max_tokens": 100, "temperature": 0.7})

    call_args = novita_client.call_args
    model_config = {k: v for k, v in call_args.kwargs.items() if k not in ["client", "client_args"]}
    assert model_config.get("params") == {"max_tokens": 100, "temperature": 0.7}


def test__init__client_args_merged(novita_client):
    """Test that custom client_args are merged with Novita defaults."""
    NovitaModel(api_key="test-key", client_args={"timeout": 30.0})

    call_args = novita_client.call_args
    client_args = call_args.kwargs.get("client_args", {})
    assert client_args.get("base_url") == NOVITA_BASE_URL  # Novita default
    assert client_args.get("timeout") == 30.0  # Custom arg


def test__init__env_var_api_key(monkeypatch, novita_client):
    """Test that API key is read from environment variable."""
    monkeypatch.setenv(NOVITA_API_KEY_ENV_VAR, "env-var-api-key")

    NovitaModel()

    call_args = novita_client.call_args
    client_args = call_args.kwargs.get("client_args", {})
    assert client_args.get("api_key") == "env-var-api-key"


def test__init__explicit_api_key_overrides_env(monkeypatch, novita_client):
    """Test that explicit API key overrides environment variable."""
    monkeypatch.setenv(NOVITA_API_KEY_ENV_VAR, "env-var-api-key")

    NovitaModel(api_key="explicit-api-key")

    call_args = novita_client.call_args
    client_args = call_args.kwargs.get("client_args", {})
    assert client_args.get("api_key") == "explicit-api-key"


def test__init__no_api_key(novita_client):
    """Test initialization without API key (OpenAI client will handle the error)."""
    # This should not raise an error at init time - the OpenAI client will
    # raise an error when a request is made if no API key is available
    NovitaModel()

    call_args = novita_client.call_args
    client_args = call_args.kwargs.get("client_args", {})
    # API key should not be in client_args
    assert "api_key" not in client_args or client_args.get("api_key") is None
    # But base_url should still be set
    assert client_args.get("base_url") == NOVITA_BASE_URL


def test_get_config():
    """Test get_config returns the model configuration."""
    with unittest.mock.patch.object(strands.models.novita.OpenAIModel, "__init__", return_value=None):
        model = NovitaModel(api_key="test-key", model_id="moonshotai/kimi-k2.5", params={"max_tokens": 100})
        model.config = {"model_id": "moonshotai/kimi-k2.5", "params": {"max_tokens": 100}}

        config = model.get_config()
        assert config["model_id"] == "moonshotai/kimi-k2.5"
        assert config["params"] == {"max_tokens": 100}
