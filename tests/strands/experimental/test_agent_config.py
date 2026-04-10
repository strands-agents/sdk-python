"""Tests for experimental config_to_agent function."""

import json
import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands.experimental import config_to_agent
from strands.experimental.agent_config import (
    PROVIDER_MAP,
    _create_model_from_dict,
    _resolve_env_vars,
)

# =============================================================================
# Backward compatibility tests (existing)
# =============================================================================


def test_config_to_agent_with_dict():
    """Test config_to_agent can be created with dict config."""
    config = {"model": "test-model"}
    agent = config_to_agent(config)
    assert agent.model.config["model_id"] == "test-model"


def test_config_to_agent_with_system_prompt():
    """Test config_to_agent handles system prompt correctly."""
    config = {"model": "test-model", "prompt": "Test prompt"}
    agent = config_to_agent(config)
    assert agent.system_prompt == "Test prompt"


def test_config_to_agent_with_tools_list():
    """Test config_to_agent handles tools list without failing."""
    # Use a simple test that doesn't require actual tool loading
    config = {"model": "test-model", "tools": []}
    agent = config_to_agent(config)
    assert agent.model.config["model_id"] == "test-model"


def test_config_to_agent_with_kwargs_override():
    """Test that kwargs can override config values."""
    config = {"model": "test-model", "prompt": "Config prompt"}
    agent = config_to_agent(config, system_prompt="Override prompt")
    assert agent.system_prompt == "Override prompt"


def test_config_to_agent_file_prefix_required():
    """Test that file paths without file:// prefix work."""
    import json
    import tempfile

    config_data = {"model": "test-model"}
    temp_path = ""

    # We need to create files like this for windows compatibility
    try:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            temp_path = f.name

        agent = config_to_agent(temp_path)
        assert agent.model.config["model_id"] == "test-model"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_config_to_agent_file_prefix_valid():
    """Test that file:// prefix is properly handled."""
    config_data = {"model": "test-model", "prompt": "Test prompt"}
    temp_path = ""

    try:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            temp_path = f.name

        agent = config_to_agent(f"file://{temp_path}")
        assert agent.model.config["model_id"] == "test-model"
        assert agent.system_prompt == "Test prompt"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_config_to_agent_file_not_found():
    """Test that FileNotFoundError is raised for missing files."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        config_to_agent("/nonexistent/path/config.json")


def test_config_to_agent_invalid_json():
    """Test that JSONDecodeError is raised for invalid JSON."""
    try:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        with pytest.raises(json.JSONDecodeError):
            config_to_agent(temp_path)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_config_to_agent_invalid_config_type():
    """Test that ValueError is raised for invalid config types."""
    with pytest.raises(ValueError, match="Config must be a file path string or dictionary"):
        config_to_agent(123)


def test_config_to_agent_with_name():
    """Test config_to_agent handles agent name."""
    config = {"model": "test-model", "name": "TestAgent"}
    agent = config_to_agent(config)
    assert agent.name == "TestAgent"


def test_config_to_agent_ignores_none_values():
    """Test that None values in config are ignored."""
    config = {"model": "test-model", "prompt": None, "name": None}
    agent = config_to_agent(config)
    assert agent.model.config["model_id"] == "test-model"
    # Agent should use its defaults for None values


def test_config_to_agent_validation_error_invalid_field():
    """Test that invalid fields raise validation errors."""
    config = {"model": "test-model", "invalid_field": "value"}
    with pytest.raises(ValueError, match="Configuration validation error"):
        config_to_agent(config)


def test_config_to_agent_validation_error_wrong_type():
    """Test that wrong field types raise validation errors."""
    config = {"model": "test-model", "tools": "not-a-list"}
    with pytest.raises(ValueError, match="Configuration validation error"):
        config_to_agent(config)


def test_config_to_agent_validation_error_invalid_tool_item():
    """Test that invalid tool items raise validation errors."""
    config = {"model": "test-model", "tools": ["valid-tool", 123]}
    with pytest.raises(ValueError, match="Configuration validation error"):
        config_to_agent(config)


def test_config_to_agent_validation_error_invalid_tool():
    """Test that invalid tools raise helpful error messages."""
    config = {"model": "test-model", "tools": ["nonexistent_tool"]}
    with pytest.raises(ValueError, match="Failed to load tool nonexistent_tool"):
        config_to_agent(config)


def test_config_to_agent_validation_error_missing_module():
    """Test that missing modules raise helpful error messages."""
    config = {"model": "test-model", "tools": ["nonexistent.module.tool"]}
    with pytest.raises(ValueError, match="Failed to load tool nonexistent.module.tool"):
        config_to_agent(config)


def test_config_to_agent_validation_error_missing_function():
    """Test that missing functions in existing modules raise helpful error messages."""
    config = {"model": "test-model", "tools": ["json.nonexistent_function"]}
    with pytest.raises(ValueError, match="Failed to load tool json.nonexistent_function"):
        config_to_agent(config)


def test_config_to_agent_with_tool():
    """Test that missing functions in existing modules raise helpful error messages."""
    config = {"model": "test-model", "tools": ["tests.fixtures.say_tool:say"]}
    agent = config_to_agent(config)
    assert "say" in agent.tool_names


# =============================================================================
# Environment variable resolution tests
# =============================================================================


class TestResolveEnvVars:
    """Tests for the _resolve_env_vars utility function."""

    def test_resolve_dollar_prefix(self):
        """Test resolving $VAR_NAME format."""
        with patch.dict(os.environ, {"MY_API_KEY": "secret123"}):
            assert _resolve_env_vars("$MY_API_KEY") == "secret123"

    def test_resolve_braced_format(self):
        """Test resolving ${VAR_NAME} format."""
        with patch.dict(os.environ, {"MY_API_KEY": "secret456"}):
            assert _resolve_env_vars("${MY_API_KEY}") == "secret456"

    def test_resolve_nested_dict(self):
        """Test recursive resolution in nested dicts."""
        with patch.dict(os.environ, {"KEY1": "val1", "KEY2": "val2"}):
            data = {"outer": {"inner": "$KEY1"}, "flat": "${KEY2}"}
            result = _resolve_env_vars(data)
            assert result == {"outer": {"inner": "val1"}, "flat": "val2"}

    def test_resolve_list(self):
        """Test recursive resolution in lists."""
        with patch.dict(os.environ, {"KEY1": "val1", "KEY2": "val2"}):
            data = ["$KEY1", "${KEY2}", "literal"]
            result = _resolve_env_vars(data)
            assert result == ["val1", "val2", "literal"]

    def test_missing_env_var_raises(self):
        """Test that missing env vars raise ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the var is not set
            os.environ.pop("NONEXISTENT_VAR", None)
            with pytest.raises(ValueError, match="Environment variable 'NONEXISTENT_VAR' is not set"):
                _resolve_env_vars("$NONEXISTENT_VAR")

    def test_missing_braced_env_var_raises(self):
        """Test that missing braced env vars raise ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NONEXISTENT_VAR", None)
            with pytest.raises(ValueError, match="Environment variable 'NONEXISTENT_VAR' is not set"):
                _resolve_env_vars("${NONEXISTENT_VAR}")

    def test_non_env_string_unchanged(self):
        """Test that regular strings are returned unchanged."""
        assert _resolve_env_vars("just-a-string") == "just-a-string"

    def test_non_string_values_unchanged(self):
        """Test that non-string values pass through unchanged."""
        assert _resolve_env_vars(42) == 42
        assert _resolve_env_vars(True) is True
        assert _resolve_env_vars(3.14) == 3.14
        assert _resolve_env_vars(None) is None

    def test_deeply_nested_resolution(self):
        """Test env var resolution in deeply nested structures."""
        with patch.dict(os.environ, {"DEEP_VAL": "found"}):
            data = {"a": {"b": {"c": [{"d": "$DEEP_VAL"}]}}}
            result = _resolve_env_vars(data)
            assert result == {"a": {"b": {"c": [{"d": "found"}]}}}


# =============================================================================
# Schema validation tests — dual-format model field
# =============================================================================


class TestSchemaValidation:
    """Tests for the updated AGENT_CONFIG_SCHEMA that supports both string and object model formats."""

    def test_string_model_valid(self):
        """Test that string model format still passes validation."""
        config = {"model": "us.anthropic.claude-sonnet-4-20250514-v1:0"}
        agent = config_to_agent(config)
        assert agent.model.config["model_id"] == "us.anthropic.claude-sonnet-4-20250514-v1:0"

    def test_object_model_valid(self):
        """Test that object model format passes schema validation."""
        mock_model = MagicMock()
        with patch(
            "strands.experimental.agent_config._create_model_from_dict",
            return_value=mock_model,
        ):
            config = {
                "model": {
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-4-20250514",
                    "max_tokens": 10000,
                }
            }
            agent = config_to_agent(config)
            assert agent.model is mock_model

    def test_object_model_missing_provider_raises(self):
        """Test that object model without provider raises validation error."""
        config = {"model": {"model_id": "some-model"}}
        with pytest.raises(ValueError, match="Configuration validation error"):
            config_to_agent(config)

    def test_object_model_allows_additional_properties(self):
        """Test that object model format allows provider-specific properties."""
        mock_model = MagicMock()
        with patch(
            "strands.experimental.agent_config._create_model_from_dict",
            return_value=mock_model,
        ):
            config = {
                "model": {
                    "provider": "openai",
                    "model_id": "gpt-4o",
                    "client_args": {"api_key": "test"},
                    "custom_field": "allowed",
                }
            }
            # Should not raise
            config_to_agent(config)

    def test_null_model_still_valid(self):
        """Test that null model is still accepted for default behavior."""
        config = {"model": None}
        agent = config_to_agent(config)
        # Should use default model
        assert agent is not None

    def test_model_wrong_type_raises(self):
        """Test that model field with invalid type raises validation error."""
        config = {"model": 12345}
        with pytest.raises(ValueError, match="Configuration validation error"):
            config_to_agent(config)

    def test_object_model_from_file(self):
        """Test object model format loaded from a JSON file."""
        mock_model = MagicMock()
        config_data = {
            "model": {
                "provider": "anthropic",
                "model_id": "claude-sonnet-4-20250514",
            }
        }
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
                json.dump(config_data, f)
                f.flush()
                temp_path = f.name

            with patch(
                "strands.experimental.agent_config._create_model_from_dict",
                return_value=mock_model,
            ):
                agent = config_to_agent(temp_path)
                assert agent.model is mock_model
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# =============================================================================
# Provider factory tests — all 12 providers
# =============================================================================


class TestProviderMap:
    """Test that all 12 providers are registered in PROVIDER_MAP."""

    EXPECTED_PROVIDERS = [
        "bedrock",
        "anthropic",
        "openai",
        "gemini",
        "ollama",
        "litellm",
        "mistral",
        "llamaapi",
        "llamacpp",
        "sagemaker",
        "writer",
        "openai_responses",
    ]

    def test_all_providers_registered(self):
        """Test that all 12 providers are in PROVIDER_MAP."""
        for provider in self.EXPECTED_PROVIDERS:
            assert provider in PROVIDER_MAP, f"Provider '{provider}' not found in PROVIDER_MAP"

    def test_no_extra_providers(self):
        """Test that only the expected 12 providers are registered."""
        assert set(PROVIDER_MAP.keys()) == set(self.EXPECTED_PROVIDERS)


class TestCreateModelFromConfig:
    """Tests for _create_model_from_dict dispatching to cls.from_dict."""

    def test_unknown_provider_raises(self):
        """Test that an unknown provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model provider: 'nonexistent'"):
            _create_model_from_dict({"provider": "nonexistent", "model_id": "x"})

    def _patch_model_class(self, class_name):
        """Patch a model class on the strands.models module and return the mock."""
        mock_cls = MagicMock()
        mock_cls.from_dict.return_value = MagicMock()
        return patch(f"strands.models.{class_name}", mock_cls, create=True), mock_cls

    def test_dispatches_to_from_dict(self):
        """Test that _create_model_from_dict calls cls.from_dict on the resolved model class."""
        mock_model = MagicMock()
        mock_cls = MagicMock()
        mock_cls.from_dict.return_value = mock_model

        with patch("strands.models.AnthropicModel", mock_cls, create=True):
            result = _create_model_from_dict(
                {
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-4-20250514",
                    "max_tokens": 8192,
                    "client_args": {"api_key": "test-key"},
                }
            )
            mock_cls.from_dict.assert_called_once()
            call_config = mock_cls.from_dict.call_args[0][0]
            assert call_config["model_id"] == "claude-sonnet-4-20250514"
            assert call_config["max_tokens"] == 8192
            assert call_config["client_args"] == {"api_key": "test-key"}
            assert "provider" not in call_config
            assert result is mock_model

    def test_does_not_mutate_input(self):
        """Test that _create_model_from_dict does not mutate the input dict."""
        original = {"provider": "anthropic", "model_id": "test"}
        original_copy = original.copy()

        mock_cls = MagicMock()
        mock_cls.from_dict.return_value = MagicMock()
        with patch("strands.models.AnthropicModel", mock_cls, create=True):
            _create_model_from_dict(original)

        assert original == original_copy

    @pytest.mark.parametrize(
        "provider,class_name",
        list(PROVIDER_MAP.items()),
    )
    def test_all_providers_dispatch(self, provider, class_name):
        """Test that each registered provider dispatches to the correct class."""
        patcher, mock_cls = self._patch_model_class(class_name)
        with patcher:
            _create_model_from_dict({"provider": provider, "model_id": "test"})
            mock_cls.from_dict.assert_called_once()


# =============================================================================
# Model from_dict tests — provider-specific parameter handling
# =============================================================================


class TestModelFromConfig:
    """Tests for from_dict on model classes with non-standard constructors.

    Patches __init__ on each model class to capture the arguments passed by from_dict
    without actually initializing the model (which would require real provider dependencies).
    """

    def test_bedrock_from_dict_boto_client_config_conversion(self):
        """Test that BedrockModel.from_dict converts boto_client_config dict to BotocoreConfig."""
        from botocore.config import Config as BotocoreConfig

        from strands.models.bedrock import BedrockModel

        with patch.object(BedrockModel, "__init__", return_value=None) as mock_init:
            BedrockModel.from_dict(
                {
                    "model_id": "test-model",
                    "region_name": "us-west-2",
                    "boto_client_config": {"read_timeout": 300},
                }
            )
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["region_name"] == "us-west-2"
            assert isinstance(call_kwargs["boto_client_config"], BotocoreConfig)
            assert call_kwargs["model_id"] == "test-model"

    def test_bedrock_from_dict_without_boto_client_config(self):
        """Test BedrockModel.from_dict without boto_client_config."""
        from strands.models.bedrock import BedrockModel

        with patch.object(BedrockModel, "__init__", return_value=None) as mock_init:
            BedrockModel.from_dict(
                {
                    "model_id": "test-model",
                    "region_name": "us-east-1",
                }
            )
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["region_name"] == "us-east-1"
            assert "boto_client_config" not in call_kwargs

    def test_bedrock_from_dict_endpoint_url(self):
        """Test BedrockModel.from_dict with endpoint_url."""
        from strands.models.bedrock import BedrockModel

        with patch.object(BedrockModel, "__init__", return_value=None) as mock_init:
            BedrockModel.from_dict(
                {
                    "model_id": "test-model",
                    "endpoint_url": "https://vpce-1234.bedrock-runtime.us-west-2.vpce.amazonaws.com",
                }
            )
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["endpoint_url"] == "https://vpce-1234.bedrock-runtime.us-west-2.vpce.amazonaws.com"

    def test_ollama_from_dict_host_and_client_args_mapping(self):
        """Test that OllamaModel.from_dict routes host and maps client_args to ollama_client_args."""
        from strands.models.ollama import OllamaModel

        with patch.object(OllamaModel, "__init__", return_value=None) as mock_init:
            OllamaModel.from_dict(
                {
                    "model_id": "llama3",
                    "host": "http://localhost:11434",
                    "client_args": {"timeout": 30},
                }
            )
            call_args = mock_init.call_args
            assert call_args[0][0] == "http://localhost:11434"  # host is positional
            assert call_args[1]["ollama_client_args"] == {"timeout": 30}
            assert call_args[1]["model_id"] == "llama3"

    def test_ollama_from_dict_default_host(self):
        """Test OllamaModel.from_dict with no host specified defaults to None."""
        from strands.models.ollama import OllamaModel

        with patch.object(OllamaModel, "__init__", return_value=None) as mock_init:
            OllamaModel.from_dict({"model_id": "llama3"})
            call_args = mock_init.call_args
            assert call_args[0][0] is None  # host defaults to None

    def test_mistral_from_dict_api_key_extraction(self):
        """Test that MistralModel.from_dict extracts api_key separately."""
        from strands.models.mistral import MistralModel

        with patch.object(MistralModel, "__init__", return_value=None) as mock_init:
            MistralModel.from_dict(
                {
                    "model_id": "mistral-large-latest",
                    "api_key": "test-key",
                    "client_args": {"timeout": 60},
                }
            )
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["api_key"] == "test-key"
            assert call_kwargs["client_args"] == {"timeout": 60}
            assert call_kwargs["model_id"] == "mistral-large-latest"

    def test_llamacpp_from_dict_base_url_and_timeout(self):
        """Test that LlamaCppModel.from_dict extracts base_url and timeout."""
        from strands.models.llamacpp import LlamaCppModel

        with patch.object(LlamaCppModel, "__init__", return_value=None) as mock_init:
            LlamaCppModel.from_dict(
                {
                    "model_id": "default",
                    "base_url": "http://myhost:8080",
                    "timeout": 30.0,
                }
            )
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["base_url"] == "http://myhost:8080"
            assert call_kwargs["timeout"] == 30.0
            assert call_kwargs["model_id"] == "default"

    def test_sagemaker_from_dict_dict_params(self):
        """Test that SageMakerAIModel.from_dict receives endpoint_config and payload_config as dicts."""
        from strands.models.sagemaker import SageMakerAIModel

        with patch.object(SageMakerAIModel, "__init__", return_value=None) as mock_init:
            SageMakerAIModel.from_dict(
                {
                    "endpoint_config": {"endpoint_name": "my-ep", "region_name": "us-west-2"},
                    "payload_config": {"max_tokens": 1024, "stream": True},
                }
            )
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["endpoint_config"] == {"endpoint_name": "my-ep", "region_name": "us-west-2"}
            assert call_kwargs["payload_config"] == {"max_tokens": 1024, "stream": True}

    def test_sagemaker_from_dict_boto_client_config_conversion(self):
        """Test that SageMakerAIModel.from_dict converts boto_client_config dict to BotocoreConfig."""
        from botocore.config import Config as BotocoreConfig

        from strands.models.sagemaker import SageMakerAIModel

        with patch.object(SageMakerAIModel, "__init__", return_value=None) as mock_init:
            SageMakerAIModel.from_dict(
                {
                    "endpoint_config": {"endpoint_name": "my-ep"},
                    "payload_config": {"max_tokens": 1024},
                    "boto_client_config": {"read_timeout": 300},
                }
            )
            call_kwargs = mock_init.call_args[1]
            assert isinstance(call_kwargs["boto_client_config"], BotocoreConfig)

    def test_default_from_dict_client_args_pattern(self):
        """Test the default from_dict (inherited) handles client_args + remaining kwargs."""
        from strands.models.bedrock import BedrockModel

        with patch.object(BedrockModel, "__init__", return_value=None) as mock_init:
            # BedrockModel overrides from_dict, so use AnthropicModel which inherits the default
            from strands.models.anthropic import AnthropicModel

        with patch.object(AnthropicModel, "__init__", return_value=None) as mock_init:
            AnthropicModel.from_dict(
                {
                    "model_id": "claude-sonnet-4-20250514",
                    "max_tokens": 4096,
                    "client_args": {"api_key": "test"},
                    "params": {"temperature": 0.5},
                }
            )
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["client_args"] == {"api_key": "test"}
            assert call_kwargs["model_id"] == "claude-sonnet-4-20250514"
            assert call_kwargs["max_tokens"] == 4096
            assert call_kwargs["params"] == {"temperature": 0.5}

    def test_default_from_dict_without_client_args(self):
        """Test the default from_dict works without client_args."""
        from strands.models.anthropic import AnthropicModel

        with patch.object(AnthropicModel, "__init__", return_value=None) as mock_init:
            AnthropicModel.from_dict({"model_id": "test-model", "max_tokens": 1024})
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["model_id"] == "test-model"
            assert call_kwargs["max_tokens"] == 1024
            assert "client_args" not in call_kwargs


# =============================================================================
# Error handling tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in model creation."""

    def test_missing_optional_dependency(self):
        """Test clear error when provider dependency is not installed."""
        mock_cls = MagicMock()
        mock_cls.from_dict.side_effect = ImportError("No module named 'anthropic'")

        with patch("strands.models.AnthropicModel", mock_cls, create=True):
            with pytest.raises(ImportError, match="anthropic"):
                _create_model_from_dict(
                    {
                        "provider": "anthropic",
                        "model_id": "claude-sonnet-4-20250514",
                    }
                )

    def test_unknown_provider_error_message(self):
        """Test that unknown provider gives helpful error message."""
        with pytest.raises(ValueError, match="Unknown model provider: 'my_custom_provider'"):
            _create_model_from_dict({"provider": "my_custom_provider"})


# =============================================================================
# Integration: config_to_agent with object model
# =============================================================================


class TestConfigToAgentObjectModel:
    """Tests for config_to_agent using the object model format end-to-end."""

    def test_object_model_creates_agent(self):
        """Test that object model config creates an agent with the correct model."""
        mock_model = MagicMock()
        with patch(
            "strands.experimental.agent_config._create_model_from_dict",
            return_value=mock_model,
        ):
            config = {
                "model": {
                    "provider": "openai",
                    "model_id": "gpt-4o",
                },
                "prompt": "You are helpful",
            }
            agent = config_to_agent(config)
            assert agent.model is mock_model
            assert agent.system_prompt == "You are helpful"

    def test_object_model_env_var_resolution(self):
        """Test that env vars are resolved in object model config before provider creation."""
        mock_model = MagicMock()
        with patch.dict(os.environ, {"TEST_API_KEY": "resolved-key"}):
            with patch(
                "strands.experimental.agent_config._create_model_from_dict",
                return_value=mock_model,
            ) as mock_create:
                config = {
                    "model": {
                        "provider": "openai",
                        "model_id": "gpt-4o",
                        "client_args": {"api_key": "$TEST_API_KEY"},
                    }
                }
                config_to_agent(config)
                # Verify the env var was resolved before passing to the factory
                call_args = mock_create.call_args[0][0]
                assert call_args["client_args"]["api_key"] == "resolved-key"

    def test_string_model_backward_compat(self):
        """Test that string model still works as Bedrock model_id."""
        config = {"model": "us.anthropic.claude-sonnet-4-20250514-v1:0"}
        agent = config_to_agent(config)
        # String model is passed directly to Agent, which interprets it as Bedrock model_id
        assert agent.model.config["model_id"] == "us.anthropic.claude-sonnet-4-20250514-v1:0"

    def test_object_model_with_kwargs_override(self):
        """Test that kwargs can still override when using object model."""
        mock_model = MagicMock()
        with patch(
            "strands.experimental.agent_config._create_model_from_dict",
            return_value=mock_model,
        ):
            config = {
                "model": {"provider": "openai", "model_id": "gpt-4o"},
                "prompt": "Original prompt",
            }
            agent = config_to_agent(config, system_prompt="Override prompt")
            assert agent.system_prompt == "Override prompt"
