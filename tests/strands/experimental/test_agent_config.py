"""Tests for experimental config_to_agent function."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from strands.experimental import config_to_agent
from strands.experimental.agent_config import (
    PROVIDER_MAP,
    _create_model_from_dict,
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

    def _set_mock_on_models(self, class_name):
        """Inject a mock class directly into strands.models.__dict__ to avoid triggering lazy imports."""
        import strands.models as models_pkg

        mock_cls = MagicMock()
        mock_cls.from_dict.return_value = MagicMock()
        original = models_pkg.__dict__.get(class_name)
        models_pkg.__dict__[class_name] = mock_cls
        return mock_cls, original

    def _restore_models(self, class_name, original):
        """Restore original state of strands.models after test."""
        import strands.models as models_pkg

        if original is None:
            models_pkg.__dict__.pop(class_name, None)
        else:
            models_pkg.__dict__[class_name] = original

    def test_dispatches_to_from_dict(self):
        """Test that _create_model_from_dict calls cls.from_dict on the resolved model class."""
        mock_cls, original = self._set_mock_on_models("AnthropicModel")
        mock_model = MagicMock()
        mock_cls.from_dict.return_value = mock_model
        try:
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
        finally:
            self._restore_models("AnthropicModel", original)

    def test_does_not_mutate_input(self):
        """Test that _create_model_from_dict does not mutate the input dict."""
        mock_cls, original = self._set_mock_on_models("AnthropicModel")
        try:
            original_input = {"provider": "anthropic", "model_id": "test"}
            original_copy = original_input.copy()
            _create_model_from_dict(original_input)
            assert original_input == original_copy
        finally:
            self._restore_models("AnthropicModel", original)

    @pytest.mark.parametrize(
        "provider,class_name",
        list(PROVIDER_MAP.items()),
    )
    def test_all_providers_dispatch(self, provider, class_name):
        """Test that each registered provider dispatches to the correct class."""
        mock_cls, original = self._set_mock_on_models(class_name)
        try:
            _create_model_from_dict({"provider": provider, "model_id": "test"})
            mock_cls.from_dict.assert_called_once()
        finally:
            self._restore_models(class_name, original)


# =============================================================================
# Error handling tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in model creation."""

    def test_missing_optional_dependency(self):
        """Test clear error when provider dependency is not installed."""
        import strands.models as models_pkg

        mock_cls = MagicMock()
        mock_cls.from_dict.side_effect = ImportError("No module named 'anthropic'")

        original = models_pkg.__dict__.get("AnthropicModel")
        models_pkg.__dict__["AnthropicModel"] = mock_cls
        try:
            with pytest.raises(ImportError, match="anthropic"):
                _create_model_from_dict(
                    {
                        "provider": "anthropic",
                        "model_id": "claude-sonnet-4-20250514",
                    }
                )
        finally:
            if original is None:
                models_pkg.__dict__.pop("AnthropicModel", None)
            else:
                models_pkg.__dict__["AnthropicModel"] = original

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
