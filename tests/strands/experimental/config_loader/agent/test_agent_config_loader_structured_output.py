"""Tests for AgentConfigLoader structured output functionality."""

import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel

from strands.experimental.config_loader.agent import AgentConfigLoader


class BusinessModel(BaseModel):
    """Test Pydantic model for structured output tests."""

    company_name: str
    revenue: Optional[float] = None
    industry: Optional[str] = None


class TestAgentConfigLoaderStructuredOutput:
    """Test cases for AgentConfigLoader structured output functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = AgentConfigLoader()

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_load_agent_with_simple_structured_output_reference(self, mock_agent_class):
        """Test loading agent with simple structured output schema reference."""
        # Mock the Agent class
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent_class.return_value = mock_agent

        config = {
            "schemas": [
                {
                    "name": "UserProfile",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                        "required": ["name"],
                    },
                }
            ],
            "agent": {
                "name": "test_agent",
                "model": "test_model",
                "system_prompt": "Test prompt",
                "structured_output": "UserProfile",
            },
        }

        agent = self.loader.load_agent(config)

        # Verify agent was created
        assert agent is mock_agent
        mock_agent_class.assert_called_once()

        # Verify structured output was configured
        assert hasattr(mock_agent, "_structured_output_schema")
        assert mock_agent._structured_output_schema.__name__ == "UserProfile"
        assert hasattr(mock_agent, "extract_userprofile")

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_load_agent_with_python_class_reference(self, mock_agent_class):
        """Test loading agent with direct Python class reference."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent_class.return_value = mock_agent

        config = {
            "agent": {
                "name": "test_agent",
                "model": "test_model",
                "system_prompt": "Test prompt",
                "structured_output": (
                    "tests.strands.experimental.config_loader.agent."
                    "test_agent_config_loader_structured_output.BusinessModel"
                ),
            }
        }

        self.loader.load_agent(config)

        # Verify structured output was configured with the Python class
        assert hasattr(mock_agent, "_structured_output_schema")
        assert mock_agent._structured_output_schema is BusinessModel

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_load_agent_with_detailed_structured_output_config(self, mock_agent_class):
        """Test loading agent with detailed structured output configuration."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent_class.return_value = mock_agent

        config = {
            "schemas": [
                {
                    "name": "CustomerData",
                    "schema": {
                        "type": "object",
                        "properties": {"customer_id": {"type": "string"}, "name": {"type": "string"}},
                        "required": ["customer_id", "name"],
                    },
                }
            ],
            "structured_output_defaults": {
                "validation": {"strict": False, "allow_extra_fields": True},
                "error_handling": {"retry_on_validation_error": False, "max_retries": 1},
            },
            "agent": {
                "name": "test_agent",
                "model": "test_model",
                "system_prompt": "Test prompt",
                "structured_output": {
                    "schema": "CustomerData",
                    "validation": {"strict": True, "allow_extra_fields": False},
                    "error_handling": {"retry_on_validation_error": True, "max_retries": 3},
                },
            },
        }

        self.loader.load_agent(config)

        # Verify structured output was configured
        assert hasattr(mock_agent, "_structured_output_schema")
        assert mock_agent._structured_output_schema.__name__ == "CustomerData"
        assert hasattr(mock_agent, "_structured_output_validation")
        assert mock_agent._structured_output_validation["strict"] is True
        assert hasattr(mock_agent, "_structured_output_error_handling")
        assert mock_agent._structured_output_error_handling["max_retries"] == 3

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_load_agent_with_external_schema_file(self, mock_agent_class):
        """Test loading agent with external schema file."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent_class.return_value = mock_agent

        # Create temporary schema file
        schema_dict = {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "name": {"type": "string"},
                "price": {"type": "number", "minimum": 0},
            },
            "required": ["product_id", "name"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(schema_dict, f)
            temp_file = f.name

        try:
            config = {
                "schemas": [{"name": "Product", "schema_file": temp_file}],
                "agent": {
                    "name": "test_agent",
                    "model": "test_model",
                    "system_prompt": "Test prompt",
                    "structured_output": "Product",
                },
            }

            self.loader.load_agent(config)

            # Verify structured output was configured
            assert hasattr(mock_agent, "_structured_output_schema")
            assert mock_agent._structured_output_schema.__name__ == "Product"

        finally:
            Path(temp_file).unlink()

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_load_agent_with_structured_output_defaults(self, mock_agent_class):
        """Test loading agent with structured output defaults."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent_class.return_value = mock_agent

        config = {
            "schemas": [
                {
                    "name": "TestSchema",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                }
            ],
            "structured_output_defaults": {
                "validation": {"strict": False, "allow_extra_fields": True},
                "error_handling": {"retry_on_validation_error": False, "max_retries": 1},
            },
            "agent": {
                "name": "test_agent",
                "model": "test_model",
                "system_prompt": "Test prompt",
                "structured_output": {
                    "schema": "TestSchema",
                    "validation": {
                        "strict": True  # Should override default
                    },
                },
            },
        }

        self.loader.load_agent(config)

        # Verify defaults were merged with specific config
        validation_config = mock_agent._structured_output_validation
        error_config = mock_agent._structured_output_error_handling

        assert validation_config["strict"] is True  # Overridden
        assert validation_config["allow_extra_fields"] is True  # From defaults
        assert error_config["retry_on_validation_error"] is False  # From defaults
        assert error_config["max_retries"] == 1  # From defaults

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_load_multiple_agents_with_shared_schemas(self, mock_agent_class):
        """Test loading multiple agents that share schemas."""
        mock_agent1 = MagicMock()
        mock_agent1.name = "agent1"
        mock_agent2 = MagicMock()
        mock_agent2.name = "agent2"
        mock_agent_class.side_effect = [mock_agent1, mock_agent2]

        # Configuration with shared schemas
        base_schemas = [
            {
                "name": "SharedSchema",
                "schema": {"type": "object", "properties": {"data": {"type": "string"}}, "required": ["data"]},
            }
        ]

        agent1_config = {
            "schemas": base_schemas,
            "agent": {"name": "agent1", "model": "test_model", "structured_output": "SharedSchema"},
        }

        agent2_config = {
            "schemas": base_schemas,
            "agent": {"name": "agent2", "model": "test_model", "structured_output": "SharedSchema"},
        }

        # Load first agent (should load schemas)
        self.loader.load_agent(agent1_config)

        # Load second agent (should reuse schemas)
        self.loader.load_agent(agent2_config)

        # Both agents should have the same schema class
        assert hasattr(mock_agent1, "_structured_output_schema")
        assert hasattr(mock_agent2, "_structured_output_schema")
        assert mock_agent1._structured_output_schema is mock_agent2._structured_output_schema

    def test_schema_registry_operations(self):
        """Test schema registry operations."""
        # Test getting empty registry
        schemas = self.loader.list_schemas()
        assert len(schemas) == 0

        # Register a schema
        schema_dict = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        self.loader.schema_registry.register_schema("TestSchema", schema_dict)

        # Test listing schemas
        schemas = self.loader.list_schemas()
        assert "TestSchema" in schemas
        assert schemas["TestSchema"] == "programmatic"

        # Test getting schema registry
        registry = self.loader.get_schema_registry()
        assert registry is self.loader.schema_registry

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_error_handling_invalid_schema_reference(self, mock_agent_class):
        """Test error handling for invalid schema reference."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent_class.return_value = mock_agent

        config = {"agent": {"name": "test_agent", "model": "test_model", "structured_output": "NonExistentSchema"}}

        with pytest.raises(ValueError, match="Schema 'NonExistentSchema' not found"):
            self.loader.load_agent(config)

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_error_handling_invalid_python_class(self, mock_agent_class):
        """Test error handling for invalid Python class reference."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent_class.return_value = mock_agent

        config = {
            "agent": {"name": "test_agent", "model": "test_model", "structured_output": "non.existent.module.Class"}
        }

        with pytest.raises(ValueError, match="Cannot import Pydantic class"):
            self.loader.load_agent(config)

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_error_handling_missing_schema_in_detailed_config(self, mock_agent_class):
        """Test error handling for missing schema in detailed configuration."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent_class.return_value = mock_agent

        config = {
            "agent": {
                "name": "test_agent",
                "model": "test_model",
                "structured_output": {
                    "validation": {"strict": True}
                    # Missing "schema" field
                },
            }
        }

        with pytest.raises(ValueError, match="Structured output configuration must specify 'schema'"):
            self.loader.load_agent(config)

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_error_handling_invalid_structured_output_type(self, mock_agent_class):
        """Test error handling for invalid structured output configuration type."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent_class.return_value = mock_agent

        config = {
            "agent": {
                "name": "test_agent",
                "model": "test_model",
                "structured_output": 123,  # Invalid type
            }
        }

        with pytest.raises(ValueError, match="structured_output must be a string reference or configuration dict"):
            self.loader.load_agent(config)

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_structured_output_method_replacement(self, mock_agent_class):
        """Test that structured output methods are properly replaced."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        # Mock original methods
        original_structured_output = MagicMock()
        original_structured_output_async = MagicMock()
        mock_agent.structured_output = original_structured_output
        mock_agent.structured_output_async = original_structured_output_async

        mock_agent_class.return_value = mock_agent

        config = {
            "schemas": [
                {
                    "name": "TestSchema",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                }
            ],
            "agent": {
                "name": "test_agent",
                "model": "test_model",
                "structured_output": "TestSchema",
            },
        }

        agent = self.loader.load_agent(config)

        # Verify structured_output method was replaced
        assert agent.structured_output != original_structured_output

        # Verify original methods are stored
        assert hasattr(agent, "_original_structured_output")
        assert agent._original_structured_output == original_structured_output

        # Test that calling the new method calls the original with the schema
        agent.structured_output("test prompt")

        # The original method should have been called with the schema class
        original_structured_output.assert_called_once()
        call_args = original_structured_output.call_args
        assert len(call_args[0]) == 2  # schema_class and prompt
        assert call_args[0][1] == "test prompt"

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_convenience_method_creation(self, mock_agent_class):
        """Test that convenience methods are created for schemas."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent_class.return_value = mock_agent

        config = {
            "schemas": [
                {
                    "name": "CustomerProfile",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                }
            ],
            "agent": {
                "name": "test_agent",
                "model": "test_model",
                "structured_output": "CustomerProfile",
            },
        }

        self.loader.load_agent(config)

        # Verify convenience method was created
        assert hasattr(mock_agent, "extract_customerprofile")

    def test_global_schemas_loaded_once(self):
        """Test that global schemas are only loaded once."""
        config_with_schemas = {
            "schemas": [
                {
                    "name": "GlobalSchema",
                    "schema": {"type": "object", "properties": {"data": {"type": "string"}}, "required": ["data"]},
                }
            ],
            "agent": {
                "name": "test_agent",
                "model": "test_model",
            },
        }

        # Mock the _load_global_schemas method to track calls
        with patch.object(self.loader, "_load_global_schemas") as mock_load_schemas:
            # First call should load schemas
            with patch("strands.experimental.config_loader.agent.agent_config_loader.Agent"):
                self.loader.load_agent(config_with_schemas)
                mock_load_schemas.assert_called_once()

            # Second call should not load schemas again
            mock_load_schemas.reset_mock()
            with patch("strands.experimental.config_loader.agent.agent_config_loader.Agent"):
                self.loader.load_agent(config_with_schemas)
                mock_load_schemas.assert_not_called()

    @patch("strands.experimental.config_loader.agent.agent_config_loader.Agent")
    def test_agent_without_structured_output(self, mock_agent_class):
        """Test loading agent without structured output configuration."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent_class.return_value = mock_agent

        config = {
            "agent": {
                "name": "test_agent",
                "model": "test_model",
                "system_prompt": "Test prompt",
                # No structured_output configuration
            }
        }

        # Mock the _configure_agent_structured_output method to track if it's called
        with patch.object(self.loader, "_configure_agent_structured_output") as mock_configure:
            agent = self.loader.load_agent(config)

            # Verify agent was created normally
            assert agent is mock_agent

            # Verify structured output configuration was not called
            mock_configure.assert_not_called()
