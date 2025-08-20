"""Integration tests for AgentConfigLoader demonstrating real-world usage."""

from unittest.mock import Mock, patch

from strands.agent.agent import Agent
from strands.experimental.config_loader.agent.agent_config_loader import AgentConfigLoader
from strands.types.tools import AgentTool, ToolSpec, ToolUse


class MockWeatherTool(AgentTool):
    """Mock weather tool for testing."""

    @property
    def tool_name(self) -> str:
        return "weather_tool.weather"

    @property
    def tool_spec(self) -> ToolSpec:
        return {
            "name": "weather_tool.weather",
            "description": "Get weather information",
            "inputSchema": {"type": "object", "properties": {"location": {"type": "string"}}},
        }

    @property
    def tool_type(self) -> str:
        return "weather"

    async def stream(self, tool_use: ToolUse, invocation_state: dict, **kwargs):
        yield {"result": f"Weather for {tool_use['input'].get('location', 'unknown')}: Sunny, 72°F"}


class TestAgentConfigLoaderIntegration:
    """Integration tests for AgentConfigLoader."""

    def test_load_agent_from_yaml_config(self):
        """Test loading agent from YAML-like configuration."""
        # This represents the YAML config from the feature description
        config = {
            "agent": {
                "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "system_prompt": (
                    "You're a helpful assistant. You can do simple math calculation, and tell the weather."
                ),
                "tools": [{"name": "weather_tool.weather"}],
            }
        }

        loader = AgentConfigLoader()
        mock_weather_tool = MockWeatherTool()

        with patch.object(loader, "_get_tool_config_loader") as mock_get_loader:
            mock_tool_loader = Mock()
            mock_tool_loader.load_tool.return_value = mock_weather_tool
            mock_get_loader.return_value = mock_tool_loader

            # Load the agent from the full config
            agent = loader.load_agent(config)

            # Verify the agent was created correctly
            assert isinstance(agent, Agent)
            assert (
                agent.system_prompt
                == "You're a helpful assistant. You can do simple math calculation, and tell the weather."
            )

            # Verify the tool was loaded
            mock_tool_loader.load_tool.assert_called_once_with("weather_tool.weather", None)

    def test_roundtrip_serialization(self):
        """Test that we can serialize and deserialize an agent."""
        # Create an agent
        original_config = {
            "agent": {
                "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "system_prompt": "You're a helpful assistant.",
                "agent_id": "test_agent",
                "name": "Test Agent",
                "description": "A test agent for roundtrip testing",
            }
        }

        loader = AgentConfigLoader()

        # Load agent from config
        agent = loader.load_agent(original_config)

        # Serialize the agent back to config
        serialized_config = loader.serialize_agent(agent)

        # Verify key fields are preserved
        assert serialized_config["agent"]["system_prompt"] == original_config["agent"]["system_prompt"]
        assert serialized_config["agent"]["agent_id"] == original_config["agent"]["agent_id"]
        assert serialized_config["agent"]["name"] == original_config["agent"]["name"]
        assert serialized_config["agent"]["description"] == original_config["agent"]["description"]

    def test_agent_with_config_parameter(self):
        """Test that Agent could theoretically accept a config parameter."""
        # This test demonstrates how the Agent constructor could be extended
        # to accept a config parameter as mentioned in the feature description

        config = {
            "agent": {
                "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "system_prompt": "You're a helpful assistant.",
                "tools": [],
            }
        }

        loader = AgentConfigLoader()

        # Load agent using the config loader
        agent = loader.load_agent(config)

        # Verify the agent was created with the correct configuration
        assert isinstance(agent, Agent)
        assert agent.system_prompt == config["agent"]["system_prompt"]

        # This demonstrates how Agent.__init__ could be extended:
        # def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        #     if config:
        #         loader = AgentConfigLoader()
        #         loaded_agent = loader.load_agent(config)
        #         # Copy properties from loaded_agent to self
        #     else:
        #         # Use existing initialization logic

    def test_circular_reference_protection(self):
        """Test that circular references between AgentConfigLoader and ToolConfigLoader are handled."""
        loader = AgentConfigLoader()

        # The lazy loading mechanism should prevent circular imports
        tool_config_loader1 = loader._get_tool_config_loader()
        tool_config_loader2 = loader._get_tool_config_loader()

        # Should return the same instance (cached)
        assert tool_config_loader1 is tool_config_loader2

        # The ToolConfigLoader should be able to work independently
        assert tool_config_loader1 is not None
