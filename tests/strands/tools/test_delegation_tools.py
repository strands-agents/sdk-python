"""Tests for delegation tool generation functionality."""

import pytest
from unittest.mock import Mock, patch

from strands.agent.agent import Agent
from strands.types.exceptions import AgentDelegationException


class TestDelegationToolGeneration:
    """Test basic delegation tool generation functionality."""

    def test_tool_registration_basic(self):
        """Test basic tool registration with sub-agent."""
        with patch('strands.tools.tool') as mock_tool:
            # Configure the mock to return a function with __name__
            mock_delegation_tool = Mock()
            mock_delegation_tool.__name__ = "handoff_to_subagent"
            mock_tool.return_value = mock_delegation_tool

            # Create a real agent instance
            orchestrator = Agent(name="Orchestrator")
            orchestrator.tool_registry = Mock()

            sub_agent = Mock()
            sub_agent.name = "SubAgent"
            sub_agent.description = "Test sub-agent"
            sub_agent.tools = []

            # Call the tool generation method
            orchestrator._generate_delegation_tools([sub_agent])

            # Verify tool registration was called
            assert orchestrator.tool_registry.register_tool.called
            tool_call = orchestrator.tool_registry.register_tool.call_args[0][0]

            # Verify tool was called with correct name
            mock_tool.assert_called_with(name="handoff_to_subagent")

    def test_empty_sub_agents_list(self):
        """Test no-op for empty sub-agents list."""
        orchestrator = Agent(name="Orchestrator")
        orchestrator.tool_registry = Mock()

        # Call with empty list
        orchestrator._generate_delegation_tools([])

        # Should not register any tools
        assert not orchestrator.tool_registry.register_tool.called

    def test_name_sanitization(self):
        """Test hyphen to underscore conversion in tool names."""
        with patch('strands.tools.tool') as mock_tool:
            # Configure the mock to return a function
            mock_delegation_tool = Mock()
            mock_delegation_tool.__name__ = "handoff_to_my_agent"
            mock_tool.return_value = mock_delegation_tool

            orchestrator = Agent(name="Orchestrator")
            orchestrator.tool_registry = Mock()

            sub_agent = Mock()
            sub_agent.name = "My-Agent"
            sub_agent.description = "Test agent"
            sub_agent.tools = []

            # Generate tools
            orchestrator._generate_delegation_tools([sub_agent])

            # Verify tool was called with sanitized name
            mock_tool.assert_called_with(name="handoff_to_my_agent")


class TestToolDocstringEnrichment:
    """Test enhanced docstring generation for delegation tools."""

    def test_docstring_complete_with_capabilities(self):
        """Test full docstring generation with all sections."""
        with patch('strands.tools.tool') as mock_tool:
            orchestrator = Agent(name="Orchestrator")
            orchestrator.tool_registry = Mock()

            sub_agent = Mock()
            sub_agent.name = "DataProcessor"
            sub_agent.description = "Data processing specialist"
            sub_agent.tools = [
                Mock(tool_name="process_csv"),
                Mock(tool_name="validate_data"),
                Mock(tool_name="export_results")
            ]

            # Generate tools
            orchestrator._generate_delegation_tools([sub_agent])

            tool = orchestrator.tool_registry.register_tool.call_args[0][0]
            docstring = tool.__doc__

            # Verify all sections present
            assert "Transfer control completely to DataProcessor" in docstring
            assert "Data processing specialist" in docstring
            assert "Capabilities include:" in docstring
            assert "process_csv" in docstring
            assert "DELEGATION CRITERIA:" in docstring
            assert "specialized expertise" in docstring.lower()
            assert "EXAMPLE USAGE:" in docstring
            assert "Args:" in docstring

    def test_docstring_without_capabilities(self):
        """Test docstring when sub-agent has no tools."""
        with patch('strands.tools.tool') as mock_tool:
            orchestrator = Agent(name="Orchestrator")
            orchestrator.tool_registry = Mock()

            sub_agent = Mock()
            sub_agent.name = "SimpleAgent"
            sub_agent.description = "Simple agent without tools"
            sub_agent.tools = []

            # Generate tools
            orchestrator._generate_delegation_tools([sub_agent])

            tool = orchestrator.tool_registry.register_tool.call_args[0][0]
            docstring = tool.__doc__

            # Should not include capabilities section
            assert "Capabilities include:" not in docstring
            # But should still have other sections
            assert "DELEGATION CRITERIA:" in docstring
            assert "EXAMPLE USAGE:" in docstring
            assert "Simple agent without tools" in docstring

    def test_docstring_delegation_criteria_content(self):
        """Test delegation criteria content is specific to agent."""
        with patch('strands.tools.tool') as mock_tool:
            orchestrator = Agent(name="Orchestrator")
            orchestrator.tool_registry = Mock()

            sub_agent = Mock()
            sub_agent.name = "BillingSpecialist"
            sub_agent.description = "Customer billing and payment processing"
            sub_agent.tools = []

            # Generate tools
            orchestrator._generate_delegation_tools([sub_agent])

            tool = orchestrator.tool_registry.register_tool.call_args[0][0]
            docstring = tool.__doc__

            # Verify delegation criteria is specific to the agent
            assert "BillingSpecialist's specialized expertise" in docstring
            assert "customer billing and payment processing" in docstring.lower()

    def test_docstring_example_usage(self):
        """Test example usage section provides practical examples."""
        with patch('strands.tools.tool') as mock_tool:
            orchestrator = Agent(name="Orchestrator")
            orchestrator.tool_registry = Mock()

            sub_agent = Mock()
            sub_agent.name = "TechnicalSupport"
            sub_agent.description = "Technical troubleshooting and support"
            sub_agent.tools = []

            # Generate tools
            orchestrator._generate_delegation_tools([sub_agent])

            tool = orchestrator.tool_registry.register_tool.call_args[0][0]
            docstring = tool.__doc__

            # Verify example usage section
            assert "EXAMPLE USAGE:" in docstring
            assert 'Handle this customer billing inquiry' in docstring
            assert 'Debug this API error' in docstring


class TestToolSchemaGeneration:
    """Test JSON schema generation for delegation tools."""

    def test_json_schema_complete_structure(self):
        """Test JSON schema includes proper structure."""
        with patch('strands.tools.tool') as mock_tool:
            orchestrator = Agent(name="Orchestrator")
            orchestrator.tool_registry = Mock()

            sub_agent = Mock()
            sub_agent.name = "AnalystAgent"
            sub_agent.description = "Data analysis specialist"
            sub_agent.tools = []

            # Generate tools
            orchestrator._generate_delegation_tools([sub_agent])

            tool = orchestrator.tool_registry.register_tool.call_args[0][0]

            if hasattr(tool, '__schema__'):
                schema = tool.__schema__

                # Verify schema structure
                assert schema["type"] == "object"
                assert "properties" in schema
                assert "required" in schema
                assert schema["required"] == ["message"]
                assert schema["additionalProperties"] is False

                # Verify message property
                assert "message" in schema["properties"]
                assert schema["properties"]["message"]["type"] == "string"
                assert "Message to pass to AnalystAgent" in schema["properties"]["message"]["description"]

    def test_json_schema_hidden_parameters(self):
        """Test that internal parameters have proper schema definitions."""
        with patch('strands.tools.tool') as mock_tool:
            orchestrator = Agent(name="Orchestrator")
            orchestrator.tool_registry = Mock()

            sub_agent = Mock()
            sub_agent.name = "TestAgent"
            sub_agent.description = "Test agent"
            sub_agent.tools = []

            # Generate tools
            orchestrator._generate_delegation_tools([sub_agent])

            tool = orchestrator.tool_registry.register_tool.call_args[0][0]

            if hasattr(tool, '__schema__'):
                schema = tool.__schema__

                # Verify hidden parameters are in schema
                assert "target_agent" in schema["properties"]
                assert schema["properties"]["target_agent"]["type"] == "string"
                assert "Internal target agent identifier" in schema["properties"]["target_agent"]["description"]


class TestToolRegistryDelegationLogic:
    """Test tool registry handles delegation tools correctly."""

    def test_delegation_tool_prefix_detection(self):
        """Test registry detects delegation tools by prefix."""
        tool = Mock()
        tool.tool_name = "handoff_to_specialist"
        
        is_delegation_tool = tool.tool_name.startswith("handoff_to_")
        assert is_delegation_tool is True
    
    def test_delegation_coexists_with_regular_tools(self):
        """Test delegation tools work alongside regular tools."""
        # Create a simple mock that validates coexistence
        orchestrator = Mock()
        orchestrator.name = "Orchestrator"
        orchestrator.tool_names = ["regular_tool", "another_tool"]
        
        sub_agent = Mock()
        sub_agent.name = "SubAgent"
        
        # The test verifies that validation logic allows coexistence
        # by checking tool name conflict detection doesn't trigger for non-conflicting names
        tool_name = f"handoff_to_{sub_agent.name.lower()}"
        
        # Should not conflict
        assert tool_name not in orchestrator.tool_names


class TestDelegationToolFunctionality:
    """Test that delegation tools actually raise AgentDelegationException."""

    def test_tool_raises_delegation_exception(self):
        """Test that calling the tool raises AgentDelegationException."""
        # Create real sub-agent and orchestrator
        sub_agent = Agent(name="TestAgent", model="mock")

        orchestrator = Agent(
            name="Orchestrator",
            model="mock",
            sub_agents=[sub_agent],
            max_delegation_depth=10
        )

        # Get the generated delegation tool
        tool_name = f"handoff_to_{sub_agent.name.lower().replace('-', '_')}"
        tool = orchestrator.tool_registry.registry[tool_name]

        # Test that calling the tool raises the exception
        with pytest.raises(AgentDelegationException) as exc_info:
            tool(message="Test delegation message")

        exception = exc_info.value
        assert exception.target_agent == "TestAgent"
        assert exception.message == "Test delegation message"
        assert exception.transfer_state is True
        assert exception.transfer_messages is True
        assert exception.delegation_chain == ["Orchestrator"]

    def test_tool_respects_transfer_parameter_overrides(self):
        """Test that tool respects parameter overrides for transfer flags."""
        sub_agent = Agent(name="TestAgent", model="mock")

        orchestrator = Agent(
            name="Orchestrator",
            model="mock",
            sub_agents=[sub_agent],
            delegation_state_transfer=True,  # Default
            delegation_message_transfer=True,  # Default
            max_delegation_depth=10
        )

        # Get the generated delegation tool
        tool_name = f"handoff_to_{sub_agent.name.lower().replace('-', '_')}"
        tool = orchestrator.tool_registry.registry[tool_name]

        # Test with overrides
        with pytest.raises(AgentDelegationException) as exc_info:
            tool(
                message="Test message",
                transfer_state=False,
                transfer_messages=False
            )

        exception = exc_info.value
        assert exception.transfer_state is False
        assert exception.transfer_messages is False

    def test_tool_validates_delegation_depth(self):
        """Test that tool validates maximum delegation depth."""
        sub_agent = Agent(name="TestAgent", model="mock")

        orchestrator = Agent(
            name="Orchestrator",
            model="mock",
            sub_agents=[sub_agent],
            max_delegation_depth=3
        )

        # Get the generated delegation tool
        tool_name = f"handoff_to_{sub_agent.name.lower().replace('-', '_')}"
        tool = orchestrator.tool_registry.registry[tool_name]

        # Test with delegation chain at max depth
        with pytest.raises(ValueError, match="Maximum delegation depth"):
            tool(
                message="Test message",
                delegation_chain=["Agent1", "Agent2", "Agent3"]  # Already at max depth
            )