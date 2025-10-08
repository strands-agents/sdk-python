"""Unit tests for ToolRegistry ToolProvider functionality."""

from unittest.mock import MagicMock, patch

import pytest

from strands.experimental.tools.tool_provider import ToolProvider
from strands.tools.registry import ToolRegistry
from strands.types.tools import AgentTool


class MockToolProvider(ToolProvider):
    """Mock ToolProvider for testing."""

    def __init__(self, tools=None, cleanup_error=None):
        self._tools = tools or []
        self._cleanup_error = cleanup_error
        self.cleanup_called = False
        self.remove_consumer_called = False
        self.remove_consumer_id = None
        self.add_consumer_called = False
        self.add_consumer_id = None

    async def load_tools(self):
        return self._tools

    async def cleanup(self):
        self.cleanup_called = True
        if self._cleanup_error:
            raise self._cleanup_error

    async def add_provider_consumer(self, consumer_id):
        self.add_consumer_called = True
        self.add_consumer_id = consumer_id

    async def remove_provider_consumer(self, consumer_id):
        self.remove_consumer_called = True
        self.remove_consumer_id = consumer_id


class TestToolRegistryToolProvider:
    """Test ToolRegistry integration with ToolProvider."""

    def test_process_tools_with_tool_provider(self):
        """Test that process_tools handles ToolProvider correctly."""
        # Create mock tools
        mock_tool1 = MagicMock(spec=AgentTool)
        mock_tool1.tool_name = "provider_tool_1"
        mock_tool2 = MagicMock(spec=AgentTool)
        mock_tool2.tool_name = "provider_tool_2"

        # Create mock provider
        provider = MockToolProvider([mock_tool1, mock_tool2])

        registry = ToolRegistry()

        with patch("strands.tools.registry.run_async") as mock_run_async:
            # Mock run_async to return the tools directly
            mock_run_async.return_value = [mock_tool1, mock_tool2]

            tool_names = registry.process_tools([provider])

            # Verify run_async was called with the provider's load_tools method
            mock_run_async.assert_called_once()

            # Verify tools were registered
            assert "provider_tool_1" in tool_names
            assert "provider_tool_2" in tool_names
            assert len(tool_names) == 2

            # Verify provider was tracked
            assert provider in registry.tool_providers

            # Verify tools are in registry
            assert registry.registry["provider_tool_1"] is mock_tool1
            assert registry.registry["provider_tool_2"] is mock_tool2

    def test_process_tools_with_multiple_providers(self):
        """Test that process_tools handles multiple ToolProviders."""
        # Create mock tools for first provider
        mock_tool1 = MagicMock(spec=AgentTool)
        mock_tool1.tool_name = "provider1_tool"
        provider1 = MockToolProvider([mock_tool1])

        # Create mock tools for second provider
        mock_tool2 = MagicMock(spec=AgentTool)
        mock_tool2.tool_name = "provider2_tool"
        provider2 = MockToolProvider([mock_tool2])

        registry = ToolRegistry()

        with patch("strands.tools.registry.run_async") as mock_run_async:
            # Mock run_async to return appropriate tools for each call
            mock_run_async.side_effect = [[mock_tool1], [mock_tool2]]

            tool_names = registry.process_tools([provider1, provider2])

            # Verify run_async was called twice
            assert mock_run_async.call_count == 2

            # Verify all tools were registered
            assert "provider1_tool" in tool_names
            assert "provider2_tool" in tool_names
            assert len(tool_names) == 2

            # Verify both providers were tracked
            assert provider1 in registry.tool_providers
            assert provider2 in registry.tool_providers
            assert len(registry.tool_providers) == 2

    def test_process_tools_with_mixed_tools_and_providers(self):
        """Test that process_tools handles mix of regular tools and providers."""
        # Create regular tool
        regular_tool = MagicMock(spec=AgentTool)
        regular_tool.tool_name = "regular_tool"

        # Create provider tool
        provider_tool = MagicMock(spec=AgentTool)
        provider_tool.tool_name = "provider_tool"
        provider = MockToolProvider([provider_tool])

        registry = ToolRegistry()

        with patch("strands.tools.registry.run_async") as mock_run_async:
            mock_run_async.return_value = [provider_tool]

            tool_names = registry.process_tools([regular_tool, provider])

            # Verify both tools were registered
            assert "regular_tool" in tool_names
            assert "provider_tool" in tool_names
            assert len(tool_names) == 2

            # Verify only provider was tracked
            assert provider in registry.tool_providers
            assert len(registry.tool_providers) == 1

    def test_process_tools_with_empty_provider(self):
        """Test that process_tools handles provider with no tools."""
        provider = MockToolProvider([])  # Empty tools list

        registry = ToolRegistry()

        with patch("strands.tools.registry.run_async") as mock_run_async:
            mock_run_async.return_value = []

            tool_names = registry.process_tools([provider])

            # Verify no tools were registered
            assert not tool_names

            # Verify provider was still tracked
            assert provider in registry.tool_providers

    def test_tool_providers_public_access(self):
        """Test that tool_providers can be accessed directly."""
        provider1 = MockToolProvider()
        provider2 = MockToolProvider()

        registry = ToolRegistry()
        registry.tool_providers = [provider1, provider2]

        # Verify direct access works
        assert len(registry.tool_providers) == 2
        assert provider1 in registry.tool_providers
        assert provider2 in registry.tool_providers

    def test_tool_providers_empty_by_default(self):
        """Test that tool_providers is empty by default."""
        registry = ToolRegistry()

        assert not registry.tool_providers
        assert isinstance(registry.tool_providers, list)

    def test_process_tools_provider_load_exception(self):
        """Test that process_tools handles exceptions from provider.load_tools()."""
        provider = MockToolProvider()

        registry = ToolRegistry()

        with patch("strands.tools.registry.run_async") as mock_run_async:
            # Make load_tools raise an exception
            mock_run_async.side_effect = Exception("Load tools failed")

            # Should raise the exception from load_tools
            with pytest.raises(Exception, match="Load tools failed"):
                registry.process_tools([provider])

            # Provider should still be tracked even if load_tools failed
            assert provider in registry.tool_providers

    def test_tool_provider_tracking_persistence(self):
        """Test that tool providers are tracked across multiple process_tools calls."""
        provider1 = MockToolProvider([MagicMock(spec=AgentTool, tool_name="tool1")])
        provider2 = MockToolProvider([MagicMock(spec=AgentTool, tool_name="tool2")])

        registry = ToolRegistry()

        with patch("strands.tools.registry.run_async") as mock_run_async:
            mock_run_async.side_effect = [
                [MagicMock(spec=AgentTool, tool_name="tool1")],
                [MagicMock(spec=AgentTool, tool_name="tool2")],
            ]

            # Process first provider
            registry.process_tools([provider1])
            assert len(registry.tool_providers) == 1
            assert provider1 in registry.tool_providers

            # Process second provider
            registry.process_tools([provider2])
            assert len(registry.tool_providers) == 2
            assert provider1 in registry.tool_providers
            assert provider2 in registry.tool_providers

    def test_process_tools_provider_async_optimization(self):
        """Test that load_tools and add_provider_consumer are called in same async context."""
        mock_tool = MagicMock(spec=AgentTool)
        mock_tool.tool_name = "test_tool"

        class TestProvider(ToolProvider):
            def __init__(self):
                self.load_tools_called = False
                self.add_consumer_called = False
                self.add_consumer_id = None

            async def load_tools(self):
                self.load_tools_called = True
                return [mock_tool]

            async def add_provider_consumer(self, consumer_id):
                self.add_consumer_called = True
                self.add_consumer_id = consumer_id

            async def remove_provider_consumer(self, consumer_id):
                pass

        provider = TestProvider()
        registry = ToolRegistry()

        # Process the provider - this should call both methods in same async context
        tool_names = registry.process_tools([provider])

        # Verify both methods were called
        assert provider.load_tools_called
        assert provider.add_consumer_called
        assert provider.add_consumer_id == registry._registry_id

        # Verify tool was registered
        assert "test_tool" in tool_names
        assert provider in registry.tool_providers

    @pytest.mark.asyncio
    async def test_registry_cleanup(self):
        """Test that registry cleanup calls remove_provider_consumer on all providers."""
        provider1 = MockToolProvider()
        provider2 = MockToolProvider()

        registry = ToolRegistry()
        registry.tool_providers = [provider1, provider2]

        await registry.cleanup_async()

        # Verify both providers had remove_provider_consumer called
        assert provider1.remove_consumer_called
        assert provider2.remove_consumer_called

    @pytest.mark.asyncio
    async def test_registry_cleanup_with_provider_consumer_removal(self):
        """Test that cleanup removes provider consumers correctly."""

        class TestProvider(ToolProvider):
            def __init__(self):
                self.remove_consumer_called = False
                self.remove_consumer_id = None

            async def load_tools(self):
                return []

            async def add_provider_consumer(self, consumer_id):
                pass

            async def remove_provider_consumer(self, consumer_id):
                self.remove_consumer_called = True
                self.remove_consumer_id = consumer_id

        provider = TestProvider()
        registry = ToolRegistry()
        registry.tool_providers = [provider]

        # Call cleanup
        await registry.cleanup_async()

        # Verify remove_provider_consumer was called with correct ID
        assert provider.remove_consumer_called
        assert provider.remove_consumer_id == registry._registry_id
