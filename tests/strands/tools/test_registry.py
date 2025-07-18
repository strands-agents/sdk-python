"""Unit tests for the tool registry module."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strands.tools.registry import ToolRegistry
from strands.types.tools import AgentTool, ToolSpec


class MockAgentTool(AgentTool):
    """Mock implementation of AgentTool for testing."""

    def __init__(self, tool_name, tool_spec):
        """Initialize a mock agent tool."""
        super().__init__()
        self._tool_name = tool_name
        self._tool_spec = tool_spec
        self._tool_type = "mock"
        self._is_dynamic = False
        self._supports_hot_reload = False

    @property
    def tool_name(self) -> str:
        """Get the tool name."""
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Get the tool specification."""
        return self._tool_spec

    @property
    def tool_type(self) -> str:
        """Get the tool type."""
        return self._tool_type

    @property
    def supports_hot_reload(self) -> bool:
        """Whether the tool supports hot reloading."""
        return self._supports_hot_reload

    async def stream(self, tool_use, invocation_state, **kwargs):
        """Stream tool events and return the final result."""
        result = {"toolUseId": tool_use["toolUseId"], "status": "success", "content": [{"text": "success"}]}
        yield result

    def mark_dynamic(self):
        """Mark the tool as dynamic."""
        self._is_dynamic = True

    def copy(self):
        """Create a copy of this tool."""
        return MockAgentTool(self._tool_name, self._tool_spec.copy())


class TestToolRegistry(unittest.TestCase):
    """Test cases for the ToolRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
        self.sample_tool_spec = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {"param1": {"type": "string", "description": "Parameter 1"}},
                "required": ["param1"],
            },
        }
        self.mock_tool = MockAgentTool("test_tool", self.sample_tool_spec)

    def test_create_tool_with_agent_tool(self):
        """Test creating tools with an AgentTool instance."""
        # Create a tool
        created_tool = self.registry.create_tool(self.mock_tool)

        # Verify the tool was created
        self.assertEqual(created_tool, "test_tool")
        self.assertIn("test_tool", self.registry.agent_tools)

        # Verify tool properties
        registered_tool = self.registry.agent_tools["test_tool"]
        self.assertEqual(registered_tool.tool_name, "test_tool")
        self.assertEqual(registered_tool.tool_spec, self.sample_tool_spec)

    def test_create_tool_with_duplicate_name(self):
        """Test creating tools with duplicate names raises ValueError."""
        # Create a tool first
        self.registry.create_tool(self.mock_tool)

        # Try to create another tool with the same name
        with self.assertRaises(ValueError):
            self.registry.create_tool(self.mock_tool)

    def test_create_tool_with_similar_names(self):
        """Test creating tools with names that differ only by - and _."""
        # Create a tool with underscore
        underscore_tool = MockAgentTool("test_tool", self.sample_tool_spec)
        self.registry.create_tool(underscore_tool)

        # Try to create another tool with hyphen instead of underscore
        hyphen_tool = MockAgentTool("test-tool", self.sample_tool_spec)
        with self.assertRaises(ValueError):
            self.registry.create_tool(hyphen_tool)

    def test_list_tools(self):
        """Test reading all tool specifications."""
        # Create some tools
        tool1 = MockAgentTool("tool1", {"name": "tool1", "description": "Tool 1", "inputSchema": {"type": "object"}})
        tool2 = MockAgentTool("tool2", {"name": "tool2", "description": "Tool 2", "inputSchema": {"type": "object"}})
        tools = [tool1, tool2]
        for tool in tools:
            self.registry.create_tool(tool)

        # Read all tools
        tool_specs = self.registry.list_tools()

        # Verify the tool specs
        self.assertEqual(len(tool_specs), 2)
        self.assertIn("tool1", tool_specs)
        self.assertIn("tool2", tool_specs)
        self.assertEqual(tool_specs["tool1"]["name"], "tool1")
        self.assertEqual(tool_specs["tool2"]["name"], "tool2")

    def test_update_tool(self):
        """Test updating existing tools."""
        # Create a tool
        self.registry.create_tool(self.mock_tool)

        # Create an updated version of the tool
        updated_spec = self.sample_tool_spec.copy()
        updated_spec["description"] = "Updated description"
        updated_tool = MockAgentTool("test_tool", updated_spec)

        # Update the tool
        updated_tool = self.registry.update_tool(updated_tool)

        # Verify the tool was updated
        self.assertEqual(updated_tool, "test_tool")
        self.assertEqual(self.registry.agent_tools["test_tool"].tool_spec["description"], "Updated description")

    def test_update_nonexistent_tool(self):
        """Test updating a tool that doesn't exist."""
        # Create a tool that doesn't exist in the registry
        nonexistent_tool = MockAgentTool("nonexistent", self.sample_tool_spec)

        # Try to update the nonexistent tool
        with self.assertRaises(ValueError):
            self.registry.update_tool(nonexistent_tool)

    def test_delete_tool(self):
        """Test deleting tools from the registry."""
        # Create some tools
        tool1 = MockAgentTool("tool1", {"name": "tool1", "description": "Tool 1", "inputSchema": {"type": "object"}})
        tool2 = MockAgentTool("tool2", {"name": "tool2", "description": "Tool 2", "inputSchema": {"type": "object"}})
        tools = [tool1, tool2]
        for tool in tools:
            self.registry.create_tool(tool)

        # Delete one tool
        deleted_tool = self.registry.delete_tool("tool1")

        # Verify the tool was deleted
        self.assertEqual(deleted_tool, "tool1")
        self.assertNotIn("tool1", self.registry.agent_tools)
        self.assertIn("tool2", self.registry.agent_tools)

    def test_delete_nonexistent_tool(self):
        """Test deleting a tool that doesn't exist raises ValueError."""
        with self.assertRaises(ValueError):
            self.registry.delete_tool("nonexistent")

    def test_delete_dynamic_tool(self):
        """Test deleting a dynamic tool."""
        # Create a dynamic tool
        dynamic_tool = MockAgentTool(
            "dynamic_tool", {"name": "dynamic_tool", "description": "Dynamic Tool", "inputSchema": {"type": "object"}}
        )
        dynamic_tool.mark_dynamic()
        self.registry.create_tool(dynamic_tool)

        # Add to hot reload tools
        self.registry._hot_reload_tools[dynamic_tool.tool_name] = dynamic_tool

        # Delete the tool
        deleted_tool = self.registry.delete_tool("dynamic_tool")

        # Verify the tool was deleted from both registries
        self.assertEqual(deleted_tool, "dynamic_tool")
        self.assertNotIn("dynamic_tool", self.registry.agent_tools)
        self.assertNotIn("dynamic_tool", self.registry._hot_reload_tools)

    def test_create_tool_with_string_path(self):
        """Test creating tools with a string file path."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(b'''
                TOOL_SPEC = {
                    "name": "temp_tool",
                    "description": "A temporary tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "param1": {"type": "string", "description": "Parameter 1"}
                        },
                        "required": ["param1"]
                    }
                }

                def temp_tool(param1):
                    """A temporary tool function."""
                    return {"result": param1}
                ''')
            temp_path = temp_file.name

        try:
            # Mock the _load_tool_from_filepath method
            with mock.patch.object(self.registry, "_load_tool_from_filepath") as mock_load:
                # Create a tool with the file path
                self.registry.create_tool(temp_path)

                # Verify _load_tool_from_filepath was called with the correct arguments
                tool_name = os.path.basename(temp_path).split(".")[0]
                mock_load.assert_called_once_with(tool_name=tool_name, tool_path=temp_path)
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)

    def test_create_tool_with_dict_path(self):
        """Test creating tools with a dictionary containing a path."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(b'''
                TOOL_SPEC = {
                    "name": "dict_tool",
                    "description": "A tool from dict path",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "param1": {"type": "string", "description": "Parameter 1"}
                        },
                        "required": ["param1"]
                    }
                }

                def dict_tool(param1):
                    """A tool function from dict path."""
                    return {"result": param1}
                ''')
            temp_path = temp_file.name

        try:
            # Mock the _load_tool_from_filepath method
            with mock.patch.object(self.registry, "_load_tool_from_filepath") as mock_load:
                # Create a tool with a dictionary containing a path
                self.registry.create_tool({"path": temp_path})

                # Verify _load_tool_from_filepath was called with the correct arguments
                tool_name = os.path.basename(temp_path).split(".")[0]
                mock_load.assert_called_once_with(tool_name=tool_name, tool_path=temp_path)
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)

    def test_create_tool_with_dict_name_path(self):
        """Test creating tools with a dictionary containing a name and path."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(b'''
                TOOL_SPEC = {
                    "name": "custom_name_tool",
                    "description": "A tool with custom name",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "param1": {"type": "string", "description": "Parameter 1"}
                        },
                        "required": ["param1"]
                    }
                }

                def custom_name_tool(param1):
                    """A tool function with custom name."""
                    return {"result": param1}
                ''')
            temp_path = temp_file.name

        try:
            # Mock the _load_tool_from_filepath method
            with mock.patch.object(self.registry, "_load_tool_from_filepath") as mock_load:
                # Create a tool with a dictionary containing a name and path
                self.registry.create_tool({"name": "custom_name", "path": temp_path})

                # Verify _load_tool_from_filepath was called with the correct arguments
                mock_load.assert_called_once_with(tool_name="custom_name", tool_path=temp_path)
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)

    def test_create_tool_with_module(self):
        """Test creating tools with an imported Python module."""
        # Create a mock module that satisfies the conditions for _load_tool_from_filepath
        mock_module = mock.MagicMock()
        mock_module.__file__ = "/path/to/module.py"
        mock_module.__name__ = "test_module"
        mock_module.TOOL_SPEC = self.sample_tool_spec
        mock_module.test_module = lambda: None  # Function with same name as module

        # Mock inspect.ismodule to return True
        with mock.patch("inspect.ismodule", return_value=True):
            # Mock the _load_tool_from_filepath method
            with mock.patch.object(self.registry, "_load_tool_from_filepath") as mock_load:
                # Create a tool with the module
                self.registry.create_tool(mock_module)

                # Verify _load_tool_from_filepath was called with the correct arguments
                mock_load.assert_called_once_with(tool_name="test_module", tool_path="/path/to/module.py")

    def test_extract_tool_name(self):
        """Test extracting tool names from various formats."""
        # Test with string path
        self.assertEqual(self.registry._extract_tool_name("/path/to/tool.py"), "tool")

        # Test with dict containing name
        self.assertEqual(self.registry._extract_tool_name({"name": "tool_name"}), "tool_name")

        # Test with dict containing path
        self.assertEqual(self.registry._extract_tool_name({"path": "/path/to/tool.py"}), "tool")

        # Test with module
        mock_module = mock.MagicMock()
        mock_module.__file__ = "/path/to/module.py"
        mock_module.__name__ = "module.tool"
        with mock.patch("inspect.ismodule", return_value=True):
            self.assertEqual(self.registry._extract_tool_name(mock_module), "tool")

        # Test with AgentTool
        self.assertEqual(self.registry._extract_tool_name(self.mock_tool), "test_tool")

        # Test with unsupported type
        self.assertIsNone(self.registry._extract_tool_name(123))

    @mock.patch("os.path.exists")
    def test_load_tool_from_filepath_file_not_found(self, mock_exists):
        """Test loading a tool from a nonexistent file path."""
        mock_exists.return_value = False

        with self.assertRaises(ValueError):
            self.registry._load_tool_from_filepath("nonexistent", "/path/to/nonexistent.py")

    @mock.patch("strands.tools.loader.ToolLoader")
    @mock.patch("os.path.exists")
    def test_load_tool_from_filepath_success(self, mock_exists, mock_loader):
        """Test successfully loading a tool from a file path."""
        mock_exists.return_value = True
        mock_tool = MockAgentTool("loaded_tool", {"name": "loaded_tool", "description": "Loaded Tool"})
        mock_loader.load_tool.return_value = mock_tool

        # Load the tool
        self.registry._load_tool_from_filepath("loaded_tool", "/path/to/loaded_tool.py")

        # Verify the tool was registered
        mock_loader.load_tool.assert_called_once_with("/path/to/loaded_tool.py", "loaded_tool")
        self.assertIn("loaded_tool", self.registry.agent_tools)

    def test_get_tools_dirs(self):
        """Test getting tool directory paths."""
        with mock.patch("pathlib.Path.exists") as mock_exists, mock.patch("pathlib.Path.is_dir") as mock_is_dir:
            # Mock the existence of the tools directory
            mock_exists.return_value = True
            mock_is_dir.return_value = True

            # Create a new registry to test the initialization with mocked paths
            test_registry = ToolRegistry()

            # Verify the tools directory was found
            self.assertEqual(len(test_registry.tool_directories), 1)
            self.assertEqual(test_registry.tool_directories[0], Path.cwd() / "tools")

    def test_discover_tool_modules(self):
        """Test discovering tool modules in the tools directory."""
        # Mock the tools directory
        tools_dir = mock.MagicMock()

        # Set the tool_directories attribute directly on the registry
        self.registry.tool_directories = [tools_dir]

        # Create mock path objects
        tool1_path = mock.MagicMock()
        tool1_path.is_file.return_value = True
        tool1_path.name = "tool1.py"
        tool1_path.stem = "tool1"

        tool2_path = mock.MagicMock()
        tool2_path.is_file.return_value = True
        tool2_path.name = "tool2.py"
        tool2_path.stem = "tool2"

        # Mock the glob method on the tools directory
        tools_dir.glob.return_value = [tool1_path, tool2_path]

        # Discover the tool modules
        tool_modules = self.registry._discover_tool_modules()

        # Verify the tool modules were discovered
        self.assertEqual(len(tool_modules), 2)
        self.assertEqual(tool_modules["tool1"], tool1_path)
        self.assertEqual(tool_modules["tool2"], tool2_path)

    def test_update_tools_from_directory(self):
        """Test updating tools from a directory."""
        # Create a registry with mocked methods
        registry = ToolRegistry()

        # Mock the _discover_tool_modules method
        with (
            mock.patch.object(registry, "_discover_tool_modules") as mock_discover,
            mock.patch.object(registry, "_reload_tool") as mock_reload,
        ):
            # Mock the discovered tool modules
            tool_modules = {
                "tool1": Path("/path/to/tools/tool1.py"),
                "tool2": Path("/path/to/tools/tool2.py"),
                "__init__": Path("/path/to/tools/__init__.py"),
            }
            mock_discover.return_value = tool_modules

            # Add tools to the registry
            registry.agent_tools = {
                "tool1": MockAgentTool("tool1", {"name": "tool1", "description": "Tool 1"}),
                "tool2": MockAgentTool("tool2", {"name": "tool2", "description": "Tool 2"}),
            }

            # Update tools from the directory
            updated_tools = registry._update_tools_from_directory()

            # Verify the tools were updated
            self.assertEqual(mock_reload.call_count, 2)
            self.assertEqual(set(updated_tools), {"tool1", "tool2"})

    def test_reload_tool(self):
        """Test reloading a specific tool."""
        # Create a registry with mocked methods
        registry = ToolRegistry()

        # Mock the required methods
        with (
            mock.patch("pathlib.Path.exists") as mock_exists,
            mock.patch("importlib.util.spec_from_file_location") as mock_spec_from_file,
            mock.patch("importlib.util.module_from_spec") as mock_module_from_spec,
            mock.patch.object(registry, "_register_tool") as mock_register,
        ):
            # Set the tools directory directly on the registry
            tools_dir = Path("/path/to/tools")
            registry.tool_directories = [tools_dir]

            # Mock the tool file existence
            mock_exists.return_value = True

            # Mock the module spec and module
            mock_spec = mock.MagicMock()
            mock_spec.loader = mock.MagicMock()
            mock_spec_from_file.return_value = mock_spec

            mock_module = mock.MagicMock()
            mock_module.TOOL_SPEC = {"name": "tool1", "description": "Tool 1", "inputSchema": {"type": "object"}}
            mock_module.tool1 = lambda: None
            mock_module_from_spec.return_value = mock_module

            # Reload the tool
            registry._reload_tool("tool1")

            # Verify the tool was registered
            mock_register.assert_called_once()

    def test_scan_module_for_tools(self):
        """Test scanning a module for function-based tools."""
        from strands.tools.decorator import DecoratedFunctionTool

        # Create a mock module with a decorated function tool
        mock_module = mock.MagicMock()
        mock_tool = mock.MagicMock(spec=DecoratedFunctionTool)
        mock_tool.tool_name = "function_tool"

        # Add the tool to the module
        mock_module.function_tool = mock_tool

        # Scan the module for tools
        tools = self.registry._scan_module_for_tools(mock_module)

        # Verify the tool was found
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].tool_name, "function_tool")

    def test_validate_tool_spec(self):
        """Test validating tool specifications."""
        # Test with a valid tool spec
        valid_spec = {
            "name": "valid_tool",
            "description": "A valid tool",
            "inputSchema": {
                "type": "object",
                "properties": {"param1": {"type": "string", "description": "Parameter 1"}},
                "required": ["param1"],
            },
        }

        # This should not raise an exception
        self.registry._validate_tool_spec(valid_spec)

        # Test with an invalid tool spec (missing description)
        invalid_spec = {
            "name": "invalid_tool",
            "inputSchema": {
                "type": "object",
                "properties": {"param1": {"type": "string", "description": "Parameter 1"}},
                "required": ["param1"],
            },
        }

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.registry._validate_tool_spec(invalid_spec)


if __name__ == "__main__":
    unittest.main()
