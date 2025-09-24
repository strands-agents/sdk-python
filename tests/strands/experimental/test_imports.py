"""Tests for experimental namespace imports."""


class TestExperimentalImports:
    """Test experimental namespace imports."""
    
    def test_import_agent_config(self):
        """Test importing AgentConfig from experimental namespace."""
        from strands.experimental import AgentConfig
        assert AgentConfig is not None
    
    def test_import_tool_pool(self):
        """Test importing ToolBox from experimental namespace."""
        from strands.experimental import ToolBox
        assert ToolBox is not None
    
    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import strands.experimental
        
        expected_exports = {"AgentConfig", "ToolBox"}
        actual_exports = set(strands.experimental.__all__)
        
        assert expected_exports == actual_exports
    
    def test_direct_module_imports(self):
        """Test importing modules directly."""
        from strands.experimental.agent_config import AgentConfig
        from strands.experimental.tool_box import ToolBox
        
        assert AgentConfig is not None
        assert ToolBox is not None
