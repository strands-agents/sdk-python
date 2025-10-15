from typing import Any

import pytest

from strands.types.tools import AgentTool, ToolSpec
from strands.types.content import ToolUse


class MockAgentTool(AgentTool):
    """Mock AgentTool implementation for testing."""
    
    def __init__(self, name: str):
        super().__init__()
        self._tool_name = name
    
    @property
    def tool_name(self) -> str:
        return self._tool_name
    
    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(name=self._tool_name, description="Mock tool", input_schema={})
    
    @property
    def tool_type(self) -> str:
        return "mock"
    
    def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any):
        yield f"Mock result for {self._tool_name}"
