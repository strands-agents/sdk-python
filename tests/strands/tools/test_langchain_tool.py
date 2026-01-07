"""Tests for the LangChain tool wrapper."""

from typing import Optional, Type

import pytest
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as langchain_tool
from pydantic import BaseModel, Field

from strands.experimental.tools import LangChainTool
from strands.types.tools import ToolUse


class MockArgsSchema(BaseModel):
    """Mock Pydantic schema for testing."""

    query: str = Field(description="The search query")
    max_results: int = Field(default=10, description="Maximum number of results")


class MockBaseTool(BaseTool):
    """Mock LangChain BaseTool for testing."""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    args_schema: Optional[Type[BaseModel]] = None
    return_value: str = "Mock result"
    should_error: bool = False

    def _run(self, **kwargs: object) -> str:
        if self.should_error:
            raise ValueError("Mock error")
        return self.return_value

    async def _arun(self, **kwargs: object) -> str:
        if self.should_error:
            raise ValueError("Mock async error")
        return f"Async: {self.return_value}"


class MockToolWithSchema(BaseTool):
    """Mock LangChain tool with args_schema."""

    name: str = "schema_tool"
    description: str = "A tool with schema"
    args_schema: Type[BaseModel] = MockArgsSchema

    def _run(self, query: str, max_results: int = 10) -> str:
        return f"Searched: {query}, max: {max_results}"


# Tests for _build_input_schema


def test_build_input_schema_no_schema() -> None:
    """Test schema building with no args_schema."""
    tool = MockBaseTool()
    result = LangChainTool._build_input_schema(tool)

    assert result == {
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_build_input_schema_with_schema() -> None:
    """Test schema building with args_schema."""
    tool = MockToolWithSchema()
    result = LangChainTool._build_input_schema(tool)

    assert result["type"] == "object"
    assert "properties" in result
    assert "query" in result["properties"]
    assert "max_results" in result["properties"]


# Tests for LangChainTool


def test_langchain_tool_init() -> None:
    """Test LangChainTool initialization."""
    mock_tool = MockBaseTool()
    tool = LangChainTool(mock_tool)

    assert tool.tool_name == "mock_tool"
    assert tool.tool_type == "langchain"
    assert tool.wrapped_tool is mock_tool


def test_langchain_tool_spec() -> None:
    """Test tool spec generation."""
    mock_tool = MockToolWithSchema()
    tool = LangChainTool(mock_tool)

    spec = tool.tool_spec
    assert spec["name"] == "schema_tool"
    assert spec["description"] == "A tool with schema"
    assert "inputSchema" in spec
    assert "json" in spec["inputSchema"]


# Tests for stream execution


@pytest.mark.asyncio
async def test_langchain_tool_stream_async() -> None:
    """Test stream with async execution."""
    mock_tool = MockBaseTool()
    tool = LangChainTool(mock_tool)

    tool_use: ToolUse = {
        "toolUseId": "test-123",
        "name": "mock_tool",
        "input": {},
    }

    results = []
    async for event in tool.stream(tool_use, {}):
        results.append(event)

    assert len(results) == 1
    result = results[0]["tool_result"]
    assert result["toolUseId"] == "test-123"
    assert result["status"] == "success"
    assert "Async: Mock result" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_langchain_tool_stream_with_input() -> None:
    """Test stream passes input correctly."""
    mock_tool = MockToolWithSchema()
    tool = LangChainTool(mock_tool)

    tool_use: ToolUse = {
        "toolUseId": "test-input",
        "name": "schema_tool",
        "input": {"query": "hello", "max_results": 5},
    }

    results = []
    async for event in tool.stream(tool_use, {}):
        results.append(event)

    assert len(results) == 1
    result = results[0]["tool_result"]
    assert result["status"] == "success"


# Test with @tool decorator


def test_langchain_tool_with_decorator() -> None:
    """Test wrapping a tool created with @tool decorator."""

    @langchain_tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    tool = LangChainTool(search)

    assert tool.tool_name == "search"
    assert "Search for information" in tool.tool_spec["description"]


@pytest.mark.asyncio
async def test_langchain_tool_decorator_execution() -> None:
    """Test executing a tool created with @tool decorator."""

    @langchain_tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    tool = LangChainTool(greet)

    tool_use: ToolUse = {
        "toolUseId": "test-greet",
        "name": "greet",
        "input": {"name": "World"},
    }

    results = []
    async for event in tool.stream(tool_use, {}):
        results.append(event)

    assert len(results) == 1
    result = results[0]["tool_result"]
    assert result["status"] == "success"
    assert "Hello, World!" in result["content"][0]["text"]
