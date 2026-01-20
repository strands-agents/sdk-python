"""Tests for the LangChain tool wrapper."""

from typing import Optional, Type

import pytest
from langchain_core.tools import BaseTool, StructuredTool
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

    def _run(self, **kwargs: object) -> str:
        return self.return_value


class MockToolWithSchema(BaseTool):
    """Mock LangChain tool with args_schema."""

    name: str = "schema_tool"
    description: str = "A tool with schema"
    args_schema: Type[BaseModel] = MockArgsSchema

    def _run(self, query: str, max_results: int = 10) -> str:
        return f"Searched: {query}, max: {max_results}"


# LangChain tool with explicit schema
class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    a: int = Field(description="First number")
    b: int = Field(description="Second number")
    operation: str = Field(description="Operation: add, subtract, multiply, divide")


@langchain_tool(args_schema=CalculatorInput)
def calculator(a: int, b: int, operation: str) -> str:
    """Perform basic arithmetic operations."""
    if operation == "add":
        return f"Result: {a + b}"
    elif operation == "subtract":
        return f"Result: {a - b}"
    elif operation == "multiply":
        return f"Result: {a * b}"
    elif operation == "divide":
        return f"Result: {a / b}" if b != 0 else "Error: Division by zero"
    return f"Unknown operation: {operation}"


# BaseTool subclass with schema
class GreetingInput(BaseModel):
    """Input for greeting tool."""

    person_name: str = Field(description="Name of the person to greet")


class GreetingTool(BaseTool):
    """A tool that generates greetings."""

    name: str = "greeting"
    description: str = "Generate a greeting for a person"
    args_schema: type[BaseModel] = GreetingInput

    def _run(self, person_name: str) -> str:
        return f"Hello, {person_name}! Welcome!"


# StructuredTool.from_function()
def _reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


reverse_tool = StructuredTool.from_function(
    func=_reverse_string,
    name="reverse_string",
    description="Reverse the characters in a string",
)


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
    assert "Mock result" in result["content"][0]["text"]


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


# Tests for _convert_result_to_content


def test_convert_result_string() -> None:
    """Test converting a string result."""
    tool = LangChainTool(MockBaseTool())
    content = tool._convert_result_to_content("hello world")

    assert content == [{"text": "hello world"}]


def test_convert_result_non_string_raises() -> None:
    """Test that non-string results raise ValueError."""
    tool = LangChainTool(MockBaseTool())

    with pytest.raises(ValueError, match="Unsupported LangChain result type"):
        tool._convert_result_to_content({"key": "value"})

    with pytest.raises(ValueError, match="Unsupported LangChain result type"):
        tool._convert_result_to_content(42)


# Tests for tool_spec with different LangChain tool types


def test_tool_spec_with_schema() -> None:
    """Test tool_spec for a LangChain tool with explicit schema."""
    tool = LangChainTool(calculator)

    expected_spec = {
        "name": "calculator",
        "description": "Perform basic arithmetic operations.",
        "inputSchema": {
            "json": {
                "description": "Input for calculator tool.",
                "type": "object",
                "properties": {
                    "a": {
                        "title": "A",
                        "type": "integer",
                        "description": "First number",
                    },
                    "b": {
                        "title": "B",
                        "type": "integer",
                        "description": "Second number",
                    },
                    "operation": {
                        "title": "Operation",
                        "type": "string",
                        "description": "Operation: add, subtract, multiply, divide",
                    },
                },
                "required": ["a", "b", "operation"],
            }
        },
    }

    assert tool.tool_spec == expected_spec


def test_tool_spec_basetool_subclass() -> None:
    """Test tool_spec for a BaseTool subclass."""
    tool = LangChainTool(GreetingTool())

    expected_spec = {
        "name": "greeting",
        "description": "Generate a greeting for a person",
        "inputSchema": {
            "json": {
                "description": "Input for greeting tool.",
                "type": "object",
                "properties": {
                    "person_name": {
                        "title": "Person Name",
                        "type": "string",
                        "description": "Name of the person to greet",
                    },
                },
                "required": ["person_name"],
            }
        },
    }

    assert tool.tool_spec == expected_spec


def test_tool_spec_structured_tool() -> None:
    """Test tool_spec for StructuredTool.from_function()."""
    tool = LangChainTool(reverse_tool)

    assert tool.tool_name == "reverse_string"
    assert tool.tool_spec["description"] == "Reverse the characters in a string"
    assert "text" in tool.tool_spec["inputSchema"]["json"]["properties"]


# Tests for execution of different LangChain tool types


@pytest.mark.asyncio
async def test_stream_basetool_subclass() -> None:
    """Test stream execution of a BaseTool subclass."""
    tool = LangChainTool(GreetingTool())

    tool_use: ToolUse = {
        "toolUseId": "test-greeting",
        "name": "greeting",
        "input": {"person_name": "Alice"},
    }

    results = []
    async for event in tool.stream(tool_use, {}):
        results.append(event)

    assert len(results) == 1
    result = results[0]["tool_result"]
    assert result["status"] == "success"
    assert "Hello, Alice" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_stream_structured_tool() -> None:
    """Test stream execution of StructuredTool.from_function()."""
    tool = LangChainTool(reverse_tool)

    tool_use: ToolUse = {
        "toolUseId": "test-reverse",
        "name": "reverse_string",
        "input": {"text": "hello"},
    }

    results = []
    async for event in tool.stream(tool_use, {}):
        results.append(event)

    assert len(results) == 1
    result = results[0]["tool_result"]
    assert result["status"] == "success"
    assert "olleh" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_stream_tool_with_schema() -> None:
    """Test stream execution of a tool with explicit schema."""
    tool = LangChainTool(calculator)

    tool_use: ToolUse = {
        "toolUseId": "test-calc",
        "name": "calculator",
        "input": {"a": 10, "b": 5, "operation": "add"},
    }

    results = []
    async for event in tool.stream(tool_use, {}):
        results.append(event)

    assert len(results) == 1
    result = results[0]["tool_result"]
    assert result["status"] == "success"
    assert "15" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_stream_async_tool() -> None:
    """Test stream execution of an async @langchain_tool."""

    @langchain_tool
    async def async_uppercase(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()

    tool = LangChainTool(async_uppercase)

    tool_use: ToolUse = {
        "toolUseId": "test-async",
        "name": "async_uppercase",
        "input": {"text": "hello"},
    }

    results = []
    async for event in tool.stream(tool_use, {}):
        results.append(event)

    assert len(results) == 1
    result = results[0]["tool_result"]
    assert result["status"] == "success"
    assert "HELLO" in result["content"][0]["text"]
