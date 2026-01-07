#!/usr/bin/env python3
"""Integration tests for LangChainTool with real agent interactions."""

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools import tool as langchain_tool
from pydantic import BaseModel, Field

from strands import Agent
from strands.experimental.tools import LangChainTool


# Simple LangChain tool using @tool decorator (sync)
@langchain_tool
def word_count(text: str) -> str:
    """Count the number of words in text."""
    count = len(text.split())
    return f"The text contains {count} words."


# Async LangChain tool using @tool decorator
@langchain_tool
async def async_uppercase(text: str) -> str:
    """Convert text to uppercase asynchronously."""
    return text.upper()


# LangChain tool with schema
class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    a: int = Field(description="First number")
    b: int = Field(description="Second number")
    operation: str = Field(description="Operation: add, subtract, multiply, divide")


@langchain_tool(args_schema=CalculatorInput)
def calculator(a: int, b: int, operation: str) -> str:
    """Perform basic arithmetic operations."""
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero"
        result = a / b
    else:
        return f"Unknown operation: {operation}"
    return f"Result: {result}"


# LangChain BaseTool subclass (async)
class GreetingTool(BaseTool):
    """A tool that generates greetings."""

    name: str = "greeting"
    description: str = "Generate a greeting for a person"

    def _run(self, name: str) -> str:
        return f"Hello, {name}! Welcome!"

    async def _arun(self, name: str) -> str:
        return f"Hello, {name}! Welcome!"


# StructuredTool.from_function()
def _reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


reverse_tool = StructuredTool.from_function(
    func=_reverse_string,
    name="reverse_string",
    description="Reverse the characters in a string",
)


def test_langchain_tool_direct_invocation():
    """Test direct tool invocation through agent.tool interface."""
    strands_tool = LangChainTool(word_count)
    agent = Agent(tools=[strands_tool])

    result = agent.tool.word_count(text="Hello world this is a test")

    assert result["status"] == "success"
    assert "6 words" in result["content"][0]["text"]


def test_langchain_tool_with_schema_direct():
    """Test LangChain tool with schema through direct invocation."""
    strands_tool = LangChainTool(calculator)
    agent = Agent(tools=[strands_tool])

    result = agent.tool.calculator(a=10, b=5, operation="add")

    assert result["status"] == "success"
    assert "15" in result["content"][0]["text"]


def test_langchain_basetool_subclass():
    """Test LangChain BaseTool subclass."""
    greeting_tool = GreetingTool()
    strands_tool = LangChainTool(greeting_tool)
    agent = Agent(tools=[strands_tool])

    result = agent.tool.greeting(name="Alice")

    assert result["status"] == "success"
    assert "Hello, Alice" in result["content"][0]["text"]


def test_structured_tool_from_function():
    """Test StructuredTool.from_function()."""
    strands_tool = LangChainTool(reverse_tool)
    agent = Agent(tools=[strands_tool])

    result = agent.tool.reverse_string(text="hello")

    assert result["status"] == "success"
    assert "olleh" in result["content"][0]["text"]


def test_async_langchain_tool():
    """Test async @langchain_tool decorated function."""
    strands_tool = LangChainTool(async_uppercase)
    agent = Agent(tools=[strands_tool])

    result = agent.tool.async_uppercase(text="hello world")

    assert result["status"] == "success"
    assert "HELLO WORLD" in result["content"][0]["text"]


def test_langchain_tool_natural_language():
    """Test LangChain tool invocation through natural language."""
    strands_tool = LangChainTool(word_count)
    agent = Agent(tools=[strands_tool])

    agent("Count the words in: 'The quick brown fox jumps over the lazy dog'")

    # Verify the agent used the tool
    tool_results = [
        block["toolResult"]
        for message in agent.messages
        for block in message.get("content", [])
        if "toolResult" in block
    ]
    assert len(tool_results) > 0
    assert tool_results[0]["status"] == "success"


def test_langchain_tool_calculator_natural_language():
    """Test calculator tool through natural language."""
    strands_tool = LangChainTool(calculator)
    agent = Agent(tools=[strands_tool])

    agent("What is 25 multiplied by 4? Use the calculator tool.")

    # Verify the agent used the tool
    tool_results = [
        block["toolResult"]
        for message in agent.messages
        for block in message.get("content", [])
        if "toolResult" in block
    ]
    assert len(tool_results) > 0
    assert tool_results[0]["status"] == "success"


def test_multiple_langchain_tools():
    """Test agent with multiple LangChain tools."""
    tools = [
        LangChainTool(word_count),
        LangChainTool(calculator),
        LangChainTool(GreetingTool()),
    ]
    agent = Agent(tools=tools)

    # Test each tool directly
    word_result = agent.tool.word_count(text="one two three")
    assert word_result["status"] == "success"
    assert "3 words" in word_result["content"][0]["text"]

    calc_result = agent.tool.calculator(a=7, b=3, operation="subtract")
    assert calc_result["status"] == "success"
    assert "4" in calc_result["content"][0]["text"]

    greet_result = agent.tool.greeting(name="Bob")
    assert greet_result["status"] == "success"
    assert "Hello, Bob" in greet_result["content"][0]["text"]


def test_tool_spec_simple_tool_in_registry():
    """Test that a simple LangChain tool's spec is correctly registered in the agent."""
    strands_tool = LangChainTool(word_count)
    agent = Agent(tools=[strands_tool])

    tools_config = agent.tool_registry.get_all_tools_config()

    expected_spec = {
        "name": "word_count",
        "description": "Count the number of words in text.",
        "inputSchema": {
            "json": {
                "description": "Count the number of words in text.",
                "type": "object",
                "properties": {
                    "text": {
                        "title": "Text",
                        "type": "string",
                        "description": "Property text",
                    }
                },
                "required": ["text"],
            }
        },
    }

    assert tools_config["word_count"] == expected_spec


def test_tool_spec_with_schema_in_registry():
    """Test that a LangChain tool with schema has correct properties in the registry."""
    strands_tool = LangChainTool(calculator)
    agent = Agent(tools=[strands_tool])

    tools_config = agent.tool_registry.get_all_tools_config()

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

    assert tools_config["calculator"] == expected_spec


def test_tool_spec_basetool_subclass_in_registry():
    """Test that a BaseTool subclass spec is correctly registered."""
    greeting_tool = GreetingTool()
    strands_tool = LangChainTool(greeting_tool)
    agent = Agent(tools=[strands_tool])

    tools_config = agent.tool_registry.get_all_tools_config()

    expected_spec = {
        "name": "greeting",
        "description": "Generate a greeting for a person",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        },
    }

    assert tools_config["greeting"] == expected_spec
