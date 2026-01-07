#!/usr/bin/env python3
"""Integration tests for LangChainTool with real agent interactions.

These tests verify that LangChain tools work correctly when invoked by an agent
through natural language, which requires actual model inference.
"""

from langchain_core.tools import tool as langchain_tool
from pydantic import BaseModel, Field

from strands import Agent
from strands.experimental.tools import LangChainTool


@langchain_tool
def word_count(text: str) -> str:
    """Count the number of words in text."""
    count = len(text.split())
    return f"The text contains {count} words."


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


def test_langchain_tool_natural_language():
    """Test LangChain tool invocation through natural language."""
    strands_tool = LangChainTool(word_count)
    agent = Agent(tools=[strands_tool])

    agent("Count the words in: 'The quick brown fox jumps over the lazy dog'")

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

    tool_results = [
        block["toolResult"]
        for message in agent.messages
        for block in message.get("content", [])
        if "toolResult" in block
    ]
    assert len(tool_results) > 0
    assert tool_results[0]["status"] == "success"
