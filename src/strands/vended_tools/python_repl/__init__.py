"""Python REPL tool for executing Python code in the agent's sandbox.

Example::

    from strands import Agent
    from strands.vended_tools import python_repl

    agent = Agent(tools=[python_repl])
    agent("Calculate the first 10 Fibonacci numbers")

Configuration via agent state::

    agent.state.set("strands_python_repl_tool", {
        "timeout": 30,  # Default timeout in seconds
    })
"""

from .python_repl import python_repl

__all__ = ["python_repl"]
