"""CodeAct plugin for code-based agent interaction.

CodeAct replaces standard tool calling with code-based orchestration.
The model generates Python code that calls tools as async functions,
and the plugin executes this code and feeds results back to the model.

Example:
    ```python
    from strands import Agent
    from strands.vended_plugins.codeact import CodeActPlugin

    agent = Agent(
        tools=[shell, calculator],
        plugins=[CodeActPlugin()],
    )

    result = agent("Calculate squares of 1-10 and sum them")
    ```
"""

from .codeact_plugin import CodeActPlugin

__all__ = ["CodeActPlugin"]
