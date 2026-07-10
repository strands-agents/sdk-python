"""Shared constants for the programmatic tool caller."""

DEFAULT_PROGRAMMATIC_TOOL_CALLER_DESCRIPTION = (
    "Execute Python code that calls the agent's other tools as async functions. "
    'Each tool is exposed as an awaitable, e.g. `result = await calculator(expression="2 + 2")`; '
    "the code runs in an async context, so `await` and `asyncio.gather(...)` work without boilerplate. "
    "Only text sent to `print()` is returned to you \u2014 tool results stay in the code's local scope "
    "unless you print them. Use this to chain, loop over, or parallelize tool calls in a single turn "
    "instead of one tool call per model round-trip."
)
"""Default description for the programmatic tool caller, shown to the model."""
