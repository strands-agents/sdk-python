"""Shared constants for the programmatic tool caller."""

DEFAULT_PROGRAMMATIC_TOOL_CALLER_DESCRIPTION = (
    "Execute Python code that calls the agent's other tools as async functions. "
    'Each tool is an awaitable \u2014 always `await` it, e.g. `result = await calculator(expression="2 + 2")`. '
    "The code runs in an async context, so `await` and `asyncio.gather(...)` work without boilerplate. "
    "Only text sent to `print()` is returned to you \u2014 a tool's return value stays in the code's local "
    "scope unless you print it, and a tool that fails raises an exception you can catch with try/except. "
    "A tool whose name is not a valid Python identifier (for example `fetch-url` or `ns.fetch`) is also "
    "available with those characters replaced by underscores (`fetch_url`, `ns_fetch`). "
    "Use this to chain, loop over, or parallelize tool calls in a single turn instead of one tool call "
    "per model round-trip."
)
"""Default description for the programmatic tool caller, shown to the model."""
