ContextOffloader plugin for managing large tool outputs.

This module provides the ContextOffloader plugin that intercepts oversized tool results, persists each content block to a storage backend, and replaces the in-context result with a truncated preview and per-block references.

**Example**:

```python
from strands import Agent
from strands.vended_plugins.context_offloader import (
    ContextOffloader,
    InMemoryStorage,
    FileStorage,
)

# In-memory storage
agent = Agent(plugins=[
    ContextOffloader(storage=InMemoryStorage())
])

# File storage with custom thresholds and retrieval tool enabled
agent = Agent(plugins=[
    ContextOffloader(
        storage=FileStorage("./artifacts"),
        max_result_tokens=5_000,
        preview_tokens=2_000,
        include_retrieval_tool=True,
    )
])
```

## LineRange

```python
class LineRange(TypedDict)
```

Defined in: [src/strands/vended\_plugins/context\_offloader/plugin.py:124](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/plugin.py#L124)

A span of lines to retrieve (1-indexed, inclusive).

## ContextOffloader

```python
class ContextOffloader(Plugin)
```

Defined in: [src/strands/vended\_plugins/context\_offloader/plugin.py:141](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/plugin.py#L141)

Plugin that offloads oversized tool results to reduce context consumption.

When a tool result exceeds the configured token threshold, this plugin stores each content block individually to a storage backend and replaces the in-context result with a truncated text preview plus per-block references.

Token estimation uses the agent’s model `count_tokens` method, which leverages tiktoken when available and falls back to character-based heuristics.

Content type handling:

-   **Text**: stored as `text/plain`, replaced with a preview
-   **JSON**: stored as `application/json`, replaced with a preview
-   **Image**: stored in its native format (e.g., `image/png`), replaced with a placeholder showing format and size
-   **Document**: stored in its native format (e.g., `application/pdf`), replaced with a placeholder showing format, name, and size
-   **Unknown types**: passed through unchanged

This operates proactively at tool execution time via `AfterToolCallEvent`, before the result enters the conversation — unlike `SlidingWindowConversationManager` which truncates reactively after context overflow.

**Arguments**:

-   `storage` - Backend for storing offloaded content (required).
-   `max_result_tokens` - Offload results whose estimated token count exceeds this threshold.
-   `preview_tokens` - Number of tokens to keep as a text preview in context.
-   `include_retrieval_tool` - Whether to register the `retrieve_offloaded_content` tool. Defaults to True.

**Example**:

```python
from strands import Agent
from strands.vended_plugins.context_offloader import ContextOffloader, InMemoryStorage

agent = Agent(plugins=[
    ContextOffloader(storage=InMemoryStorage())
])
```

#### \_\_init\_\_

```python
def __init__(storage: Storage | _LegacyStorage,
             max_result_tokens: int = _DEFAULT_MAX_RESULT_TOKENS,
             preview_tokens: int = _DEFAULT_PREVIEW_TOKENS,
             *,
             include_retrieval_tool: bool = True,
             evict_after_cycles: int | None = 20) -> None
```

Defined in: [src/strands/vended\_plugins/context\_offloader/plugin.py:185](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/plugin.py#L185)

Initialize the ContextOffloader plugin.

**Arguments**:

-   `storage` - Backend for storing offloaded content. Accepts either a unified `Storage` (from `strands.storage`) or a legacy offloader `Storage` (from this module).
-   `max_result_tokens` - Offload results whose estimated token count exceeds this threshold. Defaults to `_DEFAULT_MAX_RESULT_TOKENS` (2,500).
-   `preview_tokens` - Number of tokens to keep as a text preview in context. Uses tiktoken for exact slicing when available, falls back to chars/4 heuristic. Defaults to `_DEFAULT_PREVIEW_TOKENS` (1,000).
-   `include_retrieval_tool` - Whether to register the `retrieve_offloaded_content` tool so the agent can fetch offloaded content. Defaults to True.
-   `evict_after_cycles` - Number of agent loop cycles before an offloaded entry is evicted (unified Storage only). Entries stored more than this many cycles ago are deleted. Defaults to 20. Set to None to disable eviction.

**Raises**:

-   `ValueError` - If max\_result\_tokens is not positive, preview\_tokens is negative, preview\_tokens >= max\_result\_tokens, or evict\_after\_cycles is invalid.

#### init\_agent

```python
def init_agent(agent: Agent) -> None
```

Defined in: [src/strands/vended\_plugins/context\_offloader/plugin.py:263](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/plugin.py#L263)

Conditionally register the retrieval tool and bind storage.

#### retrieve\_offloaded\_content

```python
@tool(context=True)
async def retrieve_offloaded_content(
        reference: str,
        tool_context: ToolContext,
        pattern: str | None = None,
        line_range: LineRange | None = None,
        context_lines: int | None = None) -> dict | str
```

Defined in: [src/strands/vended\_plugins/context\_offloader/plugin.py:305](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/plugin.py#L305)

Retrieve offloaded content by reference.

When a tool result was too large to keep in context, it was stored externally and replaced with a preview and a reference. Use this tool with that reference to access the stored content.

**Returns**:

-   With pattern: matching lines with line numbers and surrounding context
-   With line\_range: the specified span of lines with line numbers
-   Without pattern/line\_range: the full original content (use sparingly — re-injects all tokens)

Constraints:

-   pattern/line\_range/context\_lines only work on text content. For binary content, omit them.
-   Line numbers in results are 1-indexed and can be used in follow-up line\_range calls.

**Examples**:

-   `\{"reference"` - “ref\_1”, “pattern”: “error”} -> lines containing “error” with 5 lines context
-   `\{"reference"` - “ref\_1”, “pattern”: “error|warning”, “context\_lines”: 3} -> regex, 3 lines context
-   `\{"reference"` - “ref\_1”, “line\_range”: {“start”: 10, “end”: 25}} -> lines 10-25
-   `\{"reference"` - “ref\_1”, “pattern”: “TODO”, “line\_range”: {“start”: 1, “end”: 50}} -> search within range

**Arguments**:

-   `reference` - The reference string from the offload placeholder (e.g. “mem\_1\_tool-123\_0”).
-   `pattern` - Regex or keyword to grep for. Returns only matching lines with context — not the full content.
-   `line_range` - Return only this span of lines. A dict with ‘start’ and ‘end’ keys (1-indexed). Combine with pattern to search within the range.
-   `context_lines` - Lines before AND after each match (like grep -C). Default: 5. Without pattern/line\_range, returns first N lines.
-   `tool_context` - Injected by the framework. Not user-facing.