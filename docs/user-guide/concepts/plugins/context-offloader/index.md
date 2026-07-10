The `ContextOffloader` plugin prevents large tool results from consuming your agent’s context window. When a tool returns a result that exceeds a configurable token threshold, the plugin stores each content block individually in an external storage backend and replaces it in the conversation with a truncated preview plus per-block references. Each offloaded result includes inline guidance telling the agent to use its available tools to selectively access the data it needs.

## The Problem

Tools like file readers, API clients, and database queries can return results that are tens or hundreds of thousands of characters long. When these large results enter the conversation, they crowd out other context and can exceed the model’s token limits.

The default [`SlidingWindowConversationManager`](/docs/user-guide/concepts/agents/conversation-management/index.md) handles this reactively — after the context overflows, it truncates tool results to the first and last 200 characters. This works as a safety net, but the truncation is lossy (the middle content is gone permanently) and happens after a failed API call has already been wasted.

`ContextOffloader` takes a proactive approach: it intercepts results at tool execution time, **before** they enter the conversation, so the overflow never happens in the first place.

## How It Works

After each tool call, the plugin estimates the result’s token count and compares it against the `max_result_tokens``maxResultTokens` threshold (default: 2,500 tokens). If the result exceeds it, the plugin:

1.  Stores each content block individually in the configured storage backend, preserving its content type
2.  Replaces the in-context result with the first `preview_tokens``previewTokens` tokens (default: 1,000) plus per-block storage references

Token estimation uses `model.count_tokens()``model.countTokens()`, which delegates to the model provider’s native counting API if available, otherwise falling back to a character-based heuristic (chars/4 for text, chars/2 for JSON).

Results under the threshold pass through unchanged.

### What the agent sees

For a tool that returns 150KB of JSON, the agent would see something like:

```plaintext
{"users": [{"id": 1, "name": "Alice", ...}, {"id": 2, "name": "Bob", ...},
... (first ~1,000 tokens of the result) ...

[Full content offloaded to storage - reference: a1b2c3d4]
```

For non-text content, the plugin replaces the result with a descriptive placeholder plus a reference:

| Content Type | What the agent sees |
| --- | --- |
| Text / JSON | First `preview_tokens``previewTokens` tokens + storage reference |
| Image | `[image: format, N bytes]` placeholder + storage reference |
| Document | `[document: format, name, N bytes]` placeholder + storage reference |

## Getting Started

Quick setup

You can enable a pre-configured `ContextOffloader` alongside summarization-based context management with a single parameter. See [Context Management](/docs/user-guide/concepts/context-management/index.md).

Pass a `ContextOffloader` instance to your agent’s `plugins` list with your choice of storage backend:

(( tab "Python" ))
```python
from strands import Agent
from strands.vended_plugins.context_offloader import (
    ContextOffloader,
    InMemoryStorage,
)

agent = Agent(plugins=[
    ContextOffloader(storage=InMemoryStorage())
])
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import {
  ContextOffloader,
  InMemoryStorage,
} from '@strands-agents/sdk/vended-plugins/context-offloader'

const agent = new Agent({
  plugins: [new ContextOffloader({ storage: new InMemoryStorage() })],
})
```
(( /tab "TypeScript" ))

To customize the token thresholds:

(( tab "Python" ))
```python
agent = Agent(plugins=[
    ContextOffloader(
        storage=InMemoryStorage(),
        max_result_tokens=5_000,
        preview_tokens=2_000,
    )
])
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import {
  ContextOffloader,
  InMemoryStorage,
} from '@strands-agents/sdk/vended-plugins/context-offloader'

const agent = new Agent({
  plugins: [
    new ContextOffloader({
      storage: new InMemoryStorage(),
      maxResultTokens: 5_000,
      previewTokens: 2_000,
    }),
  ],
})
```
(( /tab "TypeScript" ))

### Storage Backends

Choose a storage backend based on your needs:

| Backend | Persistence | Best for |
| --- | --- | --- |
| `InMemoryStorage` | Process lifetime only (auto-evicted after 20 idle cycles by default) | Development, testing, serverless, short-lived agents |
| `FileStorage` | Disk | Local development, debugging, inspecting stored artifacts |
| `S3Storage` | Amazon S3 | Production workloads, shared or durable artifact retention |

All backends implement the `Storage` protocol and preserve content type metadata, so you can also build your own.

**In-memory storage**: stores content in process memory. Entries not accessed within `evict_after_turns` agent loop cycles are automatically evicted to prevent unbounded memory growth. The default is 20 cycles. Retrieving content resets the counter for that entry, so frequently accessed content stays alive. To disable eviction and manage memory manually, pass `None`/`null` and call `clear()` when entries are no longer needed.

(( tab "Python" ))
```python
from strands import Agent
from strands.vended_plugins.context_offloader import (
    ContextOffloader,
    InMemoryStorage,
)

# Default: entries evicted after 20 cycles of disuse
agent = Agent(plugins=[
    ContextOffloader(storage=InMemoryStorage())
])

# Custom eviction window
agent = Agent(plugins=[
    ContextOffloader(storage=InMemoryStorage(evict_after_turns=50))
])

# Disable eviction (accumulates until clear() is called)
agent = Agent(plugins=[
    ContextOffloader(storage=InMemoryStorage(evict_after_turns=None))
])
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import {
  ContextOffloader,
  InMemoryStorage,
} from '@strands-agents/sdk/vended-plugins/context-offloader'

// Default: entries evicted after 20 cycles of disuse
const agent = new Agent({
  plugins: [new ContextOffloader({ storage: new InMemoryStorage() })],
})

// Custom eviction window
const agent2 = new Agent({
  plugins: [
    new ContextOffloader({ storage: new InMemoryStorage(50) }),
  ],
})

// Disable eviction (accumulates until clear() is called)
const agent3 = new Agent({
  plugins: [
    new ContextOffloader({ storage: new InMemoryStorage(null) }),
  ],
})
```
(( /tab "TypeScript" ))

**File storage** — persists to a local directory with `.metadata.json` sidecars for content type tracking:

(( tab "Python" ))
```python
from strands.vended_plugins.context_offloader import (
    ContextOffloader,
    FileStorage,
)

agent = Agent(plugins=[
    ContextOffloader(
        storage=FileStorage("./artifacts"),
    )
])
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import {
  ContextOffloader,
  FileStorage,
} from '@strands-agents/sdk/vended-plugins/context-offloader'

const agent = new Agent({
  plugins: [
    new ContextOffloader({
      storage: new FileStorage('./artifacts'),
    }),
  ],
})
```
(( /tab "TypeScript" ))

**S3 storage** — persists to an Amazon S3 bucket with content type preserved via S3 object metadata:

(( tab "Python" ))
```python
from strands.vended_plugins.context_offloader import (
    ContextOffloader,
    S3Storage,
)

agent = Agent(plugins=[
    ContextOffloader(
        storage=S3Storage(
            bucket="my-agent-artifacts",
            prefix="tool-results/",
        ),
    )
])
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import {
  ContextOffloader,
  S3Storage,
} from '@strands-agents/sdk/vended-plugins/context-offloader'

const agent = new Agent({
  plugins: [
    new ContextOffloader({
      storage: new S3Storage('my-agent-artifacts', {
        prefix: 'tool-results/',
      }),
    }),
  ],
})
```
(( /tab "TypeScript" ))

## Configuration

(( tab "Python" ))
| Parameter | Default | Description |
| --- | --- | --- |
| `storage` | *(required)* | Storage backend instance |
| `max_result_tokens` | `2_500` | Results whose estimated token count exceeds this are offloaded |
| `preview_tokens` | `1_000` | Number of tokens to keep as an in-context preview |
| `include_retrieval_tool` | `True` | Registers a `retrieve_offloaded_content` tool the agent can use to fetch full content by reference. Enabled by default; set to `False` to disable |
(( /tab "Python" ))

(( tab "TypeScript" ))
| Parameter | Default | Description |
| --- | --- | --- |
| `storage` | *(required)* | Storage backend instance |
| `maxResultTokens` | `2_500` | Results whose estimated token count exceeds this are offloaded |
| `previewTokens` | `1_000` | Number of tokens to keep as an in-context preview |
| `includeRetrievalTool` | `true` | Registers a `retrieve_offloaded_content` tool the agent can use to fetch full content by reference. Enabled by default; set to `false` to disable |
(( /tab "TypeScript" ))

## Retrieval Tool

The plugin includes a `retrieve_offloaded_content` tool that lets the agent fetch offloaded content by reference, returning it in its native format — text as a string, JSON as a JSON block, images as image blocks, and documents as document blocks. This tool is registered by default.

The retrieval tool supports targeted retrieval through optional parameters, so the agent can search and filter offloaded content without loading it entirely back into context.

**Parameters:**

(( tab "Python" ))
| Parameter | Type | Description |
| --- | --- | --- |
| `reference` | `str` | *(required)* Storage reference from the offloaded result |
| `pattern` | `str` | Regex or keyword to grep for |
| `line_range` | `dict` with `start` and `end` keys | 1-indexed inclusive line span to retrieve |
| `context_lines` | `int` | Lines of context around pattern matches (default: 5) |
(( /tab "Python" ))

(( tab "TypeScript" ))
| Parameter | Type | Description |
| --- | --- | --- |
| `reference` | `string` | *(required)* Storage reference from the offloaded result |
| `pattern` | `string` | Regex or keyword to grep for |
| `line_range` | `{ start: number; end: number }` | 1-indexed inclusive line span to retrieve |
| `context_lines` | `number` | Lines of context around pattern matches (default: 5) |
(( /tab "TypeScript" ))

**Retrieval modes:**

-   **Pattern search** — Provide `pattern` to grep for regex/keyword matches with configurable `context_lines`
-   **Line range** — Provide `line_range` for random access to specific line numbers
-   **Combined** — Provide both `pattern` and `line_range` to search within a specific range
-   **Head** — Provide only `context_lines` without a `pattern` or `line_range` to return the first N lines of the content
-   **Full retrieval** — Omit all optional parameters to retrieve everything (discouraged for large content)

Results include line numbers to enable follow-up queries. Large result sets are truncated with guidance to narrow the search. Binary content cannot be searched — pattern and line range parameters return an error for binary references.

### Retrieval examples

**1\. Tool result gets offloaded (replaces original result inline)**

```plaintext
[Offloaded: 1 blocks, ~10,000 tokens]
Tool result was offloaded to external storage due to size.
Use the preview below if it answers your question.
If you need more detail, use retrieve_offloaded_content with a reference and:
  - pattern: regex or keyword to find matching lines with context
  - line_range: { start, end } to read a specific span of lines
Retrieve full content (omit pattern/line_range) as a last resort.

{"users":[{"id":1,"name":"Alice","role":"admin"},{"id":2,"name":"Bob","role":"user"},{"id":3,"name":"Charlie","rol

[Stored references:]
  mem_1_tool-123_0 (json, 42,000 bytes)
```

**2\. Agent searches with a pattern**

Input: `{ reference: "mem_1_tool-123_0", pattern: "admin", context_lines: 2 }`

```plaintext
[2 matches for /admin/]

   1| {
   2|   "users": [
>  3|     { "id": 1, "name": "Alice", "role": "admin" },
   4|     { "id": 2, "name": "Bob", "role": "user" },
   5|     { "id": 3, "name": "Charlie", "role": "user" },
---
  48|     { "id": 15, "name": "Dana", "role": "user" },
> 49|     { "id": 16, "name": "Eve", "role": "admin" },
  50|     { "id": 17, "name": "Frank", "role": "user" }
  51|   ]
```

**3\. Agent retrieves a line range**

Input: `{ reference: "mem_1_tool-123_0", line_range: { start: 45, end: 55 } }`

```plaintext
[Lines 45-55 of 120]

  45|     { "id": 14, "name": "Carol", "role": "user" },
  46|     { "id": 15, "name": "Dana", "role": "user" },
  47|     { "id": 16, "name": "Eve", "role": "admin" },
  48|     { "id": 17, "name": "Frank", "role": "user" },
  49|     { "id": 18, "name": "Grace", "role": "user" },
  50|     { "id": 19, "name": "Hank", "role": "member" },
  51|     { "id": 20, "name": "Ivy", "role": "user" },
  52|     { "id": 21, "name": "Jack", "role": "user" },
  53|     { "id": 22, "name": "Kate", "role": "user" },
  54|     { "id": 23, "name": "Leo", "role": "user" },
  55|     { "id": 24, "name": "Mia", "role": "user" },
```

### Using other tools for retrieval

When using `FileStorage`, the agent can use its existing tools (shell, grep, cat, etc.) to access offloaded content directly from the file system. The offloaded guidance includes the full storage path, so the agent knows where to look:

```plaintext
grep -n "admin" ./artifacts/mem_1_tool-123_0
cat ./artifacts/mem_1_tool-123_0 | head -50
sed -n '45,55p' ./artifacts/mem_1_tool-123_0
```

With `S3Storage`, the agent can use the AWS CLI to access offloaded content:

```plaintext
aws s3 cp s3://my-agent-artifacts/tool-results/mem_1_tool-123_0 - | grep -n "admin"
aws s3 cp s3://my-agent-artifacts/tool-results/mem_1_tool-123_0 - | head -50
```

With `InMemoryStorage`, there is no external access path: the built-in retrieval tool is the only way to access offloaded content, so keep it enabled. Entries are automatically evicted after 20 cycles of disuse (configurable via `evict_after_turns` / `evictAfterTurns`). Attempting to retrieve evicted content raises an error.

This approach is often preferable because the agent already knows these tools well and can chain them together for complex queries. To disable the built-in retrieval tool and rely on the agent’s own tools:

(( tab "Python" ))
```python
from strands_tools import shell

agent = Agent(
    tools=[shell],
    plugins=[
        ContextOffloader(
            storage=FileStorage("./artifacts"),
            include_retrieval_tool=False,
        )
    ]
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent } from '@strands-agents/sdk'
import { ContextOffloader, FileStorage } from '@strands-agents/sdk/vended-plugins/context-offloader'
import { bash } from '@strands-agents/sdk/vended-tools/bash'
import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'

const agent = new Agent({
  tools: [bash, fileEditor],
  plugins: [
    new ContextOffloader({
      storage: new FileStorage('./artifacts'),
      includeRetrievalTool: false,
    }),
  ],
})
```
(( /tab "TypeScript" ))

## Tradeoffs

-   **Preview vs. full content**: The agent reasons over the preview, not the full result. If the answer is buried deep in a large result, the agent may miss it. Tune `preview_tokens` to balance context usage against information loss for your use case. The `retrieve_offloaded_content` tool is enabled by default so the agent can fetch full offloaded content as a fallback. If the agent already has tools that can access the storage backend directly (file readers, shell, etc.), you can disable it with `include_retrieval_tool=False`.
-   **Storage costs**: `S3Storage` incurs S3 PUT/GET and storage charges. `FileStorage` writes to disk on every large result.
-   **In-memory eviction**: `InMemoryStorage` evicts entries not accessed within 20 agent loop cycles (configurable). Evicted content is permanently deleted: the agent receives an error if it tries to retrieve it later. Increase the value or pass `None`/`null` to disable eviction if your agent revisits offloaded content after many turns.
-   **Not a replacement for conversation management**: This plugin handles individual large results. You still need a conversation manager like `SlidingWindowConversationManager` to handle overall context growth across many turns.

## Related pages

- [Context Management](/docs/user-guide/concepts/context-management/index.md) (2 shared tags)
- [Conversation Management](/docs/user-guide/concepts/agents/conversation-management/index.md) (2 shared tags)
- [Steering](/docs/user-guide/concepts/plugins/steering/index.md) (2 shared tags)
- [Context Injector](/docs/user-guide/concepts/plugins/context-injector/index.md) (1 shared tag)
- [Coherence Evaluator](/docs/user-guide/evals-sdk/evaluators/coherence_evaluator/index.md) (1 shared tag)
- [Conciseness Evaluator](/docs/user-guide/evals-sdk/evaluators/conciseness_evaluator/index.md) (1 shared tag)
- [Goal Success Rate Evaluator](/docs/user-guide/evals-sdk/evaluators/goal_success_rate_evaluator/index.md) (1 shared tag)
- [Helpfulness Evaluator](/docs/user-guide/evals-sdk/evaluators/helpfulness_evaluator/index.md) (1 shared tag)
- [Interactions Evaluator](/docs/user-guide/evals-sdk/evaluators/interactions_evaluator/index.md) (1 shared tag)
- [Output Evaluator](/docs/user-guide/evals-sdk/evaluators/output_evaluator/index.md) (1 shared tag)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/vended_plugins/context_offloader/plugin.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/plugin.py)
- [harness-sdk/strands-py/src/strands/vended_plugins/context_offloader/storage.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py)

### TypeScript

- [harness-sdk/strands-ts/src/vended-plugins/context-offloader/plugin.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-plugins/context-offloader/plugin.ts)
- [harness-sdk/strands-ts/src/vended-plugins/context-offloader/storage.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-plugins/context-offloader/storage.ts)
