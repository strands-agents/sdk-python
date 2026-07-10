By default a Strands agent starts every conversation from zero: it cannot recall a user’s preferences, past decisions, or anything it learned in an earlier session. The `MemoryManager` gives an agent long-term memory that persists across sessions.

It works through **memory stores**, the backends that hold the memories. A store can be a vector database, a managed service like [Amazon Bedrock Knowledge Bases](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md), or [your own implementation](#custom-stores). The manager handles three jobs across the stores you give it:

1.  **Recall** - the agent searches stored knowledge on demand through a tool.
2.  **Injection** - the manager folds relevant knowledge into the prompt automatically, before the model runs.
3.  **Extraction** - turning conversation messages into memories and writing them to stores.

Recall and injection are enabled by default when you attach a store. Extraction and fact storage through tools is opt-in.

## Getting Started

Attach a memory manager to an agent through the `memory_manager``memoryManager` parameter. The examples below use a `store`, a `MemoryStore` you provide; see [Bedrock Knowledge Base](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md) for a managed backend or [Custom Stores](#custom-stores) to create your own.

(( tab "Python" ))
```python
from strands import Agent
from strands.memory import MemoryManager

agent = Agent(memory_manager=MemoryManager(stores=[store]))
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent, BedrockModel } from '@strands-agents/sdk'
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'preferences',
  description: 'User preferences and stable facts about the user.',
  writable: true,
  config: { knowledgeBaseId: 'KB123', dataSourceType: 'CUSTOM', dataSourceId: 'DS456' },
})

const agent = new Agent({
  model: new BedrockModel(),
  memoryManager: { stores: [store] },
})
```
(( /tab "TypeScript" ))

With no further configuration, reading through recall and injection is enabled. Writing is opt-in, and comes in two modes:

-   **The `add_memory` tool** lets the agent decide what to save. Enable it on the manager with `add_tool_config=True``addToolConfig: true`.
-   **Automatic extraction** captures memories from the conversation without tool call. Enable it on a writable store, where it runs every 5 turns by default using your agent’s model.

(( tab "Python" ))
```python
from strands import Agent
from strands.memory import MemoryManager
from strands.vended_memory_stores import BedrockKnowledgeBaseStore

store = BedrockKnowledgeBaseStore(
    name="preferences",
    writable=True,
    extraction=True,  # capture memories from the conversation, every 5 turns
    config={"knowledge_base_id": "KB123", "data_source_type": "CUSTOM", "data_source_id": "DS456"},
)

agent = Agent(
    memory_manager=MemoryManager(
        stores=[store],
        add_tool_config=True,  # let the agent save memories itself
    ),
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent, BedrockModel } from '@strands-agents/sdk'
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'preferences',
  writable: true,
  extraction: true, // capture memories from the conversation, every 5 turns
  config: { knowledgeBaseId: 'KB123', dataSourceType: 'CUSTOM', dataSourceId: 'DS456' },
})

const agent = new Agent({
  model: new BedrockModel(),
  memoryManager: {
    stores: [store],
    addToolConfig: true, // let the agent save memories itself
  },
})
```
(( /tab "TypeScript" ))

## Stores

A manager can own several stores at once, which keeps multi-tenancy out of your application code. A single agent can query personal, team, and organization knowledge together, with each store scoped to its own tenant:

(( tab "Python" ))
```python
from strands import Agent
from strands.memory import MemoryManager

# personal and team are two MemoryStore instances, each scoped to its own tenant.
agent = Agent(memory_manager=MemoryManager(stores=[personal, team]))
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent, BedrockModel } from '@strands-agents/sdk'
import {
  BedrockKnowledgeBaseStore,
  type BedrockKnowledgeBaseConfig,
} from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

// Build the connection once, vary only name and scope per store.
const connection: BedrockKnowledgeBaseConfig = {
  knowledgeBaseId: 'KB123',
  dataSourceType: 'CUSTOM',
  dataSourceId: 'DS456',
}

const personal = new BedrockKnowledgeBaseStore({
  name: 'personal',
  description: 'Knowledge specific to this user.',
  writable: true,
  scope: 'user-abc',
  config: connection,
})

const team = new BedrockKnowledgeBaseStore({
  name: 'team',
  description: 'Shared team knowledge.',
  scope: 'team-xyz',
  config: connection,
})

const agent = new Agent({
  model: new BedrockModel(),
  memoryManager: { stores: [personal, team] },
})
```
(( /tab "TypeScript" ))

Each store carries its own identity and behavior:

| Field | Purpose |
| --- | --- |
| `name` | Unique identifier, used to target the store from tools and the programmatic API. |
| `description` | Human-readable summary, surfaced in the memory tool descriptions so the model knows what each store holds. |
| `max_search_results``maxSearchResults` | Default result cap per search when a caller does not pass one. The manager falls back to `3` if neither is set. |
| `writable` | Whether the store accepts writes. |

The manager attaches each store’s `name` to its results, so the model and your code can tell which store produced each entry and target follow-up queries.

## Memory Tools

The manager can register two tools the agent can call during the loop, both configurable. `search_memory` is registered for you; `add_memory` is opt-in:

(( tab "Python" ))
```python
from strands import Agent
from strands.memory import MemoryManager
from strands.memory.types import MemoryAddToolConfig, MemoryToolConfig

agent = Agent(
    memory_manager=MemoryManager(
        stores=[store],
        search_tool_config=MemoryToolConfig(
            name="recall",
            description="Look up what you remember about the user.",
        ),
        # opt in, and return as soon as writes dispatch instead of awaiting them
        add_tool_config=MemoryAddToolConfig(wait_for_writes=False),
    ),
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent, BedrockModel } from '@strands-agents/sdk'
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'preferences',
  config: { knowledgeBaseId: 'KB123' },
})

const agent = new Agent({
  model: new BedrockModel(),
  memoryManager: {
    stores: [store],
    searchToolConfig: {
      name: 'recall',
      description: 'Look up what you remember about the user.',
    },
    // add_memory: opt in, and return as soon as writes dispatch instead of awaiting them
    addToolConfig: { waitForWrites: false },
  },
})
```
(( /tab "TypeScript" ))

`search_memory` lets the agent recall knowledge on demand. Rename or re-describe it through its config, or turn it off. When the manager owns multiple stores, their names and descriptions are folded into the tool description so the model can target a specific store by name or search them all.

`add_memory` lets the agent write new memories. Enable it to allow writes to your writable stores, or pass a config to scope it to specific ones. By default it waits for writes so it can report failures back to the model. The fire-and-forget option returns as soon as writes are dispatched, so a slow backend never blocks the agent loop. This tool can only targets stores implementing `add``add`.

## Context Injection

Injection searches memory before a model call and folds the top results into the prompt, so relevant knowledge is present on every turn. It is **on by default**: the manager injects on a fresh user turn, retrieves up to 5 entries, derives the query adaptively from the latest user message, and renders the results as a `<memory>` block. Turn it off by disabling the injection config.

The injected text is **ephemeral by design**: it augments the model input for a single call and never persists into the durable conversation or session.

Customize retrieval, timing, and formatting with a config object:

(( tab "Python" ))
```python
from strands import Agent
from strands.memory import MemoryManager
from strands.memory.types import MemoryInjectionConfig

agent = Agent(
    memory_manager=MemoryManager(
        stores=[store],
        injection=MemoryInjectionConfig(
            trigger="everyTurn",  # inject before every model call
            max_entries=3,
            format=lambda context: "\n".join(f"- {entry.content}" for entry in context.entries),
        ),
    ),
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent, BedrockModel, type MessageData } from '@strands-agents/sdk'
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'preferences',
  config: { knowledgeBaseId: 'KB123' },
})

const agent = new Agent({
  model: new BedrockModel(),
  memoryManager: {
    stores: [store],
    injection: {
      // 'userTurn' (default), 'everyTurn', or a predicate over the conversation
      trigger: ({ messages }) => messages.length >= 4,
      maxEntries: 3,
      query: ({ messages }: { messages: MessageData[] }) => {
        const block = messages.at(-1)?.content[0]
        return block && 'text' in block ? block.text : undefined
      },
      format: ({ entries }) => entries.map((entry) => `- ${entry.content}`).join('\n'),
    },
  },
})
```
(( /tab "TypeScript" ))

-   `trigger` accepts `'userTurn'` (the default, inject only on a fresh user ask), `'everyTurn'` (inject before every model call, for autonomous agents), or a predicate: a function that receives the injection context and returns whether to inject this call.
-   `max_entries``maxEntries` caps how many entries are retrieved and injected.
-   `query` overrides the adaptive default with your own query logic. Return an empty value to skip injection for this call.
-   `format` renders the retrieved entries. The default emits an escaped `<memory>` block; a custom formatter that emits markup owns its own escaping.

Injection fails open: if the search fails or a callback throws, the manager logs it and proceeds with the model call uninjected. The agent runs without the memory context rather than erroring, so a backend outage degrades silently.

### The injection engine is generic

Memory injection is built on a reusable engine. For non-memory context (a clock, a sandbox descriptor, a fixed reminder), the same mechanism is exposed as the [`ContextInjector`](/docs/user-guide/concepts/plugins/context-injector/index.md) vended plugin: supply a render callback and it folds the result into the model input the same way.

## Automatic Extraction

Extraction captures memories from the conversation automatically, instead of relying on the agent to call the `add_memory` tool. Enable it on a writable store:

(( tab "Python" ))
```python
from strands.vended_memory_stores import BedrockKnowledgeBaseStore

# Set extraction on the store at construction; True uses the defaults.
store = BedrockKnowledgeBaseStore(
    name="preferences",
    writable=True,
    extraction=True,
    config={"knowledge_base_id": "KB123", "data_source_type": "CUSTOM", "data_source_id": "DS456"},
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'preferences',
  writable: true,
  extraction: true, // extract every 5 turns with a ModelExtractor
  config: { knowledgeBaseId: 'KB123', dataSourceType: 'CUSTOM', dataSourceId: 'DS456' },
})
```
(( /tab "TypeScript" ))

With defaults, extraction runs every 5 turns. For a store that implements only `add``add` (like Bedrock Knowledge Bases), it uses a `ModelExtractor` to distill facts from the conversation; a store that implements `add_messages``addMessages` extracts server-side instead, covered under [Custom Stores](#custom-stores).

### Triggers and extractors

An extraction config has two parts. A **trigger** decides *when* extraction runs; an **extractor** decides *how* messages become entries:

(( tab "Python" ))
```python
from strands.models import BedrockModel
from strands.memory.extraction.triggers import InvocationTrigger
from strands.memory.extraction.types import ExtractionConfig
from strands.memory.extraction.model_extractor import ModelExtractor
from strands.vended_memory_stores import BedrockKnowledgeBaseStore

store = BedrockKnowledgeBaseStore(
    name="preferences",
    writable=True,
    extraction=ExtractionConfig(
        trigger=InvocationTrigger(),  # after every turn, not every 5
        extractor=ModelExtractor(
            model=BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0"),  # cheaper than the agent's
            system_prompt="Extract durable user preferences as discrete facts.",
        ),
    ),
    config={"knowledge_base_id": "KB123", "data_source_type": "CUSTOM", "data_source_id": "DS456"},
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { InvocationTrigger, ModelExtractor, BedrockModel } from '@strands-agents/sdk'
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'preferences',
  writable: true,
  extraction: {
    trigger: new InvocationTrigger(), // after every turn, not every 5
    extractor: new ModelExtractor({
      model: new BedrockModel(), // a cheaper model than the agent's to cut cost
      systemPrompt: 'Extract durable user preferences as discrete facts.',
    }),
  },
  config: { knowledgeBaseId: 'KB123', dataSourceType: 'CUSTOM', dataSourceId: 'DS456' },
})
```
(( /tab "TypeScript" ))

The `ModelExtractor` distills messages into discrete facts with a model call. It uses the agent’s own model by default. Pass a cheaper model to cut cost, or a system prompt to steer what information you want to save as memories. Some backends extract server-side instead: a store that implements the `add_messages``addMessages` sink receives the raw message batch with no model call, so for those you omit the extractor.

Two triggers ship with the SDK. `InvocationTrigger` runs after every turn, `IntervalTrigger` runs every N turns. For a custom trigger, extend `ExtractionTrigger`. A trigger registers a hook on the agent and calls `fire()` when extraction should run. Tying it to agent state can let a tool decide the moment, rather than extracting on a turn cadence:

(( tab "Python" ))
```python
from strands.memory.extraction.types import ExtractionConfig, ExtractionTrigger, ExtractionTriggerContext
from strands.hooks import AfterInvocationEvent
from strands.vended_memory_stores import BedrockKnowledgeBaseStore


class CustomTrigger(ExtractionTrigger):
    name = "custom-trigger"

    def attach(self, context: ExtractionTriggerContext) -> None:
        # Extract only after a tool has flagged extraction.
        def maybe_fire(event: AfterInvocationEvent) -> None:
            if context.agent.state.get("extract"):
                context.fire()

        context.agent.add_hook(maybe_fire, AfterInvocationEvent)


store = BedrockKnowledgeBaseStore(
    name="preferences",
    writable=True,
    extraction=ExtractionConfig(trigger=CustomTrigger()),
    config={"knowledge_base_id": "KB123", "data_source_type": "CUSTOM", "data_source_id": "DS456"},
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { ExtractionTrigger, AfterInvocationEvent } from '@strands-agents/sdk'
import type { ExtractionTriggerContext } from '@strands-agents/sdk'

// Extract only after a tool has flagged extraction
class CustomTrigger extends ExtractionTrigger {
  readonly name = 'custom-trigger'

  attach(context: ExtractionTriggerContext): void {
    context.agent.addHook(AfterInvocationEvent, () => {
      if (context.agent.appState.get('extract')) {
        context.fire()
      }
    })
  }
}
```
(( /tab "TypeScript" ))

`fire()` runs the save in the background and returns immediately, so a trigger never blocks the agent loop. A trigger that never fires never extracts; for a guaranteed final write regardless of triggers, use the manager’s flush method.

Extraction is at-least-once: a failed batch is retried, so the same entry may be written more than once. A store used with extraction should tolerate duplicate writes (the manager tracks a high-water mark per store, so a successful batch is never re-extracted).

### Flushing pending writes

Extraction writes run in the background and are not awaited by the agent loop, so the most recent turn may not be saved yet when the agent responds. The manager’s flush method closes that gap. It forces every store to save its buffered messages, even a store whose trigger has not fired this turn or one currently backed off, and awaits all of those writes (including any that start while it waits). Awaiting it as part of a graceful shutdown guarantees nothing in the buffer is lost.

When to call it differs by SDK:

(( tab "Python" ))
Whether you need to flush manually depends on how you call the agent:

-   **`agent(...)`** (the synchronous call) runs each invocation in its own event loop. Closing that loop would cancel in-flight background saves, so this path awaits a flush after every invocation. Writes already persist by the time the call returns, and you never flush manually.
-   **`agent.invoke_async(...)`** and **`agent.stream_async(...)`** share your own long-lived event loop and do not flush. Extraction stays on its trigger cadence, so flush yourself at a shutdown boundary before the loop closes:

```python
# After driving the agent with invoke_async / stream_async, before the loop closes.
await memory_manager.flush()
```
(( /tab "Python" ))

(( tab "TypeScript" ))
The agent loop never flushes for you, on any call path. Await `flush()` as part of a graceful shutdown and every outstanding write lands before the process exits; skip it and the last turns’ background writes are dropped:

```typescript
// At a shutdown boundary you control, before the process exits.
process.on('beforeExit', async () => {
  await memoryManager.flush()
})
```
(( /tab "TypeScript" ))

This protects a graceful shutdown. A process killed without one (crash, `SIGKILL`, hard timeout) can still lose the last unsaved turn, since the flush never runs; a more frequent trigger narrows that window. Do not call `flush()` after every turn alongside a periodic trigger, since that forces a save each time and defeats the trigger’s schedule.

## Programmatic Access

You can search and write directly on the memory manager, outside the agent loop. Both methods target all relevant stores by default, or a subset by name:

(( tab "Python" ))
```python
from strands.memory.types import MemoryAddOptions, MemorySearchOptions

# Search every store, or a subset by name.
all_results = await memory_manager.search("travel plans")
scoped = await memory_manager.search(
    "travel plans",
    MemorySearchOptions(stores=["personal"], max_search_results=5),
)

# Write to writable stores, with metadata.
await memory_manager.add(
    "Prefers aisle seats",
    MemoryAddOptions(stores=["personal"], metadata={"category": "travel"}),
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
// Search every store, or a subset by name.
const all = await memoryManager.search('travel plans')
const scoped = await memoryManager.search('travel plans', {
  stores: ['personal'],
  maxSearchResults: 5,
})

// Write to writable stores, with metadata.
await memoryManager.add('Prefers aisle seats', {
  stores: ['personal'],
  metadata: { category: 'travel' },
})
```
(( /tab "TypeScript" ))

Partial failures are handled per method. `search` logs and skips any store that fails, returning whatever the rest produced. `add` validates the target stores first, then raises an aggregate error if any write fails, so a failed write is never silent.

## Custom Stores

Use the memory manager with any backend by implementing the `MemoryStore` interface. Only `search` is required; add an `add` method to make the store writable, and a tool method to expose backend-native tools alongside the manager’s:

(( tab "Python" ))
A store implements the config attributes (`name`, `writable`, and so on) plus an async `search`, and optionally `add`, `add_messages`, and `get_tools`:

```python
from strands.memory.types import MemoryEntry, SearchOptions


class InMemoryStore:
    name = "preferences"
    description = "User preferences and stable facts."
    max_search_results = None
    writable = True
    extraction = None

    def __init__(self) -> None:
        self._entries: list[str] = []

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        limit = (options and options.max_search_results) or 3
        matches = [content for content in self._entries if query in content]
        return [MemoryEntry(content=content) for content in matches[:limit]]

    async def add(self, content: str, metadata: dict | None = None) -> None:
        self._entries.append(content)


store = InMemoryStore()
```
(( /tab "Python" ))

(( tab "TypeScript" ))
A store implements `search`, and optionally `add`, `addMessages`, and `getTools`:

```typescript
import type { MemoryStore, MemoryEntry, SearchOptions } from '@strands-agents/sdk'

class InMemoryStore implements MemoryStore {
  readonly name = 'preferences'
  readonly writable = true
  private readonly _entries: string[] = []

  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    const limit = options?.maxSearchResults ?? 3
    return this._entries
      .filter((content) => content.includes(query))
      .slice(0, limit)
      .map((content) => ({ content }))
  }

  async add(content: string): Promise<void> {
    this._entries.push(content)
  }
}
```
(( /tab "TypeScript" ))

A store exposes two write paths, and which ones it implements decides how it can be written to:

-   `add` takes a single piece of content. It backs the `add_memory` tool, the programmatic `add` method, and extraction that distills facts client-side with a `ModelExtractor`.
-   `add_messages``addMessages` takes a batch of raw conversation messages. It backs **server-side extraction**: the manager hands the filtered message batch straight to this method with no client-side model call, so the backend does the distillation itself. The batch preserves the conversation’s role structure.

A store can implement either path or both. The following store extracts server-side, delegating to `my_backend`, a stand-in for your managed backend’s client:

(( tab "Python" ))
```python
from strands.memory.types import AddMessagesContext, MemoryEntry, SearchOptions
from strands.types.content import Message


class ServerSideStore:
    name = "preferences"
    description = "User preferences and stable facts."
    max_search_results = None
    writable = True
    extraction = True  # extract every 5 turns; no extractor, so add_messages is used

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        return await my_backend.retrieve(query, options and options.max_search_results)

    # The manager hands the raw message batch here; the backend extracts server-side.
    async def add_messages(
        self, messages: list[Message], context: AddMessagesContext | None = None
    ) -> None:
        await my_backend.ingest_conversation(messages)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import type {
  MemoryStore,
  MemoryEntry,
  SearchOptions,
  MessageData,
  AddMessagesContext,
} from '@strands-agents/sdk'

class ServerSideStore implements MemoryStore {
  readonly name = 'preferences'
  readonly writable = true
  // Extract every 5 turns; no extractor, so the manager calls addMessages.
  readonly extraction = true

  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    return myBackend.retrieve(query, options?.maxSearchResults)
  }

  // The manager hands the raw message batch here; the backend extracts server-side.
  async addMessages(
    messages: MessageData[],
    context?: AddMessagesContext
  ): Promise<void> {
    await myBackend.ingestConversation(messages)
  }
}
```
(( /tab "TypeScript" ))

For a reference implementation backed by a managed service, see the [Bedrock Knowledge Base store](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md), which extracts client-side.

## Best Practices

-   **Choose how the agent reads.** The `search_memory` tool lets the model pull knowledge when it judges it needs it; injection guarantees relevant context on every user turn. They compose: enable both for a guaranteed baseline plus on-demand depth.
-   **Match extraction cadence to cost.** An every-turn trigger with a model extractor means a model call per turn. That is a token cost, not a latency cost, since extraction runs in the background; an interval trigger lowers it, and turns skipped between runs are still captured when the trigger next fires.
-   **Scope stores for multi-tenancy.** Give each tenant its own scoped store rather than mixing knowledge in one (see [Bedrock Knowledge Base](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md)).
-   **Give the agent context.** Add a meaningful description to each store, so the agent knows what type of knowledge each store contains.
-   **Tolerate duplicate writes** in custom stores used with extraction, since failed batches are retried.

## How Memory Relates to Other Strands Constructs

Three SDK features manage different kinds of state; memory is the one that crosses sessions:

-   [Session management](/docs/user-guide/concepts/agents/session-management/index.md) persists the full conversation so an agent can resume where it left off.
-   [Conversation management](/docs/user-guide/concepts/agents/conversation-management/index.md) keeps the conversation within the model’s context window during a session.
-   **Memory** carries durable knowledge *across* sessions, without replaying past conversations.

## Related

-   [Bedrock Knowledge Base store](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md) - the vended `MemoryStore` backed by Amazon Bedrock Knowledge Bases.
-   [Context Injector](/docs/user-guide/concepts/plugins/context-injector/index.md) - the generic injection plugin that memory injection builds on.
-   [Session management](/docs/user-guide/concepts/agents/session-management/index.md) - persist the conversation itself across restarts.
-   [Conversation management](/docs/user-guide/concepts/agents/conversation-management/index.md) - keep a session within the model’s context window.

## Related pages

- [Bedrock Knowledge Base Store](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md) (1 shared tag)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/memory/memory_manager.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/memory_manager.py)
- [harness-sdk/strands-py/src/strands/memory/types.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py)

### TypeScript

- [harness-sdk/strands-ts/src/memory/memory-manager.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/memory/memory-manager.ts)
- [harness-sdk/strands-ts/src/memory/types.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/memory/types.ts)
