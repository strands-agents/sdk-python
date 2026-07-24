`TestMemoryStore` is a [`MemoryStore`](/docs/user-guide/concepts/memory/overview/index.md#stores) backed by a JSON file. It requires no setup and no provisioned resources, so you can give an agent memory in one line and try recall, injection, and extraction for prototyping or writing tests.

It persists to disk by default, so an agent remembers across restarts with no setup. Point it at a store name and go:

(( tab "Python" ))
```python
from strands import Agent
from strands.memory import MemoryManager
from strands.vended_memory_stores.test_memory_store import TestMemoryStore

# Persists to ~/.strands/memory/notes.json by default. Survives restarts.
store = TestMemoryStore(name="notes")

agent = Agent(memory_manager=MemoryManager(stores=[store]))
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent, BedrockModel } from '@strands-agents/sdk'
import { TestMemoryStore } from '@strands-agents/sdk/vended-memory-stores/test-memory-store'

// Persists to ~/.strands/memory/notes.json by default. Survives restarts.
const store = new TestMemoryStore({ name: 'notes' })

const agent = new Agent({
  model: new BedrockModel(),
  memoryManager: { stores: [store] },
})
```
(( /tab "TypeScript" ))

The store is writable by default, unlike a read-only managed store. So the [`add_memory` tool](/docs/user-guide/concepts/memory/overview/index.md#memory-tools) and [automatic extraction](/docs/user-guide/concepts/memory/overview/index.md#automatic-extraction) work as soon as you enable them, with no data source to set up first.

It suits prototyping and tests, not a production corpus: each write rewrites the whole file and recall is keyword matching rather than semantic. For a production backend, use the [Bedrock Knowledge Base store](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md).

## Persistence and Location

The store writes to `~/.strands/memory/<store-name>.json`, deriving the filename from `name`. Give it an explicit `path` to control where the file lands, or turn persistence off for a store that lives only in memory:

(( tab "Python" ))
```python
from strands.vended_memory_stores.test_memory_store import TestMemoryStore

# Ephemeral: nothing is written to disk, and a fresh instance forgets everything.
scratch = TestMemoryStore(name="notes", persist=False)

# Explicit file location instead of the default under ~/.strands/memory/.
project = TestMemoryStore(name="notes", path="./notes.json")
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { TestMemoryStore } from '@strands-agents/sdk/vended-memory-stores/test-memory-store'

// Ephemeral: nothing is written to disk, and a fresh instance forgets everything.
const scratch = new TestMemoryStore({ name: 'notes', persist: false })

// Explicit file location instead of the default under ~/.strands/memory/.
const project = new TestMemoryStore({ name: 'notes', path: './notes.json' })
```
(( /tab "TypeScript" ))

Reach for `persist=False``persist: false` in tests and throwaway runs where you want recall and ranking without leaving a file behind. The default (persist to disk) is what demonstrates the feature: memory that survives a restart.

Persistence runs on the SDK’s `Storage` interface: `persist=False``persist: false` uses an in-memory backend, and persisting to disk uses a local-file backend. The config above is all you configure; the store selects the backend for you.

The on-disk format is shared between the Python and TypeScript SDKs. Records use the same keys and timestamp shape, so a file written by one SDK reads in the other.

## Configuration

`TestMemoryStore` takes the [shared `MemoryStore` fields](/docs/user-guide/concepts/memory/overview/index.md#stores) (`name`, `description`, `max_search_results``maxSearchResults`, `writable`, `extraction`) plus two of its own:

| Field | Purpose |
| --- | --- |
| `persist` | Whether to write entries to disk. On by default: flushes to `path`. When off, entries stay in memory only and are lost when the process exits. |
| `path` | Full path to the backing JSON file. Defaults to `~/.strands/memory/<store-name>.json`. Ignored when persistence is off. |

`writable` is on by default here. Construction rejects an empty `name`, an empty `path`, or a `max_search_results``maxSearchResults` below `1`.

## Recall

`search` ranks entries by lexical overlap: it counts how many distinct words from the query appear in each entry’s content, and returns the highest scorers first, breaking ties toward the most recent entry. Each result carries the count under a reserved `_relevanceScore` metadata key. A query with no usable words returns nothing.

Note

Recall is keyword matching, not semantic search. A query word matches only the same word, not a synonym, so “seat” does not find an entry about “chair.” For embedding-based semantic search over a managed vector store, use the [Bedrock Knowledge Base store](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md).

## Writing Memories

`add` stores a single piece of content and returns its id. Identical content is deduplicated: a repeat write returns the existing record’s id instead of storing a second copy, so the at-least-once retries that extraction may perform never accumulate duplicates.

(( tab "Python" ))
```python
from strands.vended_memory_stores.test_memory_store import TestMemoryStore

store = TestMemoryStore(name="notes")

# add returns the id of the stored (or already-present, on dedup) record.
result = await store.add("User prefers aisle seats", {"category": "travel"})
print(result.id)

results = await store.search("which seats does the user prefer?")
for entry in results:
    print(entry.content, entry.metadata.get("_relevanceScore"))
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { TestMemoryStore } from '@strands-agents/sdk/vended-memory-stores/test-memory-store'

const store = new TestMemoryStore({ name: 'notes' })

// add returns the id of the stored (or already-present, on dedup) record.
const { id } = await store.add('User prefers aisle seats', { category: 'travel' })

const results = await store.search('which seats does the user prefer?')
for (const entry of results) {
  console.log(entry.content, entry.metadata?._relevanceScore)
}
```
(( /tab "TypeScript" ))

Writing to a store constructed with `writable=False``writable: false`, or adding empty content, raises. When the backing directory is unreachable or not writable, the write raises, naming the backing file.

## Extraction

Enable [automatic extraction](/docs/user-guide/concepts/memory/overview/index.md#automatic-extraction) to capture memories from the conversation without the agent calling a tool:

(( tab "Python" ))
```python
from strands.vended_memory_stores.test_memory_store import TestMemoryStore

store = TestMemoryStore(
    name="notes",
    extraction=True,  # distill facts from the conversation, every 5 turns
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { TestMemoryStore } from '@strands-agents/sdk/vended-memory-stores/test-memory-store'

const store = new TestMemoryStore({
  name: 'notes',
  extraction: true, // distill facts from the conversation, every 5 turns
})
```
(( /tab "TypeScript" ))

The store implements `add``add` but not `add_messages``addMessages`, so extraction runs client-side: a `ModelExtractor` distills facts from the conversation and writes each through `add`. To change the cadence or swap the extractor, see [Automatic Extraction](/docs/user-guide/concepts/memory/overview/index.md#automatic-extraction) on the Memory page.

## Scale and Limitations

Each `add` reads the current file and rewrites it in full, replacing it atomically so a crash mid-write never leaves a partial file. That design suits prototyping and personal memory, hundreds to low thousands of entries, not a production corpus. For large or high-throughput workloads, use a managed store like the [Bedrock Knowledge Base store](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md).

Concurrent `add` calls on one store instance are serialized, so they can’t clobber one another. Separate instances or processes writing the same file are not coordinated, so avoid pointing two at the same file: the last write wins. A corrupt or wrong-shaped backing file raises rather than returning bad data; a missing file starts empty.

## Related

-   [Memory](/docs/user-guide/concepts/memory/overview/index.md) - the `MemoryManager` concept this store plugs into, including the tools, extraction, and injection it enables.
-   [Bedrock Knowledge Base store](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md) - the managed `MemoryStore` with semantic search, for production workloads.

## Related pages

- [Memory](/docs/user-guide/concepts/memory/overview/index.md) (1 shared tag)
- [Bedrock Knowledge Base Store](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md) (1 shared tag)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/vended_memory_stores/test_memory_store/store.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/test_memory_store/store.py)

### TypeScript

- [harness-sdk/strands-ts/src/vended-memory-stores/test-memory-store/store.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-memory-stores/test-memory-store/store.ts)
