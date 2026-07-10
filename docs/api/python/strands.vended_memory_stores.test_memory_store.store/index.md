A :class:`~strands.memory.types.MemoryStore` that persists to a local JSON file.

A zero-infrastructure store for prototyping and testing. It persists to disk by default so memories persist across sessions, and can be set to ephemeral for testing.

## TestMemoryStore

```python
class TestMemoryStore(MemoryStore)
```

Defined in: [src/strands/vended\_memory\_stores/test\_memory\_store/store.py:71](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/test_memory_store/store.py#L71)

A :class:`~strands.memory.types.MemoryStore` backed by an in-memory list and a local JSON file.

A zero-infrastructure store for prototyping and testing. It persists to disk by default so memories persist across sessions. Set `persist=False` for an ephemeral, single-session store.

Recall is lexical: results are ranked by how many query tokens overlap an entry’s content, with the most recent entry winning ties. This is keyword matching, not the semantic search a managed vector store (e.g. :class:`~strands.vended_memory_stores.bedrock_knowledge_base.BedrockKnowledgeBaseStore`) provides.

Each :meth:`add` rewrites the whole file, so this fits modest volumes (hundreds to low thousands of entries), not production workloads — use a managed store like `BedrockKnowledgeBaseStore` for that. Writes within a process are serialized; concurrent writers across processes are not.

The on-disk format is shared with the TypeScript SDK’s `TestMemoryStore`: records use the same camelCase keys (`id`, `content`, `metadata`, `createdAt`) and the same timestamp shape, so a backing file written by either SDK can be read by the other.

**Example**:

```python
from strands.vended_memory_stores.test_memory_store import TestMemoryStore

# Persists to ~/.strands/memory/notes.json by default.
store = TestMemoryStore(name="notes")

result = await store.add("User prefers dark mode")
results = await store.search("what theme does the user like?")
```

#### \_\_init\_\_

```python
def __init__(**store_config: Unpack[TestMemoryStoreConfig]) -> None
```

Defined in: [src/strands/vended\_memory\_stores/test\_memory\_store/store.py:105](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/test_memory_store/store.py#L105)

Initialize the store.

**Arguments**:

-   `**store_config` - See :class:`TestMemoryStoreConfig`.

**Raises**:

-   `ValueError` - If `name` or `path` is empty/whitespace, or `max_search_results` is less than 1.

#### search

```python
async def search(query: str,
                 options: SearchOptions | None = None) -> list[MemoryEntry]
```

Defined in: [src/strands/vended\_memory\_stores/test\_memory\_store/store.py:142](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/test_memory_store/store.py#L142)

Search stored entries for those whose content overlaps the query.

Results are ranked by query-token overlap, with the most recent entry winning ties.

**Arguments**:

-   `query` - The search query text.
-   `options` - Optional search configuration.

**Returns**:

Matching memory entries ordered by relevance. Each entry’s `metadata` includes a reserved synthetic `_relevanceScore` key (the token-overlap count). An empty or token-less query returns no results.

**Raises**:

-   `ValueError` - If `options.max_search_results` is less than 1.

#### add

```python
async def add(content: str,
              metadata: Metadata | None = None) -> TestMemoryAddResult
```

Defined in: [src/strands/vended\_memory\_stores/test\_memory\_store/store.py:184](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/test_memory_store/store.py#L184)

Add `content` (with optional `metadata`) to the store.

Identical content is deduplicated: a repeat write returns the existing record’s id without storing a second copy, so the at-least-once retries that extraction may perform never accumulate duplicates.

**Arguments**:

-   `content` - The text content to store.
-   `metadata` - Optional metadata to attach to the entry. The key `_relevanceScore` is
-   `reserved` - :meth:`search` populates it on results, so a value stored under it here is overwritten in search output.

**Returns**:

The id of the stored (or already-present) record.

**Raises**:

-   `ValueError` - If the store is not writable or `content` is empty/whitespace.
-   `OSError` - If persisting the entry to disk fails (e.g. the path is unreachable or not writable), with the target path in the message.