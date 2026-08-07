A :class:`~strands.memory.types.MemoryStore` backed by Amazon Bedrock Knowledge Bases.

Supports semantic search via `Retrieve` and document ingestion via `IngestKnowledgeBaseDocuments` for `CUSTOM` and `S3` data sources. Works with both managed and self-managed (vector) knowledge bases; the type is detected automatically via `GetKnowledgeBase` during :meth:`~BedrockKnowledgeBaseStore.initialize` and the query is shaped to match.

## BedrockKnowledgeBaseStore

```python
class BedrockKnowledgeBaseStore(MemoryStore)
```

Defined in: [src/strands/vended\_memory\_stores/bedrock\_knowledge\_base/store.py:69](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/bedrock_knowledge_base/store.py#L69)

A :class:`~strands.memory.types.MemoryStore` backed by Amazon Bedrock Knowledge Bases.

Supports semantic search via `Retrieve` and document ingestion via `IngestKnowledgeBaseDocuments` for `CUSTOM` and `S3` data sources. Works with all knowledge base types (MANAGED, VECTOR, KENDRA, SQL); the type is detected via `GetKnowledgeBase` during :meth:`initialize` and determines whether `Retrieve` uses `managedSearchConfiguration` or `vectorSearchConfiguration`. Detection requires the `bedrock:GetKnowledgeBase` permission; a failure raises at agent construction (via `MemoryManager`) or on first `search()` standalone. To skip detection, provide `knowledge_base_type` in the config.

**Example**:

```python
from strands.vended_memory_stores.bedrock_knowledge_base import BedrockKnowledgeBaseStore

store = BedrockKnowledgeBaseStore(
    config=\{
        "knowledge_base_id": "KB123",
        "data_source_type": "CUSTOM",
        "data_source_id": "DS456",
    },
    name="preferences",
    scope="user-abc",
    writable=True,
)

results = await store.search("what are my preferences?")
result = await store.add("User prefers dark mode")
```

#### \_\_init\_\_

```python
def __init__(**store_config: Unpack[BedrockKnowledgeBaseStoreConfig]) -> None
```

Defined in: [src/strands/vended\_memory\_stores/bedrock\_knowledge\_base/store.py:100](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/bedrock_knowledge_base/store.py#L100)

Initialize the store.

**Arguments**:

-   `**store_config` - See :class:`BedrockKnowledgeBaseStoreConfig`.

**Raises**:

-   `ValueError` - If `max_search_results` is less than 1, or (when `writable`) if the write configuration is invalid.

#### initialize

```python
async def initialize() -> None
```

Defined in: [src/strands/vended\_memory\_stores/bedrock\_knowledge\_base/store.py:165](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/bedrock_knowledge_base/store.py#L165)

Resolve the knowledge base type via `GetKnowledgeBase` and cache the result.

Idempotent: no-op when the type is already known (either from config or a prior call). When the store is registered with a `MemoryManager`, this runs at agent construction so permission or connectivity issues surface early. Standalone callers get the same check on first `search()`.

#### search

```python
async def search(query: str,
                 options: SearchOptions | None = None) -> list[MemoryEntry]
```

Defined in: [src/strands/vended\_memory\_stores/bedrock\_knowledge\_base/store.py:188](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/bedrock_knowledge_base/store.py#L188)

Search the knowledge base for entries matching the query.

**Arguments**:

-   `query` - The search query text.
-   `options` - Optional search configuration.

**Returns**:

Matching memory entries ordered by relevance. Each entry’s `metadata` includes user-provided attributes plus two reserved synthetic keys: `_relevance_score` (number) and `_source_location` (Bedrock retrieval location object).

**Raises**:

-   `ValueError` - If `options.max_search_results` is less than 1.

#### add

```python
async def add(
        content: str,
        metadata: Metadata | None = None) -> BedrockKnowledgeBaseAddResult
```

Defined in: [src/strands/vended\_memory\_stores/bedrock\_knowledge\_base/store.py:251](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/bedrock_knowledge_base/store.py#L251)

Ingest `content` (with optional `metadata`) into the knowledge base.

Only `CUSTOM` and `S3` data sources support this. Requires `data_source_id` (and, for `S3`, an `s3` config).

**Arguments**:

-   `content` - The text content to ingest.
-   `metadata` - Optional metadata attributes to attach to the document.

**Returns**:

The document identifier (UUID for `CUSTOM`, `s3://` URI for `S3`).

**Raises**:

-   `ValueError` - If the store is not writable or `content` is empty/whitespace.