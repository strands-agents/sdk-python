File-based memory store backed by the unified Storage interface.

Stores knowledge as markdown files under a `memory/` storage namespace. Provides keyword-based search via `search_memory` (registered by MemoryManager).

## FileMemoryStore

```python
class FileMemoryStore(MemoryStore)
```

Defined in: [src/strands/vended\_memory\_stores/file\_memory\_store/store.py:82](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/file_memory_store/store.py#L82)

A file-based memory store backed by the unified Storage interface.

Knowledge is stored as plain markdown files under a `memory/` storage namespace. Retrieval uses keyword-based token-overlap scoring against filename and body content.

The storage backend defaults to :class:`~strands.storage.LocalFileStorage`. Keys are auto-scoped under `memory/<name>/` (so a store named `agent-memory` with the default backend writes to `./.strands/memory/agent-memory/`).

**Example**:

```python
from strands import Agent
from strands.memory import MemoryManager
from strands.vended_memory_stores.file_memory_store import FileMemoryStore

memory_store = FileMemoryStore(name="agent-memory")

agent = Agent(
    model=model,
    memory_manager=MemoryManager(stores=[memory_store], injection=False),
)
```

#### \_\_init\_\_

```python
def __init__(**config: Unpack[FileMemoryStoreConfig]) -> None
```

Defined in: [src/strands/vended\_memory\_stores/file\_memory\_store/store.py:107](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/file_memory_store/store.py#L107)

Initialize the FileMemoryStore.

**Arguments**:

-   `**config` - Store configuration. See :class:`FileMemoryStoreConfig`.

#### search

```python
async def search(query: str,
                 options: SearchOptions | None = None) -> list[MemoryEntry]
```

Defined in: [src/strands/vended\_memory\_stores/file\_memory\_store/store.py:138](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/file_memory_store/store.py#L138)

Search knowledge files by keyword token-overlap scoring.

**Arguments**:

-   `query` - Natural-language search query.
-   `options` - Optional search configuration (e.g. max\_search\_results).

**Returns**:

Top matches ranked by relevance.

#### add

```python
async def add(content: str, metadata: Metadata | None = None) -> str
```

Defined in: [src/strands/vended\_memory\_stores/file\_memory\_store/store.py:172](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/file_memory_store/store.py#L172)

Add a knowledge entry to the store.

The filename is derived from the first line of content (slugified, truncated to 50 chars). If a file with the same slug already exists, new facts (lines after the heading) are appended rather than overwriting.

**Arguments**:

-   `content` - The knowledge content to store.
-   `metadata` - Unused; accepted for interface compatibility.

**Returns**:

The canonical storage key the entry was written under.