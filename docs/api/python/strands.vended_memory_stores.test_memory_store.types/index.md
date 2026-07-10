Configuration and result types for the JSON-file memory store.

## TestMemoryStoreConfig

```python
class TestMemoryStoreConfig(MemoryStoreConfig)
```

Defined in: [src/strands/vended\_memory\_stores/test\_memory\_store/types.py:10](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/test_memory_store/types.py#L10)

Full configuration for a :class:`TestMemoryStore`, passed as its constructor kwargs.

The store persists to disk by default so memories persist across sessions. Set `persist` to `False` for an ephemeral, single-session store.

**Attributes**:

-   `persist` - Whether to persist entries to disk so they survive process restarts. `True` (default) flushes writes to `path` (or the default location); `False` keeps entries in memory only, so they are lost when the process exits.
-   `path` - Full path to the JSON file backing this store. Defaults to `~/.strands/memory/<sanitized-store-name>.json`. Ignored when `persist` is `False`.

## TestMemoryAddResult

```python
@dataclass
class TestMemoryAddResult()
```

Defined in: [src/strands/vended\_memory\_stores/test\_memory\_store/types.py:34](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/test_memory_store/types.py#L34)

Result returned by :meth:`TestMemoryStore.add`.

**Attributes**:

-   `id` - The generated id of the stored (or already-present, on dedup) record.