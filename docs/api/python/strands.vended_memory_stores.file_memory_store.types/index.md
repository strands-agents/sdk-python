Configuration types for the FileMemoryStore.

## FileMemoryStoreConfig

```python
class FileMemoryStoreConfig(MemoryStoreConfig)
```

Defined in: [src/strands/vended\_memory\_stores/file\_memory\_store/types.py:13](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/file_memory_store/types.py#L13)

Configuration for :class:`~strands.vended_memory_stores.file_memory_store.FileMemoryStore`.

**Attributes**:

-   `storage` - The unified Storage backend for file operations. Defaults to LocalFileStorage at `./.strands/`. Keys are auto-scoped under `memory/<name>/` unless the provided storage is already namespaced, so stores with distinct names safely share one backend. Two stores with the same name on the same backend share storage — give them different names (or separate storage) to isolate them.