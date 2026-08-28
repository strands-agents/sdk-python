Pluggable search strategy protocol for storage backends.

## SearchStrategy

```python
class SearchStrategy(Protocol)
```

Defined in: [src/strands/storage/search/types.py:10](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/search/types.py#L10)

A pluggable search strategy for storage backends.

Strategies encapsulate a single approach to searching stored content — keyword/lexical scan, vector similarity, full-text index, etc. Storage backends delegate their `search()` to a strategy, and consumers (memory stores, context offloaders) can override the default.

#### search

```python
async def search(storage: Storage, query: str,
                 **kwargs: Any) -> list[StorageSearchResult]
```

Defined in: [src/strands/storage/search/types.py:19](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/search/types.py#L19)

Search content in storage matching query.

**Arguments**:

-   `storage` - The storage to search over.
-   `query` - A natural-language string query.
-   `**kwargs` - Strategy-specific options for forward compatibility.

**Returns**:

Matched keys with relevance scores, ranked best-first.