Keyword search strategy using token-overlap scoring.

#### tokenize

```python
def tokenize(text: str) -> set[str]
```

Defined in: [src/strands/storage/search/keyword.py:13](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/search/keyword.py#L13)

Lowercase and split text into a set of word tokens, dropping empties.

Splits on any run of non-word characters (Unicode-aware). Ensures cross-SDK compatibility with the TypeScript `/[^\p\{L}\p\{N}_]+/u` regex.

**Arguments**:

-   `text` - The text to tokenize.

**Returns**:

A set of lowercased word tokens.

#### token\_overlap\_score

```python
def token_overlap_score(query_tokens: set[str], content: str) -> int
```

Defined in: [src/strands/storage/search/keyword.py:28](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/search/keyword.py#L28)

Lexical relevance score: distinct query tokens present in the content.

A higher count means more of the query’s words are present. Returns 0 when there is no overlap.

**Arguments**:

-   `query_tokens` - Pre-tokenized query terms.
-   `content` - The content string to score against.

**Returns**:

Number of distinct query tokens found in the content.

## KeywordSearchStrategy

```python
class KeywordSearchStrategy()
```

Defined in: [src/strands/storage/search/keyword.py:44](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/search/keyword.py#L44)

Keyword search strategy using token-overlap scoring.

Tokenizes the query and each stored entry (key + content), then scores by the number of distinct query tokens that appear. Works on any storage backend with `list()` and `read()` — no index or embedding model required.

This is the default search strategy for all shipped storage backends.

**Example**:

```python
from strands.storage.search import KeywordSearchStrategy

strategy = KeywordSearchStrategy()
results = await strategy.search(storage, "dark mode toggle")
```

#### search

```python
async def search(storage: Storage, query: str,
                 **kwargs: Any) -> builtins.list[StorageSearchResult]
```

Defined in: [src/strands/storage/search/keyword.py:62](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/search/keyword.py#L62)

Search content in storage by keyword token-overlap scoring.

**Arguments**:

-   `storage` - The storage to search over.
-   `query` - A natural-language string query.
-   `**kwargs` - Unused; accepted for protocol compatibility.

**Returns**:

Matched keys with relevance scores, ranked best-first.