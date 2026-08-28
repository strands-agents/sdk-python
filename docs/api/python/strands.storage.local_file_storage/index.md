Local filesystem storage implementation.

## LocalFileStorage

```python
class LocalFileStorage()
```

Defined in: [src/strands/storage/local\_file\_storage.py:22](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L22)

Persists each key as a file under a base directory.

Key segments separated by ’/’ map to directory segments. Writes on the host filesystem are atomic (write to temp file, then rename).

**Example**:

```python
from strands.storage import LocalFileStorage

storage = LocalFileStorage("./.strands/")
await storage.write("session/abc/state.json", data)
```

#### \_\_init\_\_

```python
def __init__(base_dir: str = "./.strands/",
             *,
             sandbox: Sandbox | None = None) -> None
```

Defined in: [src/strands/storage/local\_file\_storage.py:37](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L37)

Initialize local file storage.

**Arguments**:

-   `base_dir` - Root directory under which all keys are stored.
-   `sandbox` - Optional sandbox to route I/O through.

#### base\_dir

```python
@property
def base_dir() -> str
```

Defined in: [src/strands/storage/local\_file\_storage.py:48](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L48)

The root directory under which all keys are stored.

#### for\_sandbox

```python
def for_sandbox(sandbox: Sandbox) -> LocalFileStorage
```

Defined in: [src/strands/storage/local\_file\_storage.py:52](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L52)

Return a copy bound to the given sandbox.

If already bound to the same sandbox, returns self.

**Arguments**:

-   `sandbox` - Sandbox to bind to.

**Returns**:

A LocalFileStorage instance bound to the sandbox.

#### write

```python
async def write(key: str, data: bytes) -> None
```

Defined in: [src/strands/storage/local\_file\_storage.py:70](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L70)

Store data as a file, creating parent directories as needed.

On the host filesystem, writes are atomic via write-to-temp-then-rename.

**Arguments**:

-   `key` - Opaque string key identifying the value.
-   `data` - Raw bytes to persist.

**Raises**:

-   `StorageError` - If the write fails.

#### read

```python
async def read(key: str) -> bytes | None
```

Defined in: [src/strands/storage/local\_file\_storage.py:109](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L109)

Read the file corresponding to key.

**Arguments**:

-   `key` - The key to read.

**Returns**:

The file contents as bytes, or None if the file does not exist.

**Raises**:

-   `StorageError` - If the read fails for a reason other than a missing file.

#### delete

```python
async def delete(key: str) -> None
```

Defined in: [src/strands/storage/local\_file\_storage.py:137](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L137)

Delete the file corresponding to key. No-op if it does not exist.

**Arguments**:

-   `key` - The key to delete.

**Raises**:

-   `StorageError` - If the delete fails.

#### list

```python
async def list(query: str = "") -> builtins.list[str]
```

Defined in: [src/strands/storage/local\_file\_storage.py:166](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L166)

List keys matching the given prefix by walking the directory tree.

**Arguments**:

-   `query` - A prefix string to filter keys. Empty string matches all.

**Returns**:

Matching keys sorted ascending.

**Raises**:

-   `StorageError` - If the listing fails.

#### search

```python
async def search(query: str) -> builtins.list[StorageSearchResult]
```

Defined in: [src/strands/storage/local\_file\_storage.py:191](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L191)

Search stored content by keyword token-overlap scoring.

**Arguments**:

-   `query` - Natural-language search query.

**Returns**:

All matches with relevance scores, ranked best-first.

#### namespace

```python
def namespace(prefix: str) -> LocalFileStorage
```

Defined in: [src/strands/storage/local\_file\_storage.py:202](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L202)

Return a new LocalFileStorage scoped to a subdirectory.

Unlike a generic `_NamespacedStorage` wrapper, this returns a real `LocalFileStorage` whose `base_dir` incorporates the prefix. This preserves access to `base_dir` for strategies that need the filesystem path (e.g. index-based search), and `for_sandbox` continues to work.

**Arguments**:

-   `prefix` - Prefix to prepend to all keys.

**Returns**:

A new LocalFileStorage rooted at the sub-path.