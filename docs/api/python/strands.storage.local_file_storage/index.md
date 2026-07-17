Local filesystem storage implementation.

## LocalFileStorage

```python
class LocalFileStorage()
```

Defined in: [src/strands/storage/local\_file\_storage.py:20](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L20)

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

Defined in: [src/strands/storage/local\_file\_storage.py:35](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L35)

Initialize local file storage.

**Arguments**:

-   `base_dir` - Root directory under which all keys are stored.
-   `sandbox` - Optional sandbox to route I/O through.

#### for\_sandbox

```python
def for_sandbox(sandbox: Sandbox) -> LocalFileStorage
```

Defined in: [src/strands/storage/local\_file\_storage.py:45](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L45)

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

Defined in: [src/strands/storage/local\_file\_storage.py:60](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L60)

Store data as a file, creating parent directories as needed.

On the host filesystem, writes are atomic via write-to-temp-then-rename.

**Arguments**:

-   `key` - Opaque, ’/‘-separated key identifying the value.
-   `data` - Raw bytes to persist.

**Raises**:

-   `StorageError` - If the write fails.

#### read

```python
async def read(key: str) -> bytes | None
```

Defined in: [src/strands/storage/local\_file\_storage.py:99](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L99)

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

Defined in: [src/strands/storage/local\_file\_storage.py:127](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L127)

Delete the file corresponding to key. No-op if it does not exist.

**Arguments**:

-   `key` - The key to delete.

**Raises**:

-   `StorageError` - If the delete fails.

#### list

```python
async def list(query: str = "") -> builtins.list[str]
```

Defined in: [src/strands/storage/local\_file\_storage.py:156](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L156)

List keys matching the given prefix by walking the directory tree.

**Arguments**:

-   `query` - A prefix string to filter keys. Empty string matches all.

**Returns**:

Matching keys sorted ascending.

**Raises**:

-   `StorageError` - If the listing fails.

#### namespace

```python
def namespace(prefix: str) -> _NamespacedStorage
```

Defined in: [src/strands/storage/local\_file\_storage.py:181](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/local_file_storage.py#L181)

Return a view of this storage with all keys prefixed.

The returned view preserves `for_sandbox` via delegation to the underlying storage, so sandbox routing works even when storage is pre-namespaced before being passed to a plugin.

**Arguments**:

-   `prefix` - Prefix to prepend to all keys.

**Returns**:

A namespaced storage view.