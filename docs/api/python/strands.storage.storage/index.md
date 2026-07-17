Unified storage interface and key-normalization helpers.

## Storage

```python
@runtime_checkable
class Storage(Protocol[ListQuery])
```

Defined in: [src/strands/storage/storage.py:69](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L69)

A backend for storing and retrieving raw bytes under string keys.

The interface is deliberately minimal — four operations over opaque bytes values. Implementations must treat keys as opaque path-like strings (segments separated by ’/’) and must round-trip the bytes they are given unchanged.

The `ListQuery` type parameter controls what `list` accepts. It defaults to `str` (a key prefix), which every backend supports. Implementations may widen it to accept a richer query object while still accepting a plain string for SDK-internal callers.

Implement this to add a custom backend; the SDK ships :class:`InMemoryStorage`, :class:`LocalFileStorage`, and :class:`S3Storage`.

#### write

```python
async def write(key: str, data: bytes) -> None
```

Defined in: [src/strands/storage/storage.py:85](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L85)

Store data under key, overwriting any existing value.

**Arguments**:

-   `key` - Opaque, ’/‘-separated key identifying the value.
-   `data` - Raw bytes to persist.

**Raises**:

-   `StorageError` - If the write fails.

#### read

```python
async def read(key: str) -> bytes | None
```

Defined in: [src/strands/storage/storage.py:97](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L97)

Retrieve the bytes previously stored under key.

**Arguments**:

-   `key` - The key to read.

**Returns**:

The stored bytes, or None if no value exists for key.

**Raises**:

-   `StorageError` - If the read fails for a reason other than a missing key.

#### delete

```python
async def delete(key: str) -> None
```

Defined in: [src/strands/storage/storage.py:111](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L111)

Delete the value stored under key. A no-op if the key does not exist.

**Arguments**:

-   `key` - The key to delete.

**Raises**:

-   `StorageError` - If the delete fails.

#### list

```python
async def list(query: ListQuery) -> builtins.list[str]
```

Defined in: [src/strands/storage/storage.py:122](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L122)

List keys matching the given prefix query.

Returns full keys (not the suffix after the prefix), sorted lexicographically. An empty string lists every key.

**Arguments**:

-   `query` - A string prefix to match.

**Returns**:

The matching keys, sorted ascending.

**Raises**:

-   `StorageError` - If the listing fails.

## \_NamespacedStorage

```python
class _NamespacedStorage()
```

Defined in: [src/strands/storage/storage.py:140](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L140)

A storage view that prepends a prefix to all keys.

Composable — calling `.namespace()` on the result nests prefixes.

#### write

```python
async def write(key: str, data: bytes) -> None
```

Defined in: [src/strands/storage/storage.py:153](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L153)

Store data under the prefixed key.

#### read

```python
async def read(key: str) -> bytes | None
```

Defined in: [src/strands/storage/storage.py:157](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L157)

Read from the prefixed key.

#### delete

```python
async def delete(key: str) -> None
```

Defined in: [src/strands/storage/storage.py:161](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L161)

Delete the prefixed key.

#### list

```python
async def list(query: str = "") -> builtins.list[str]
```

Defined in: [src/strands/storage/storage.py:165](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L165)

List keys under the prefix, stripping it from results.

#### namespace

```python
def namespace(prefix: str) -> _NamespacedStorage
```

Defined in: [src/strands/storage/storage.py:170](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L170)

Return a further-scoped view by nesting prefixes.

#### for\_sandbox

```python
def for_sandbox(sandbox: object) -> _NamespacedStorage
```

Defined in: [src/strands/storage/storage.py:174](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L174)

Delegate sandbox binding to the underlying storage and re-wrap.