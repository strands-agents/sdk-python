Unified storage interface and key-normalization helpers.

## Storage

```python
@runtime_checkable
class Storage(Protocol[ListQuery])
```

Defined in: [src/strands/storage/storage.py:74](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L74)

A backend for storing and retrieving raw bytes under string keys.

The interface is deliberately minimal — four operations over opaque bytes values. Keys are opaque strings — implementations must round-trip the bytes they are given unchanged. The shipped backends interpret ’/’ as a logical separator (collapsing runs, rejecting ’..’), but custom backends may apply their own key scheme.

The `ListQuery` type parameter controls what `list` accepts. It defaults to `str` (a key prefix), which every backend supports. Implementations may widen it to accept a richer query object while still accepting a plain string for SDK-internal callers.

Implement this to add a custom backend; the SDK ships :class:`InMemoryStorage`, :class:`LocalFileStorage`, and :class:`S3Storage`.

#### write

```python
async def write(key: str, data: bytes) -> None
```

Defined in: [src/strands/storage/storage.py:92](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L92)

Store data under key, overwriting any existing value.

**Arguments**:

-   `key` - Opaque string key identifying the value.
-   `data` - Raw bytes to persist.

**Raises**:

-   `StorageError` - If the write fails.

#### read

```python
async def read(key: str) -> bytes | None
```

Defined in: [src/strands/storage/storage.py:104](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L104)

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

Defined in: [src/strands/storage/storage.py:118](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L118)

Delete the value stored under key. A no-op if the key does not exist.

**Arguments**:

-   `key` - The key to delete.

**Raises**:

-   `StorageError` - If the delete fails.

#### list

```python
async def list(query: ListQuery) -> builtins.list[str]
```

Defined in: [src/strands/storage/storage.py:129](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L129)

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

Defined in: [src/strands/storage/storage.py:147](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L147)

A storage view that prepends a prefix to all keys.

Composable — calling `.namespace()` on the result nests prefixes. Uses :func:`_normalize_prefix` to sanitize the prefix, so it assumes a ’/‘-separated key scheme. Backends with a different key scheme should implement their own namespacing.

#### write

```python
async def write(key: str, data: bytes) -> None
```

Defined in: [src/strands/storage/storage.py:163](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L163)

Store data under the prefixed key.

#### read

```python
async def read(key: str) -> bytes | None
```

Defined in: [src/strands/storage/storage.py:167](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L167)

Read from the prefixed key.

#### delete

```python
async def delete(key: str) -> None
```

Defined in: [src/strands/storage/storage.py:171](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L171)

Delete the prefixed key.

#### list

```python
async def list(query: str = "") -> builtins.list[str]
```

Defined in: [src/strands/storage/storage.py:175](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L175)

List keys under the prefix, stripping it from results.

#### namespace

```python
def namespace(prefix: str) -> _NamespacedStorage
```

Defined in: [src/strands/storage/storage.py:180](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L180)

Return a further-scoped view by nesting prefixes.

#### for\_sandbox

```python
def for_sandbox(sandbox: object) -> _NamespacedStorage
```

Defined in: [src/strands/storage/storage.py:184](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/storage.py#L184)

Delegate sandbox binding to the underlying storage and re-wrap.