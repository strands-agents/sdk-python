Amazon S3 storage implementation.

## S3Storage

```python
class S3Storage()
```

Defined in: [src/strands/storage/s3\_storage.py:18](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/s3_storage.py#L18)

Persists bytes as objects in an Amazon S3 bucket.

The AWS SDK (boto3) is imported lazily on first use so applications that never use S3 don’t pay the import cost.

**Example**:

```python
from strands.storage import S3Storage

storage = S3Storage("my-bucket", prefix="agents/")
await storage.write("session/abc/state.json", data)
```

#### \_\_init\_\_

```python
def __init__(bucket: str,
             *,
             prefix: str = "",
             region_name: str | None = None,
             boto_session: Any = None,
             boto_client_config: Any = None) -> None
```

Defined in: [src/strands/storage/s3\_storage.py:33](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/s3_storage.py#L33)

Initialize S3 storage.

**Arguments**:

-   `bucket` - S3 bucket name.
-   `prefix` - Key prefix prepended to every key (namespace within the bucket).
-   `region_name` - AWS region override.
-   `boto_session` - Pre-configured boto3 session. Cannot combine with region\_name.
-   `boto_client_config` - Botocore Config object for the S3 client.

**Raises**:

-   `StorageError` - If both region\_name and boto\_session are provided.

#### write

```python
async def write(key: str, data: bytes) -> None
```

Defined in: [src/strands/storage/s3\_storage.py:65](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/s3_storage.py#L65)

Store data as an S3 object.

**Arguments**:

-   `key` - Opaque string key identifying the value.
-   `data` - Raw bytes to persist.

**Raises**:

-   `StorageError` - If the write fails.

#### read

```python
async def read(key: str) -> bytes | None
```

Defined in: [src/strands/storage/s3\_storage.py:84](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/s3_storage.py#L84)

Read an S3 object.

**Arguments**:

-   `key` - The key to read.

**Returns**:

The object contents as bytes, or None if the key does not exist.

**Raises**:

-   `StorageError` - If the read fails for a reason other than a missing key.

#### delete

```python
async def delete(key: str) -> None
```

Defined in: [src/strands/storage/s3\_storage.py:111](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/s3_storage.py#L111)

Delete an S3 object. No-op if the key does not exist.

**Arguments**:

-   `key` - The key to delete.

**Raises**:

-   `StorageError` - If the delete fails.

#### list

```python
async def list(query: str = "") -> builtins.list[str]
```

Defined in: [src/strands/storage/s3\_storage.py:129](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/s3_storage.py#L129)

List S3 objects matching the given prefix.

Paginates automatically for large result sets.

**Arguments**:

-   `query` - A prefix string to filter keys. Empty string matches all.

**Returns**:

Matching keys sorted ascending, with the storage-level prefix stripped.

**Raises**:

-   `StorageError` - If the listing fails.

#### search

```python
async def search(query: str) -> builtins.list[StorageSearchResult]
```

Defined in: [src/strands/storage/s3\_storage.py:180](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/s3_storage.py#L180)

Search stored content by keyword token-overlap scoring.

**Arguments**:

-   `query` - Natural-language search query.

**Returns**:

All matches with relevance scores, ranked best-first.

#### namespace

```python
def namespace(prefix: str) -> _NamespacedStorage
```

Defined in: [src/strands/storage/s3\_storage.py:193](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/s3_storage.py#L193)

Return a view of this storage with all keys prefixed.

**Arguments**:

-   `prefix` - Prefix to prepend to all keys.

**Returns**:

A namespaced storage view.