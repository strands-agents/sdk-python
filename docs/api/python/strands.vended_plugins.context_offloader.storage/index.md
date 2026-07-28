Storage backends for offloaded tool result content.

.. deprecated:: The storage classes in this module (`InMemoryStorage`, `FileStorage`, `S3Storage`) are deprecated. Use the unified storage backends from :mod:`strands.storage` instead::

from strands.storage import InMemoryStorage, LocalFileStorage, S3Storage

**Example**:

```python
from strands.storage import InMemoryStorage, LocalFileStorage, S3Storage

# Unified storage backends
storage = LocalFileStorage("./artifacts")
storage = InMemoryStorage()
storage = S3Storage("my-bucket", prefix="artifacts/")
```

## Storage

```python
@runtime_checkable
class Storage(Protocol)
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:57](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L57)

Backend for storing and retrieving offloaded content blocks.

.. deprecated:: Use :class:`strands.storage.Storage` instead.

Each content block from a tool result is stored individually with its content type preserved. The SDK ships three built-in implementations: `InMemoryStorage`, `FileStorage`, and `S3Storage`. Implement this protocol to create custom storage backends (e.g., Redis, DynamoDB).

Lifecycle: This protocol intentionally does not include eviction or deletion methods. Stored content accumulates for the lifetime of the storage instance. For long-running agents, create a new storage instance per session or use a backend with built-in lifecycle management (e.g., S3 lifecycle policies).

#### store

```python
async def store(key: str,
                content: bytes,
                content_type: str = "text/plain") -> str
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:75](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L75)

Store content and return a reference identifier.

**Arguments**:

-   `key` - A unique key for this content block.
-   `content` - The raw content bytes to store.
-   `content_type` - MIME type of the content (e.g., “text/plain”, “application/json”, “image/png”, “application/pdf”).

**Returns**:

A reference string that can be used to retrieve the content later.

#### retrieve

```python
async def retrieve(reference: str) -> tuple[bytes, str]
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:89](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L89)

Retrieve stored content by reference.

**Arguments**:

-   `reference` - The reference returned by a previous store() call.

**Returns**:

A tuple of (content bytes, content type).

**Raises**:

-   `KeyError` - If the reference is not found.

## FileStorage

```python
class FileStorage()
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:104](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L104)

Store offloaded content as files, on the host filesystem or through a sandbox.

.. deprecated:: Use :class:`strands.storage.LocalFileStorage` instead.

Files are written to the configured artifact directory with unique names. File extensions are derived from the content type. A `.metadata.json` sidecar file tracks content types so they survive process restarts.

When constructed without a `sandbox`, writes go to the host filesystem. When used by :class:`ContextOffloader`, the plugin binds a per-agent copy to that agent’s sandbox (which may be the host default) via :meth:`for_sandbox`.

**Arguments**:

-   `artifact_dir` - Directory path where artifact files will be stored.
-   `sandbox` - Optional sandbox to route file I/O through. When `None`, the host filesystem is used directly.

#### \_\_init\_\_

```python
def __init__(artifact_dir: str = "./artifacts",
             *,
             sandbox: "Sandbox | None" = None) -> None
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:126](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L126)

Initialize file-based storage.

**Arguments**:

-   `artifact_dir` - Directory path where artifact files will be stored.
-   `sandbox` - Optional sandbox to route file I/O through.

#### for\_sandbox

```python
def for_sandbox(sandbox: "Sandbox") -> "FileStorage"
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:146](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L146)

Return a storage instance bound to the given sandbox.

Instances constructed with an explicit sandbox keep using it (returns `self`). Otherwise a new instance is returned so a shared :class:`ContextOffloader` can isolate artifacts per agent sandbox.

**Arguments**:

-   `sandbox` - Sandbox to bind the returned instance to.

**Returns**:

A FileStorage routed through `sandbox`.

#### store

```python
async def store(key: str,
                content: bytes,
                content_type: str = "text/plain") -> str
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:209](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L209)

Store content as a file and return the path as reference.

The returned path preserves the form of `artifact_dir` passed to the constructor: a relative `artifact_dir` yields a relative reference, an absolute one yields an absolute reference.

**Arguments**:

-   `key` - A unique key for this content block.
-   `content` - The raw content bytes to store.
-   `content_type` - MIME type of the content.

**Returns**:

The file path (e.g., `./artifacts/1234_1_key.txt`).

#### retrieve

```python
async def retrieve(reference: str) -> tuple[bytes, str]
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:276](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L276)

Retrieve content from a stored file.

Accepts full paths (as returned by `store()`), bare filenames, and filename stems (without extension) for backward compatibility.

**Arguments**:

-   `reference` - The file path, filename, or stem returned by store().

**Returns**:

A tuple of (content bytes, content type).

**Raises**:

-   `KeyError` - If the file does not exist.

## InMemoryStorage

```python
class InMemoryStorage()
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:345](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L345)

Store offloaded content in memory.

.. deprecated:: Use :class:`strands.storage.InMemoryStorage` instead.

Useful for testing and serverless environments where disk access is not available or not desired. Thread-safe.

Supports turn-based eviction: entries not accessed (stored or retrieved) within `evict_after_turns` agent loop cycles are automatically removed. The `ContextOffloader` plugin triggers eviction on each model invocation cycle. Eviction is enabled by default (20 cycles). Pass `None` to disable.

**Notes**:

Content does not survive process restarts. For multi-session persistence, use `FileStorage` or `S3Storage`. Each agent should use its own `InMemoryStorage` instance — sharing one across multiple agents is not supported when eviction is enabled.

Evicted entries are permanently deleted from memory. The agent will receive an error if it attempts to retrieve evicted content. The original tool result is not preserved in the conversation history after offloading — only the preview and references remain in context.

**Arguments**:

-   `evict_after_turns` - Number of cycles of inactivity before an entry is evicted. Defaults to 20. `None` disables eviction.

#### \_\_init\_\_

```python
def __init__(
        evict_after_turns: int | None = _DEFAULT_EVICT_AFTER_TURNS) -> None
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:377](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L377)

Initialize in-memory storage.

**Arguments**:

-   `evict_after_turns` - Number of cycles of inactivity before an entry is evicted. Defaults to 20. `None` disables eviction.

**Raises**:

-   `ValueError` - If evict\_after\_turns is not a positive integer.

#### store

```python
async def store(key: str,
                content: bytes,
                content_type: str = "text/plain") -> str
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:397](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L397)

Store content in memory and return a reference.

**Arguments**:

-   `key` - A unique key for this content block.
-   `content` - The raw content bytes to store.
-   `content_type` - MIME type of the content.

**Returns**:

A unique reference string.

#### retrieve

```python
async def retrieve(reference: str) -> tuple[bytes, str]
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:414](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L414)

Retrieve content from memory.

Refreshes the last-accessed turn so the entry stays alive longer when eviction is enabled.

**Arguments**:

-   `reference` - The reference returned by store().

**Returns**:

A tuple of (content bytes, content type).

**Raises**:

-   `KeyError` - If the reference is not found (or was evicted).

#### clear

```python
def clear() -> None
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:472](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L472)

Remove all stored content.

Call this to free memory when offloaded results are no longer needed, e.g., between sessions or after an invocation completes.

## S3Storage

```python
class S3Storage()
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:482](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L482)

Store offloaded content in Amazon S3.

.. deprecated:: Use :class:`strands.storage.S3Storage` instead.

Objects are stored with unique keys under the configured prefix. Content type is preserved as S3 object metadata.

**Arguments**:

-   `bucket` - S3 bucket name.
-   `prefix` - S3 key prefix for organizing stored artifacts.
-   `boto_session` - Optional boto3 session. If not provided, a new session is created using the given region\_name.
-   `boto_client_config` - Optional botocore client configuration.
-   `region_name` - AWS region. Used only when boto\_session is not provided.

**Example**:

```python
from strands.vended_plugins.context_offloader import S3Storage

storage = S3Storage(
    bucket="my-agent-artifacts",
    prefix="tool-results/",
)
```

#### \_\_init\_\_

```python
def __init__(bucket: str,
             prefix: str = "",
             boto_session: boto3.Session | None = None,
             boto_client_config: BotocoreConfig | None = None,
             region_name: str | None = None) -> None
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:510](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L510)

Initialize S3-based storage.

**Arguments**:

-   `bucket` - S3 bucket name.
-   `prefix` - S3 key prefix for organizing stored artifacts.
-   `boto_session` - Optional boto3 session. If not provided, a new session is created using the given region\_name.
-   `boto_client_config` - Optional botocore client configuration.
-   `region_name` - AWS region. Used only when boto\_session is not provided.

#### store

```python
async def store(key: str,
                content: bytes,
                content_type: str = "text/plain") -> str
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:546](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L546)

Store content as an S3 object and return an `s3://` URI as reference.

**Arguments**:

-   `key` - A unique key for this content block.
-   `content` - The raw content bytes to store.
-   `content_type` - MIME type of the content.

**Returns**:

An S3 URI (e.g., `s3://bucket/prefix/1234_1_key`).

**Raises**:

-   `botocore.exceptions.ClientError` - If the S3 operation fails (e.g., bucket does not exist, permission denied).

#### retrieve

```python
async def retrieve(reference: str) -> tuple[bytes, str]
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:577](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L577)

Retrieve content from an S3 object.

Accepts both `s3://` URIs (as returned by `store()`) and raw S3 keys for backward compatibility. References are constrained to the configured `bucket` and `prefix`: a reference that resolves to a key outside the prefix (or to a different bucket) is rejected, mirroring the scope that `store()` enforces.

**Arguments**:

-   `reference` - The S3 URI or object key returned by store().

**Returns**:

A tuple of (content bytes, content type).

**Raises**:

-   `KeyError` - If the object does not exist or the reference resolves outside the configured bucket and prefix.