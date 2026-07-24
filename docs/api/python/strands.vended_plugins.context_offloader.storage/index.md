Storage backends for offloaded tool result content.

This module defines the Storage protocol and provides three built-in implementations: file-based, in-memory, and S3 storage. Each content block from a tool result is stored individually with its content type preserved.

**Example**:

```python
from strands.vended_plugins.context_offloader import (
    FileStorage,
    InMemoryStorage,
    S3Storage,
)

# File-based storage
storage = FileStorage(artifact_dir="./artifacts")
ref = storage.store("tool_123_0", b"large output content...", "text/plain")
content, content_type = storage.retrieve(ref)

# In-memory storage (useful for testing and serverless)
storage = InMemoryStorage()

# S3 storage
storage = S3Storage(bucket="my-bucket", prefix="artifacts/")
```

## Storage

```python
@runtime_checkable
class Storage(Protocol)
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:64](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L64)

Backend for storing and retrieving offloaded content blocks.

Each content block from a tool result is stored individually with its content type preserved. The SDK ships three built-in implementations: `InMemoryStorage`, `FileStorage`, and `S3Storage`. Implement this protocol to create custom storage backends (e.g., Redis, DynamoDB).

Lifecycle: This protocol intentionally does not include eviction or deletion methods. Stored content accumulates for the lifetime of the storage instance. For long-running agents, create a new storage instance per session or use a backend with built-in lifecycle management (e.g., S3 lifecycle policies).

#### store

```python
async def store(key: str,
                content: bytes,
                content_type: str = "text/plain") -> str
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:79](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L79)

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:93](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L93)

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:108](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L108)

Store offloaded content as files, on the host filesystem or through a sandbox.

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:127](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L127)

Initialize file-based storage.

**Arguments**:

-   `artifact_dir` - Directory path where artifact files will be stored.
-   `sandbox` - Optional sandbox to route file I/O through.

#### for\_sandbox

```python
def for_sandbox(sandbox: "Sandbox") -> "FileStorage"
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:147](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L147)

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:175](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L175)

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:217](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L217)

Retrieve content from a stored file.

Accepts both full paths (as returned by `store()`) and bare filenames for backward compatibility.

**Arguments**:

-   `reference` - The file path or filename returned by store().

**Returns**:

A tuple of (content bytes, content type).

**Raises**:

-   `KeyError` - If the file does not exist.

## InMemoryStorage

```python
class InMemoryStorage()
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:287](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L287)

Store offloaded content in memory.

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:316](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L316)

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:336](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L336)

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:353](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L353)

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:411](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L411)

Remove all stored content.

Call this to free memory when offloaded results are no longer needed, e.g., between sessions or after an invocation completes.

## S3Storage

```python
class S3Storage()
```

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:421](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L421)

Store offloaded content in Amazon S3.

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:446](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L446)

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:482](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L482)

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

Defined in: [src/strands/vended\_plugins/context\_offloader/storage.py:513](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_plugins/context_offloader/storage.py#L513)

Retrieve content from an S3 object.

Accepts both `s3://` URIs (as returned by `store()`) and raw S3 keys for backward compatibility. References are constrained to the configured `bucket` and `prefix`: a reference that resolves to a key outside the prefix (or to a different bucket) is rejected, mirroring the scope that `store()` enforces.

**Arguments**:

-   `reference` - The S3 URI or object key returned by store().

**Returns**:

A tuple of (content bytes, content type).

**Raises**:

-   `KeyError` - If the object does not exist or the reference resolves outside the configured bucket and prefix.