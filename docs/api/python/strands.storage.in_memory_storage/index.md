In-memory storage implementation.

## InMemoryStorage

```python
class InMemoryStorage()
```

Defined in: [src/strands/storage/in\_memory\_storage.py:11](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/in_memory_storage.py#L11)

Map-backed storage for testing and short-lived processes.

Content does not survive process restarts. The store is unbounded — consumers manage eviction themselves.

**Example**:

```python
from strands.storage import InMemoryStorage

storage = InMemoryStorage()
await storage.write("sessions/abc/state.json", b'\{"messages": []}')
data = await storage.read("sessions/abc/state.json")
```

#### \_\_init\_\_

```python
def __init__() -> None
```

Defined in: [src/strands/storage/in\_memory\_storage.py:27](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/in_memory_storage.py#L27)

Initialize an empty in-memory store.

#### write

```python
async def write(key: str, data: bytes) -> None
```

Defined in: [src/strands/storage/in\_memory\_storage.py:32](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/in_memory_storage.py#L32)

Store data under key, overwriting any existing value.

**Arguments**:

-   `key` - Opaque string key identifying the value.
-   `data` - Raw bytes to persist.

**Raises**:

-   `StorageError` - If the key is invalid.

#### read

```python
async def read(key: str) -> bytes | None
```

Defined in: [src/strands/storage/in\_memory\_storage.py:46](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/in_memory_storage.py#L46)

Retrieve the bytes previously stored under key.

**Arguments**:

-   `key` - The key to read.

**Returns**:

The stored bytes, or None if no value exists for key.

**Raises**:

-   `StorageError` - If the key is invalid.

#### delete

```python
async def delete(key: str) -> None
```

Defined in: [src/strands/storage/in\_memory\_storage.py:63](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/in_memory_storage.py#L63)

Delete the value stored under key. A no-op if the key does not exist.

**Arguments**:

-   `key` - The key to delete.

**Raises**:

-   `StorageError` - If the key is invalid.

#### list

```python
async def list(query: str = "") -> builtins.list[str]
```

Defined in: [src/strands/storage/in\_memory\_storage.py:76](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/in_memory_storage.py#L76)

List keys matching the given prefix.

**Arguments**:

-   `query` - A prefix string to filter keys. Empty string matches all.

**Returns**:

Matching keys sorted ascending.

**Raises**:

-   `StorageError` - If the prefix is invalid.

#### namespace

```python
def namespace(prefix: str) -> _NamespacedStorage
```

Defined in: [src/strands/storage/in\_memory\_storage.py:93](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/in_memory_storage.py#L93)

Return a view of this storage with all keys prefixed.

**Arguments**:

-   `prefix` - Prefix to prepend to all keys.

**Returns**:

A namespaced storage view.

#### clear

```python
def clear() -> None
```

Defined in: [src/strands/storage/in\_memory\_storage.py:104](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/storage/in_memory_storage.py#L104)

Remove all stored entries.