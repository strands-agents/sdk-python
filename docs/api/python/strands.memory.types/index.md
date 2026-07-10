Core types for the Strands memory module.

## MemoryEntry

```python
@dataclass
class MemoryEntry()
```

Defined in: [src/strands/memory/types.py:25](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L25)

A single memory entry retrieved from or stored to a memory store.

**Attributes**:

-   `store_name` - Name of the store this entry came from, set by `MemoryManager.search`. Stores need not set this themselves.

## SearchOptions

```python
class SearchOptions(TypedDict)
```

Defined in: [src/strands/memory/types.py:38](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L38)

Options passed to :meth:`MemoryStore.search`.

Store implementations may extend this with backend-specific fields; note that `MemoryManager.search` forwards only these base fields across its stores.

## AddMessagesContext

```python
@dataclass
class AddMessagesContext()
```

Defined in: [src/strands/memory/types.py:49](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L49)

Context the manager supplies to :meth:`MemoryStore.add_messages`.

An extension point: fields are added here without changing the :meth:`MemoryStore.add_messages` signature.

**Attributes**:

-   `sequence_numbers` - Per-message identities aligned one-to-one with `messages` (`sequence_numbers[i]` identifies `messages[i]`). A retried batch reuses the same numbers, so a store can build an idempotency key that survives retries — unlike a content hash, which collides when two messages share text (e.g. “ok”). Numbers increase with order but may have gaps (a message filtered to empty is dropped while its siblings keep their own numbers), and reset to 0 each agent run, so a durable dedup token must combine one with a run-unique id. `None` when the manager has no per-message numbers to supply.

## MemorySearchOptions

```python
class MemorySearchOptions(SearchOptions)
```

Defined in: [src/strands/memory/types.py:70](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L70)

Options for `MemoryManager.search`.

**Attributes**:

-   `stores` - Filter to specific stores by name. Omit to search all. A programmatic search with an empty list searches no stores, whereas the `search_memory` tool treats an empty list as “search all in-scope stores”.

## MemoryAddOptions

```python
class MemoryAddOptions(TypedDict)
```

Defined in: [src/strands/memory/types.py:83](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L83)

Options for `MemoryManager.add`.

**Attributes**:

-   `stores` - Filter to specific writable stores by name. Omit to write to all. A programmatic add with an empty list matches no store (raises), whereas the `add_memory` tool treats an empty list as “write to all in-scope stores”.

## MemoryToolConfig

```python
class MemoryToolConfig(TypedDict)
```

Defined in: [src/strands/memory/types.py:97](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L97)

Configuration for customizing a memory tool’s name or description.

## MemoryAddToolConfig

```python
class MemoryAddToolConfig(MemoryToolConfig)
```

Defined in: [src/strands/memory/types.py:104](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L104)

Configuration for the `add_memory` tool.

**Attributes**:

-   `stores` - The writable stores the tool may write to, as store names or :class:`MemoryStore` instances. Omit to allow all writable stores.
-   `wait_for_writes` - When `True` (default), wait for writes and return
-   \`\`\`{“stored”\` - …}`(or surface a failure to the model). When`False`, fire-and-forget: return` {“accepted”: …}“ once writes are dispatched; per-store failures are logged.

## InjectionQueryContext

```python
@dataclass
class InjectionQueryContext()
```

Defined in: [src/strands/memory/types.py:121](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L121)

Context passed to :attr:`MemoryInjectionConfig.query`.

**Attributes**:

-   `messages` - The current conversation, as data.

## InjectionFormatContext

```python
@dataclass
class InjectionFormatContext()
```

Defined in: [src/strands/memory/types.py:132](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L132)

Context passed to :attr:`MemoryInjectionConfig.format`.

**Attributes**:

-   `entries` - The retrieved memory entries to render.

## InjectionQueryCallback

```python
class InjectionQueryCallback(Protocol)
```

Defined in: [src/strands/memory/types.py:142](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L142)

Derives the injection search query from the current conversation.

Implemented by a plain function as well — the `**kwargs` tail lets the calling convention grow new keyword arguments without breaking existing callbacks.

#### \_\_call\_\_

```python
def __call__(context: InjectionQueryContext, **kwargs: Any) -> str | None
```

Defined in: [src/strands/memory/types.py:149](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L149)

Return the search query, or `None`/`""` to skip injection this call.

## InjectionFormatCallback

```python
class InjectionFormatCallback(Protocol)
```

Defined in: [src/strands/memory/types.py:154](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L154)

Renders retrieved memory entries into the injected text.

Implemented by a plain function as well — the `**kwargs` tail lets the calling convention grow new keyword arguments without breaking existing callbacks.

#### \_\_call\_\_

```python
def __call__(context: InjectionFormatContext, **kwargs: Any) -> str
```

Defined in: [src/strands/memory/types.py:161](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L161)

Return the text to inject for the given entries.

## MemoryInjectionConfig

```python
class MemoryInjectionConfig(InjectionConfig)
```

Defined in: [src/strands/memory/types.py:173](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L173)

Configuration for memory context injection.

Extends the generic :class:`~strands.injection.InjectionConfig` (which carries `trigger`) with the memory-owned knobs: how many entries to retrieve, how to derive the query, and how to render the results.

**Attributes**:

-   `max_entries` - Maximum number of entries to retrieve and inject per model call. A store ranks by semantic similarity, which is not the same as contextual usefulness, so the default injects a small candidate set rather than betting on the top hit. Raising it improves recall at the cost of a larger prepend (context bloat); lower it for a tighter injection. With multiple stores, results are concatenated in store-registration order with no cross-store ranking, so this cap can favor entries from earlier-registered stores. Defaults to 5.
-   `query` - Derives the search query from the current conversation. Return `None` or an empty string to skip injection for this call. A callback that raises fails open (injection is skipped). Defaults to an adaptive query: the latest user message’s text on a user turn, otherwise the most recent assistant message’s text (the previous step on an autonomous turn).
-   `format` - Renders retrieved entries into the injected text. A callback that raises fails open (injection is skipped). Defaults to a `<memory>` XML block with one `<entry>` per result, carrying a `source` attribute naming the originating store (when known) so the model can attribute and weigh each memory. The default escapes entry content and source, so a custom `format` that emits markup is responsible for its own escaping.

## MemoryManagerConfig

```python
class MemoryManagerConfig(TypedDict)
```

Defined in: [src/strands/memory/types.py:206](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L206)

Configuration for the `MemoryManager`, mirroring the constructor kwargs.

**Attributes**:

-   `stores` - One or more memory stores to manage.
-   `search_tool_config` - Search tool configuration. Defaults to `True`.
-   `add_tool_config` - Add tool configuration. Defaults to `False` (opt-in); `True` allows all writable stores, or pass a :class:`MemoryAddToolConfig` to restrict it.
-   `injection` - Memory context injection. Defaults to `True`. `True` uses the default injection settings; pass a :class:`MemoryInjectionConfig` to customize retrieval, timing, and formatting; `False` disables it.

## MemoryStoreConfig

```python
class MemoryStoreConfig(TypedDict)
```

Defined in: [src/strands/memory/types.py:226](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L226)

Declarative identity and behavior fields a store is configured with.

**Attributes**:

-   `name` - Unique identifier for this store, used to target it in tools.
-   `description` - Human-readable description; included in tool descriptions.
-   `max_search_results` - Default maximum results per search, used when a caller does not pass a per-call value.
-   `writable` - Whether this store accepts writes. A writable store requires at least one write sink (:meth:`MemoryStore.add` or :meth:`MemoryStore.add_messages`).
-   `extraction` - Automatic-extraction configuration for this writable store, as a `bool | config` shorthand. `True` enables it with defaults; an :class:`ExtractionConfig` defaults any unset field; `False`/omitted is off. The defaults run every 5 turns, and the extraction method depends on the store’s write methods: a store implementing only `add` uses a :class:`~strands.memory.extraction.model_extractor.ModelExtractor` for client-side extraction (a model call to distill facts, stored via `add`), while a store implementing `add_messages` uses server-side extraction (the backend extracts the raw messages, no model call).

## MemoryStore

```python
class MemoryStore(Protocol)
```

Defined in: [src/strands/memory/types.py:254](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L254)

Runtime contract for a memory store backend.

A store exposes the :class:`MemoryStoreConfig` fields as attributes and implements :meth:`search`, plus optionally :meth:`add`, :meth:`add_messages`, and :meth:`get_tools`. The fields are re-declared here because a `Protocol` cannot extend a `TypedDict`.

**Attributes**:

-   `name` - Unique identifier for this store, used to target it in tools.
-   `description` - Human-readable description; included in tool descriptions.
-   `max_search_results` - Default maximum results per search.
-   `writable` - Whether this store accepts writes.
-   `extraction` - Resolved automatic-extraction configuration, or `None`/`False` when off.

#### search

```python
async def search(query: str,
                 options: SearchOptions | None = None) -> list[MemoryEntry]
```

Defined in: [src/strands/memory/types.py:275](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L275)

Search the store for entries matching the query, ordered by relevance.

#### add

```python
async def add(content: str, metadata: Metadata | None = None) -> Any
```

Defined in: [src/strands/memory/types.py:281](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L281)

Add a single piece of content to the store.

Extraction writes are at-least-once, so implementations used with extraction should tolerate duplicate writes. The resolved value is store-specific and not consumed by the manager.

#### add\_messages

```python
async def add_messages(messages: list[Message],
                       context: AddMessagesContext | None = None) -> Any
```

Defined in: [src/strands/memory/types.py:290](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L290)

Ingest a batch of conversation messages, preserving role structure.

The sink for extraction without a client-side extractor: the manager hands the filtered batch straight here. The resolved value is store-specific.

#### initialize

```python
async def initialize() -> None
```

Defined in: [src/strands/memory/types.py:299](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L299)

Perform async setup that must succeed before the agent runs.

Called by the `MemoryManager` during `init_agent`. Stores that require remote resources (e.g. resolving a knowledge base type) implement this; the default is a no-op.

#### get\_tools

```python
def get_tools() -> list[AgentTool]
```

Defined in: [src/strands/memory/types.py:307](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/memory/types.py#L307)

Return store-specific tools to register alongside the manager’s tools.