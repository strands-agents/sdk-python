Abstract base class for Agent model providers.

## BaseModelConfig

```python
class BaseModelConfig(TypedDict)
```

Defined in: [src/strands/models/model.py:122](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L122)

Base configuration shared by all model providers.

**Attributes**:

-   `context_window_limit` - Maximum context window size in tokens for the model. This value represents the total token capacity shared between input and output.

## CacheConfig

```python
@dataclass
class CacheConfig()
```

Defined in: [src/strands/models/model.py:134](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L134)

Configuration for prompt caching.

**Attributes**:

-   `strategy` - Caching strategy to use.
    -   “auto”: Automatically detect model support and inject cachePoint to maximize cache coverage
    -   “anthropic”: Inject cachePoint in Anthropic-compatible format without model support check
-   `ttl` - Optional TTL duration for cache entries (e.g. “5m”, “1h”). When specified, auto-injected cache points will include this TTL value.

## CacheToolsConfig

```python
@dataclass
class CacheToolsConfig()
```

Defined in: [src/strands/models/model.py:150](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L150)

Configuration for the toolConfig cache point.

**Attributes**:

-   `type` - Cache point type (e.g. “default”).
-   `ttl` - Optional TTL duration for the cache entry (e.g. “5m”, “1h”).

## Model

```python
class Model(abc.ABC)
```

Defined in: [src/strands/models/model.py:162](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L162)

Abstract base class for Agent model providers.

This class defines the interface for all model implementations in the Strands Agents SDK. It provides a standardized way to configure and process requests for different AI model providers.

#### stateful

```python
@property
def stateful() -> bool
```

Defined in: [src/strands/models/model.py:170](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L170)

Whether the model manages conversation state server-side.

**Returns**:

False by default. Model providers that support server-side state should override this.

#### context\_window\_limit

```python
@property
def context_window_limit() -> int | None
```

Defined in: [src/strands/models/model.py:179](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L179)

Maximum context window size in tokens, or None if not configured.

#### update\_config

```python
@abc.abstractmethod
def update_config(**model_config: Any) -> None
```

Defined in: [src/strands/models/model.py:190](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L190)

Update the model configuration with the provided arguments.

**Arguments**:

-   `**model_config` - Configuration overrides.

#### get\_config

```python
@abc.abstractmethod
def get_config() -> Any
```

Defined in: [src/strands/models/model.py:200](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L200)

Return the model configuration.

**Returns**:

The model’s configuration.

#### structured\_output

```python
@abc.abstractmethod
def structured_output(
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any) -> AsyncGenerator[dict[str, T | Any], None]
```

Defined in: [src/strands/models/model.py:210](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L210)

Get structured output from the model.

**Arguments**:

-   `output_model` - The output model to use for the agent.
-   `prompt` - The prompt messages to use for the agent.
-   `system_prompt` - System prompt to provide context to the model.
-   `**kwargs` - Additional keyword arguments for future extensibility.

**Yields**:

Model events with the last being the structured output.

**Raises**:

-   `ValidationException` - The response format from the model does not match the output\_model

#### stream

```python
@abc.abstractmethod
def stream(messages: Messages,
           tool_specs: list[ToolSpec] | None = None,
           system_prompt: str | None = None,
           *,
           tool_choice: ToolChoice | None = None,
           system_prompt_content: list[SystemContentBlock] | None = None,
           invocation_state: dict[str, Any] | None = None,
           **kwargs: Any) -> AsyncIterable[StreamEvent]
```

Defined in: [src/strands/models/model.py:231](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L231)

Stream conversation with the model.

This method handles the full lifecycle of conversing with the model:

1.  Format the messages, tool specs, and configuration into a streaming request
2.  Send the request to the model
3.  Yield the formatted message chunks

**Arguments**:

-   `messages` - List of message objects to be processed by the model.
-   `tool_specs` - List of tool specifications to make available to the model.
-   `system_prompt` - System prompt to provide context to the model.
-   `tool_choice` - Selection strategy for tool invocation.
-   `system_prompt_content` - System prompt content blocks for advanced features like caching.
-   `invocation_state` - Caller-provided state/context that was passed to the agent when it was invoked.
-   `**kwargs` - Additional keyword arguments for future extensibility.

**Yields**:

Formatted message chunks from the model.

**Raises**:

-   `ModelThrottledException` - When the model service is throttling requests from the client.

#### count\_tokens

```python
async def count_tokens(
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None) -> int
```

Defined in: [src/strands/models/model.py:267](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L267)

Estimate token count for the given input before sending to the model.

Used for proactive context management (e.g., triggering compression at a threshold). Uses tiktoken’s cl100k\_base encoding when available, otherwise falls back to a heuristic (characters / 4 for text, characters / 2 for JSON). Accuracy varies by model provider. Not intended for billing or precise quota calculations.

Subclasses may override this method to provide model-specific token counting using native APIs for improved accuracy.

**Arguments**:

-   `messages` - List of message objects to estimate tokens for.
-   `tool_specs` - List of tool specifications to include in the estimate.
-   `system_prompt` - Plain string system prompt. Ignored if system\_prompt\_content is provided.
-   `system_prompt_content` - Structured system prompt content blocks. Takes priority over system\_prompt.

**Returns**:

Estimated total input tokens.

#### estimate\_utilization

```python
def estimate_utilization(input_tokens: int) -> float
```

Defined in: [src/strands/models/model.py:295](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L295)

Estimate the fraction of the model’s context window consumed by the given input token count.

Resolves the model’s context window limit (falling back to :data:`DEFAULT_CONTEXT_WINDOW_LIMIT` with a warning when not configured) and returns `input_tokens / context_window_limit`.

**Arguments**:

-   `input_tokens` - Total input token count (e.g. from a model event’s projected input tokens).

**Returns**:

Token usage ratio (0–1+; above 1.0 means overflow).

## \_ModelPlugin

```python
class _ModelPlugin(Plugin)
```

Defined in: [src/strands/models/model.py:321](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L321)

Plugin that manages model-related lifecycle hooks.

#### name

```python
@property
def name() -> str
```

Defined in: [src/strands/models/model.py:325](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L325)

A stable string identifier for this plugin.

#### init\_agent

```python
def init_agent(agent: "Agent") -> None
```

Defined in: [src/strands/models/model.py:343](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/model.py#L343)

Register model lifecycle hooks with the agent.

**Arguments**:

-   `agent` - The agent instance to register hooks with.