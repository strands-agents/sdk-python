Anthropic Claude model provider.

-   Docs: [https://docs.anthropic.com/claude/reference/getting-started-with-the-api](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)

## AnthropicModel

```python
class AnthropicModel(Model)
```

Defined in: [src/strands/models/anthropic.py:50](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/anthropic.py#L50)

Anthropic model provider implementation.

## AnthropicConfig

```python
class AnthropicConfig(BaseModelConfig)
```

Defined in: [src/strands/models/anthropic.py:68](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/anthropic.py#L68)

Configuration options for Anthropic models.

**Attributes**:

-   `cache_config` - Configuration for prompt caching. Adds a cache point to the last user message, caching everything before it. Caching is off when unset.
-   `cache_tools` - Caches the tool definitions.
-   `max_tokens` - Maximum number of tokens to generate.
-   `model_id` - Calude model ID (e.g., “claude-3-7-sonnet-latest”). For a complete list of supported models, see [https://docs.anthropic.com/en/docs/about-claude/models/all-models](https://docs.anthropic.com/en/docs/about-claude/models/all-models).
-   `params` - Additional model parameters (e.g., temperature). For a complete list of supported parameters, see [https://docs.anthropic.com/en/api/messages](https://docs.anthropic.com/en/api/messages).
-   `use_native_token_count` - Whether to use the native Anthropic count\_tokens API. When True, count\_tokens() calls the Anthropic API for accurate counts. When False (default), skips the API call and uses the local estimator.

#### \_\_init\_\_

```python
def __init__(*,
             client_args: dict[str, Any] | None = None,
             **model_config: Unpack[AnthropicConfig])
```

Defined in: [src/strands/models/anthropic.py:93](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/anthropic.py#L93)

Initialize provider instance.

**Arguments**:

-   `client_args` - Arguments for the underlying Anthropic client (e.g., api\_key). For a complete list of supported arguments, see [https://docs.anthropic.com/en/api/client-sdks](https://docs.anthropic.com/en/api/client-sdks).
-   `**model_config` - Configuration options for the Anthropic model.

#### update\_config

```python
@override
def update_config(**model_config: Unpack[AnthropicConfig]) -> None
```

Defined in: [src/strands/models/anthropic.py:110](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/anthropic.py#L110)

Update the Anthropic model configuration with the provided arguments.

**Arguments**:

-   `**model_config` - Configuration overrides.

#### get\_config

```python
@override
def get_config() -> AnthropicConfig
```

Defined in: [src/strands/models/anthropic.py:120](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/anthropic.py#L120)

Get the Anthropic model configuration.

**Returns**:

The Anthropic model configuration.

#### format\_request

```python
def format_request(
    messages: Messages,
    tool_specs: list[ToolSpec] | None = None,
    system_prompt: str | None = None,
    tool_choice: ToolChoice | None = None,
    dynamic_trailing_blocks: int = 0,
    *,
    system_prompt_content: list[SystemContentBlock] | None = None
) -> dict[str, Any]
```

Defined in: [src/strands/models/anthropic.py:381](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/anthropic.py#L381)

Format an Anthropic streaming request.

**Arguments**:

-   `messages` - List of message objects to be processed by the model.
-   `tool_specs` - List of tool specifications to make available to the model.
-   `system_prompt` - Plain string system prompt. Ignored when system\_prompt\_content is provided.
-   `tool_choice` - Selection strategy for tool invocation.
-   `dynamic_trailing_blocks` - How many trailing blocks of the last user message are rebuilt on every call, so the cache point stays ahead of them.
-   `system_prompt_content` - Structured system prompt content blocks, which can carry a cache point.

**Returns**:

An Anthropic streaming request.

**Raises**:

-   `TypeError` - If a message contains a content block type that cannot be converted to an Anthropic-compatible format.

#### format\_chunk

```python
def format_chunk(event: dict[str, Any]) -> StreamEvent
```

Defined in: [src/strands/models/anthropic.py:505](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/anthropic.py#L505)

Format the Anthropic response events into standardized message chunks.

**Arguments**:

-   `event` - A response event from the Anthropic model.

**Returns**:

The formatted chunk.

**Raises**:

-   `RuntimeError` - If chunk\_type is not recognized. This error should never be encountered as we control chunk\_type in the stream method.

#### count\_tokens

```python
@override
async def count_tokens(
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None) -> int
```

Defined in: [src/strands/models/anthropic.py:632](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/anthropic.py#L632)

Count tokens using Anthropic’s native count\_tokens API.

Uses the same message format as the Messages API to get accurate token counts directly from the Anthropic service.

**Arguments**:

-   `messages` - List of message objects to count tokens for.
-   `tool_specs` - List of tool specifications to include in the count.
-   `system_prompt` - Plain string system prompt. Ignored if system\_prompt\_content is provided.
-   `system_prompt_content` - Structured system prompt content blocks.

**Returns**:

Total input token count.

#### stream

```python
@override
async def stream(messages: Messages,
                 tool_specs: list[ToolSpec] | None = None,
                 system_prompt: str | None = None,
                 *,
                 tool_choice: ToolChoice | None = None,
                 system_prompt_content: list[SystemContentBlock] | None = None,
                 **kwargs: Any) -> AsyncGenerator[StreamEvent, None]
```

Defined in: [src/strands/models/anthropic.py:682](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/anthropic.py#L682)

Stream conversation with the Anthropic model.

**Arguments**:

-   `messages` - List of message objects to be processed by the model.
-   `tool_specs` - List of tool specifications to make available to the model.
-   `system_prompt` - Plain string system prompt. Ignored when system\_prompt\_content is provided.
-   `tool_choice` - Selection strategy for tool invocation.
-   `system_prompt_content` - Structured system prompt content blocks, which can carry a cache point.
-   `**kwargs` - Additional keyword arguments for future extensibility.

**Yields**:

Formatted message chunks from the model.

**Raises**:

-   `ContextWindowOverflowException` - If the input exceeds the model’s context window.
-   `ModelThrottledException` - If the request is throttled by Anthropic.

#### structured\_output

```python
@override
async def structured_output(
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any) -> AsyncGenerator[dict[str, T | Any], None]
```

Defined in: [src/strands/models/anthropic.py:759](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/anthropic.py#L759)

Get structured output from the model.

**Arguments**:

-   `output_model` - The output model to use for the agent.
-   `prompt` - The prompt messages to use for the agent.
-   `system_prompt` - System prompt to provide context to the model.
-   `**kwargs` - Additional keyword arguments for future extensibility.

**Yields**:

Model events with the last being the structured output.