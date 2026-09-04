Llama API model provider.

Deprecated: The Llama API service has been deprecated by Meta. This provider will be removed in v2.0.0. Migrate to another provider (BedrockModel, OllamaModel, or OpenAIModel) that hosts Llama or comparable models.

## LlamaAPIModel

```python
@deprecated(
    "LlamaAPIModel is deprecated and will be removed in v2.0.0. "
    "The underlying Llama API service has been deprecated by Meta. "
    "Use BedrockModel, OllamaModel, or OpenAIModel instead."
)
class LlamaAPIModel(Model)
```

Defined in: [src/strands/models/llamaapi.py:38](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L38)

Llama API model provider implementation.

Deprecated: The Llama API service has been deprecated by Meta. This class will be removed in v2.0.0. Use BedrockModel, OllamaModel, or OpenAIModel instead.

## LlamaConfig

```python
class LlamaConfig(BaseModelConfig)
```

Defined in: [src/strands/models/llamaapi.py:54](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L54)

Configuration options for Llama API models.

**Attributes**:

-   `model_id` - Model ID (e.g., “Llama-4-Maverick-17B-128E-Instruct-FP8”).
-   `repetition_penalty` - Repetition penalty.
-   `temperature` - Temperature.
-   `top_p` - Top-p.
-   `max_completion_tokens` - Maximum completion tokens.
-   `top_k` - Top-k.

#### \_\_init\_\_

```python
def __init__(*,
             client_args: dict[str, Any] | None = None,
             **model_config: Unpack[LlamaConfig]) -> None
```

Defined in: [src/strands/models/llamaapi.py:73](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L73)

Initialize provider instance.

**Arguments**:

-   `client_args` - Arguments for the Llama API client.
-   `**model_config` - Configuration options for the Llama API model.

#### update\_config

```python
@override
def update_config(**model_config: Unpack[LlamaConfig]) -> None
```

Defined in: [src/strands/models/llamaapi.py:95](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L95)

Update the Llama API Model configuration with the provided arguments.

**Arguments**:

-   `**model_config` - Configuration overrides.

#### get\_config

```python
@override
def get_config() -> LlamaConfig
```

Defined in: [src/strands/models/llamaapi.py:105](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L105)

Get the Llama API model configuration.

**Returns**:

The Llama API model configuration.

#### format\_request

```python
def format_request(messages: Messages,
                   tool_specs: list[ToolSpec] | None = None,
                   system_prompt: str | None = None) -> dict[str, Any]
```

Defined in: [src/strands/models/llamaapi.py:235](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L235)

Format a Llama API chat streaming request.

**Arguments**:

-   `messages` - List of message objects to be processed by the model.
-   `tool_specs` - List of tool specifications to make available to the model.
-   `system_prompt` - System prompt to provide context to the model.

**Returns**:

An Llama API chat streaming request.

**Raises**:

-   `TypeError` - If a message contains a content block type that cannot be converted to a LlamaAPI-compatible format.

#### format\_chunk

```python
def format_chunk(event: dict[str, Any]) -> StreamEvent
```

Defined in: [src/strands/models/llamaapi.py:281](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L281)

Format the Llama API model response events into standardized message chunks.

**Arguments**:

-   `event` - A response event from the model.

**Returns**:

The formatted chunk.

#### stream

```python
@override
async def stream(messages: Messages,
                 tool_specs: list[ToolSpec] | None = None,
                 system_prompt: str | None = None,
                 *,
                 tool_choice: ToolChoice | None = None,
                 **kwargs: Any) -> AsyncGenerator[StreamEvent, None]
```

Defined in: [src/strands/models/llamaapi.py:355](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L355)

Stream conversation with the LlamaAPI model.

**Arguments**:

-   `messages` - List of message objects to be processed by the model.
-   `tool_specs` - List of tool specifications to make available to the model.
-   `system_prompt` - System prompt to provide context to the model.
-   `tool_choice` - Selection strategy for tool invocation. **Note: This parameter is accepted for interface consistency but is currently ignored for this model provider.**
-   `**kwargs` - Additional keyword arguments for future extensibility.

**Yields**:

Formatted message chunks from the model.

**Raises**:

-   `ContextWindowOverflowException` - If the input exceeds the model’s context window.
-   `ModelThrottledException` - When the model service is throttling requests from the client.

#### structured\_output

```python
@override
def structured_output(
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any) -> AsyncGenerator[dict[str, T | Any], None]
```

Defined in: [src/strands/models/llamaapi.py:450](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L450)

Get structured output from the model.

**Arguments**:

-   `output_model` - The output model to use for the agent.
-   `prompt` - The prompt messages to use for the agent.
-   `system_prompt` - System prompt to provide context to the model.
-   `**kwargs` - Additional keyword arguments for future extensibility.

**Yields**:

Model events with the last being the structured output.

**Raises**:

-   `NotImplementedError` - Structured output is not currently supported for LlamaAPI models.