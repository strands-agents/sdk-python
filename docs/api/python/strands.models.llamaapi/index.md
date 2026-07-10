Llama API model provider.

-   Docs: [https://llama.developer.meta.com/](https://llama.developer.meta.com/)

## LlamaAPIModel

```python
class LlamaAPIModel(Model)
```

Defined in: [src/strands/models/llamaapi.py:31](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L31)

Llama API model provider implementation.

## LlamaConfig

```python
class LlamaConfig(BaseModelConfig)
```

Defined in: [src/strands/models/llamaapi.py:43](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L43)

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

Defined in: [src/strands/models/llamaapi.py:62](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L62)

Initialize provider instance.

**Arguments**:

-   `client_args` - Arguments for the Llama API client.
-   `**model_config` - Configuration options for the Llama API model.

#### update\_config

```python
@override
def update_config(**model_config: Unpack[LlamaConfig]) -> None
```

Defined in: [src/strands/models/llamaapi.py:84](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L84)

Update the Llama API Model configuration with the provided arguments.

**Arguments**:

-   `**model_config` - Configuration overrides.

#### get\_config

```python
@override
def get_config() -> LlamaConfig
```

Defined in: [src/strands/models/llamaapi.py:94](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L94)

Get the Llama API model configuration.

**Returns**:

The Llama API model configuration.

#### format\_request

```python
def format_request(messages: Messages,
                   tool_specs: list[ToolSpec] | None = None,
                   system_prompt: str | None = None) -> dict[str, Any]
```

Defined in: [src/strands/models/llamaapi.py:224](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L224)

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

Defined in: [src/strands/models/llamaapi.py:270](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L270)

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

Defined in: [src/strands/models/llamaapi.py:344](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L344)

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

Defined in: [src/strands/models/llamaapi.py:439](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/llamaapi.py#L439)

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