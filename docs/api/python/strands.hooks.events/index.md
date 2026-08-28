Hook events emitted as part of invoking Agents.

This module defines the events that are emitted as Agents run through the lifecycle of a request.

## AgentInitializedEvent

```python
@dataclass
class AgentInitializedEvent(HookEvent)
```

Defined in: [src/strands/hooks/events.py:27](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L27)

Event triggered when an agent has finished initialization.

This event is fired after the agent has been fully constructed and all built-in components have been initialized. Hook providers can use this event to perform setup tasks that require a fully initialized agent.

## BeforeInvocationEvent

```python
@dataclass
class BeforeInvocationEvent(HookEvent)
```

Defined in: [src/strands/hooks/events.py:39](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L39)

Event triggered at the beginning of a new agent request.

This event is fired before the agent begins processing a new user request, before any model inference or tool execution occurs. Hook providers can use this event to perform request-level setup, logging, or validation.

This event is triggered at the beginning of the following api calls:

-   Agent.**call**
-   Agent.stream\_async
-   Agent.structured\_output

**Attributes**:

-   `invocation_state` - State and configuration passed through the agent invocation. This can include shared context for multi-agent coordination, request tracking, and dynamic configuration.
-   `messages` - The input messages for this invocation. Can be modified by hooks to redact or transform content before processing.
-   `cancel` - When set, cancels the invocation. If a string, used as the cancellation message. If True, a default message is used.

## AfterInvocationEvent

```python
@dataclass
class AfterInvocationEvent(HookEvent)
```

Defined in: [src/strands/hooks/events.py:70](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L70)

Event triggered at the end of an agent request.

This event is fired after the agent has completed processing a request, regardless of whether it completed successfully or encountered an error. Hook providers can use this event for cleanup, logging, or state persistence.

Note: This event uses reverse callback ordering, meaning callbacks registered later will be invoked first during cleanup.

This event is triggered at the end of the following api calls:

-   Agent.**call**
-   Agent.stream\_async
-   Agent.structured\_output

Resume: When `resume` is set to a non-None value by a hook callback, the agent will automatically re-invoke itself with the provided input. This enables hooks to implement autonomous looping patterns where the agent continues processing based on its previous result. The resume triggers a full new invocation cycle including `BeforeInvocationEvent`.

**Attributes**:

-   `invocation_state` - State and configuration passed through the agent invocation. This can include shared context for multi-agent coordination, request tracking, and dynamic configuration.
-   `result` - The result of the agent invocation, if available. This will be None when invoked from structured\_output methods, as those return typed output directly rather than AgentResult.
-   `resume` - When set to a non-None agent input by a hook callback, the agent will re-invoke itself with this input. The value can be any valid AgentInput (str, content blocks, messages, etc.). Defaults to None (no resume).

#### should\_reverse\_callbacks

```python
@property
def should_reverse_callbacks() -> bool
```

Defined in: [src/strands/hooks/events.py:112](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L112)

True to invoke callbacks in reverse order.

## MessageAddedEvent

```python
@dataclass
class MessageAddedEvent(HookEvent)
```

Defined in: [src/strands/hooks/events.py:118](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L118)

Event triggered when a message is added to the agent’s conversation.

This event is fired whenever the agent adds a new message to its internal message history, including user messages, assistant responses, and tool results. Hook providers can use this event for logging, monitoring, or implementing custom message processing logic.

Note: This event is only triggered for messages added by the framework itself, not for messages manually added by tools or external code.

**Attributes**:

-   `message` - The message that was added to the conversation history.

## BeforeToolsEvent

```python
@dataclass
class BeforeToolsEvent(HookEvent, _Interruptible)
```

Defined in: [src/strands/hooks/events.py:137](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L137)

Event triggered before executing tools.

This event is fired when the model returns tool use blocks that need to be executed. Hook callbacks can set `cancel` to prevent all tools from executing. Fires once per cycle, so may fire more than once per assistant message when a per-tool interrupt splits the batch.

**Attributes**:

-   `message` - The assistant message containing tool use requests.
-   `invocation_state` - State and configuration passed through the agent invocation.
-   `cancel` - When set, cancels all tool calls. If a string, used as the tool result error message. If True, a default message is used.

## AfterToolsEvent

```python
@dataclass
class AfterToolsEvent(HookEvent)
```

Defined in: [src/strands/hooks/events.py:173](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L173)

Event triggered after all tools complete execution.

This event is fired after tool results are collected and ready to be added to conversation. Paired with a preceding `BeforeToolsEvent` when the batch proceeds past the pre-execution phase (cancel, interrupt, and error paths included). Fires once per cycle, so may fire more than once per assistant message when a per-tool interrupt splits the batch.

Note: This event uses reverse callback ordering, meaning callbacks registered later will be invoked first during cleanup.

**Attributes**:

-   `message` - The user-role message containing the tool results.
-   `invocation_state` - State and configuration passed through the agent invocation.
-   `end_turn` - When set, the agent loop halts after this tool batch without calling the model again. A string becomes the final assistant text. A list of content blocks becomes the final assistant message content. If True, a default message is used. In any case, the stop\_reason is “end\_turn”.

#### should\_reverse\_callbacks

```python
@property
def should_reverse_callbacks() -> bool
```

Defined in: [src/strands/hooks/events.py:202](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L202)

True to invoke callbacks in reverse order.

## BeforeToolCallEvent

```python
@dataclass
class BeforeToolCallEvent(HookEvent, _Interruptible)
```

Defined in: [src/strands/hooks/events.py:208](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L208)

Event triggered before a tool is invoked.

This event is fired just before the agent executes a tool, allowing hook providers to inspect, modify, or replace the tool that will be executed. The selected\_tool can be modified by hook callbacks to change which tool gets executed.

**Attributes**:

-   `selected_tool` - The tool that will be invoked. Can be modified by hooks to change which tool gets executed. This may be None if tool lookup failed.
-   `tool_use` - The tool parameters that will be passed to selected\_tool.
-   `invocation_state` - Keyword arguments that will be passed to the tool.
-   `cancel_tool` - A user defined message that when set, will cancel the tool call. The message will be placed into a tool result with an error status. If set to `True`, Strands will cancel the tool call and use a default cancel message.

## AfterToolCallEvent

```python
@dataclass
class AfterToolCallEvent(HookEvent)
```

Defined in: [src/strands/hooks/events.py:248](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L248)

Event triggered after a tool invocation completes.

This event is fired after the agent has finished executing a tool, regardless of whether the execution was successful or resulted in an error. Hook providers can use this event for cleanup, logging, or post-processing.

Note: This event uses reverse callback ordering, meaning callbacks registered later will be invoked first during cleanup.

Tool Retrying: When `retry` is set to True by a hook callback, the tool executor will discard the current tool result and invoke the tool again. This has important implications for streaming consumers:

-   ToolStreamEvents (intermediate streaming events) from the discarded tool execution will have already been emitted to callers before the retry occurs. Agent invokers consuming streamed events should be prepared to handle this scenario, potentially by tracking retry state or implementing idempotent event processing
-   ToolResultEvent is NOT emitted for discarded attempts - only the final attempt’s result is emitted and added to the conversation history

**Attributes**:

-   `selected_tool` - The tool that was invoked. It may be None if tool lookup failed.
-   `tool_use` - The tool parameters that were passed to the tool invoked.
-   `invocation_state` - Keyword arguments that were passed to the tool
-   `result` - The result of the tool invocation. Either a ToolResult on success or an Exception if the tool execution failed.
-   `cancel_message` - The cancellation message if the user cancelled the tool call.
-   `duration` - Elapsed time in seconds spent executing the tool. Starts after BeforeToolCallEvent returns and stops before AfterToolCallEvent is constructed. None when the tool call was cancelled by a BeforeToolCallEvent hook before execution.
-   `retry` - Whether to retry the tool invocation. Can be set by hook callbacks to trigger a retry. When True, the current result is discarded and the tool is called again. Defaults to False.

#### should\_reverse\_callbacks

```python
@property
def should_reverse_callbacks() -> bool
```

Defined in: [src/strands/hooks/events.py:299](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L299)

True to invoke callbacks in reverse order.

## BeforeModelCallEvent

```python
@dataclass
class BeforeModelCallEvent(HookEvent)
```

Defined in: [src/strands/hooks/events.py:305](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L305)

Event triggered before the model is invoked.

This event is fired just before the agent calls the model for inference, allowing hook providers to inspect or modify the messages and configuration that will be sent to the model.

Note: This event is not fired for invocations to structured\_output.

**Attributes**:

-   `invocation_state` - State and configuration passed through the agent invocation. This can include shared context for multi-agent coordination, request tracking, and dynamic configuration.
-   `projected_input_tokens` - Projected input token count for the upcoming model call. Computed by the agent loop from message metadata and token estimation. Available for hooks and plugins (e.g. conversation managers) to make proactive decisions about context management. None if estimation failed.
-   `cancel` - When set, cancels the model call. If a string, used as the cancellation message. If True, a default message is used.

## AfterModelCallEvent

```python
@dataclass
class AfterModelCallEvent(HookEvent)
```

Defined in: [src/strands/hooks/events.py:335](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L335)

Event triggered after the model invocation completes.

This event is fired after the agent has finished calling the model, regardless of whether the invocation was successful or resulted in an error. Hook providers can use this event for cleanup, logging, or post-processing.

Note: This event uses reverse callback ordering, meaning callbacks registered later will be invoked first during cleanup.

Note: This event is not fired for invocations to structured\_output.

Model Retrying: When `retry` is set to True by a hook callback, the agent will discard the current model response and invoke the model again. This has important implications for streaming consumers:

-   Streaming events from the discarded response will have already been emitted to callers before the retry occurs. Agent invokers consuming streamed events should be prepared to handle this scenario, potentially by tracking retry state or implementing idempotent event processing
-   The original model message is thrown away internally and not added to the conversation history

**Attributes**:

-   `invocation_state` - State and configuration passed through the agent invocation. This can include shared context for multi-agent coordination, request tracking, and dynamic configuration.
-   `stop_response` - The model response data if invocation was successful, None if failed.
-   `exception` - Exception if the model invocation failed, None if successful.
-   `retry` - Whether to retry the model invocation. Can be set by hook callbacks to trigger a retry. When True, the current response is discarded and the model is called again. Defaults to False.

## ModelStopResponse

```python
@dataclass
class ModelStopResponse()
```

Defined in: [src/strands/hooks/events.py:371](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L371)

Model response data from successful invocation.

**Attributes**:

-   `stop_reason` - The reason the model stopped generating.
-   `message` - The generated message from the model.

#### should\_reverse\_callbacks

```python
@property
def should_reverse_callbacks() -> bool
```

Defined in: [src/strands/hooks/events.py:391](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L391)

True to invoke callbacks in reverse order.

## MultiAgentInitializedEvent

```python
@dataclass
class MultiAgentInitializedEvent(BaseHookEvent)
```

Defined in: [src/strands/hooks/events.py:398](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L398)

Event triggered when multi-agent orchestrator initialized.

**Attributes**:

-   `source` - The multi-agent orchestrator instance
-   `invocation_state` - Configuration that user passes in

## BeforeNodeCallEvent

```python
@dataclass
class BeforeNodeCallEvent(BaseHookEvent, _Interruptible)
```

Defined in: [src/strands/hooks/events.py:411](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L411)

Event triggered before individual node execution starts.

**Attributes**:

-   `source` - The multi-agent orchestrator instance
-   `node_id` - ID of the node about to execute
-   `invocation_state` - Configuration that user passes in
-   `cancel_node` - A user defined message that when set, will cancel the node execution with status FAILED. The message will be emitted under a MultiAgentNodeCancel event. If set to `True`, Strands will cancel the node using a default cancel message.

## AfterNodeCallEvent

```python
@dataclass
class AfterNodeCallEvent(BaseHookEvent)
```

Defined in: [src/strands/hooks/events.py:447](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L447)

Event triggered after individual node execution completes.

**Attributes**:

-   `source` - The multi-agent orchestrator instance
-   `node_id` - ID of the node that just completed execution
-   `invocation_state` - Configuration that user passes in

#### should\_reverse\_callbacks

```python
@property
def should_reverse_callbacks() -> bool
```

Defined in: [src/strands/hooks/events.py:461](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L461)

True to invoke callbacks in reverse order.

## BeforeMultiAgentInvocationEvent

```python
@dataclass
class BeforeMultiAgentInvocationEvent(BaseHookEvent)
```

Defined in: [src/strands/hooks/events.py:467](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L467)

Event triggered before orchestrator execution starts.

**Attributes**:

-   `source` - The multi-agent orchestrator instance
-   `invocation_state` - Configuration that user passes in

## AfterMultiAgentInvocationEvent

```python
@dataclass
class AfterMultiAgentInvocationEvent(BaseHookEvent)
```

Defined in: [src/strands/hooks/events.py:480](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L480)

Event triggered after orchestrator execution completes.

**Attributes**:

-   `source` - The multi-agent orchestrator instance
-   `invocation_state` - Configuration that user passes in

#### should\_reverse\_callbacks

```python
@property
def should_reverse_callbacks() -> bool
```

Defined in: [src/strands/hooks/events.py:492](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/hooks/events.py#L492)

True to invoke callbacks in reverse order.