Experimental hook events emitted as part of invoking Agents and BidiAgents.

This module defines the events that are emitted as Agents and BidiAgents run through the lifecycle of a request.

## BidiHookEvent

```python
@dataclass
class BidiHookEvent(BaseHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:44](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L44)

Base class for BidiAgent hook events.

**Attributes**:

-   `agent` - The BidiAgent instance that triggered this event.

## BidiAgentInitializedEvent

```python
@dataclass
class BidiAgentInitializedEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:55](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L55)

Event triggered when a BidiAgent has finished initialization.

This event is fired after the BidiAgent has been fully constructed and all built-in components have been initialized. Hook providers can use this event to perform setup tasks that require a fully initialized agent.

## BidiBeforeInvocationEvent

```python
@dataclass
class BidiBeforeInvocationEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:67](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L67)

Event triggered when BidiAgent starts a streaming session.

This event is fired before the BidiAgent begins a streaming session, before any model connection or audio processing occurs. Hook providers can use this event to perform session-level setup, logging, or validation.

This event is triggered at the beginning of agent.start().

## BidiAfterInvocationEvent

```python
@dataclass
class BidiAfterInvocationEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:81](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L81)

Event triggered when BidiAgent ends a streaming session.

This event is fired after the BidiAgent has completed a streaming session, regardless of whether it completed successfully or encountered an error. Hook providers can use this event for cleanup, logging, or state persistence.

Note: This event uses reverse callback ordering, meaning callbacks registered later will be invoked first during cleanup.

This event is triggered at the end of agent.stop().

#### should\_reverse\_callbacks

```python
@property
def should_reverse_callbacks() -> bool
```

Defined in: [src/strands/experimental/hooks/events.py:95](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L95)

True to invoke callbacks in reverse order.

## BidiMessageAddedEvent

```python
@dataclass
class BidiMessageAddedEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:101](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L101)

Event triggered when BidiAgent adds a message to the conversation.

This event is fired whenever the BidiAgent adds a new message to its internal message history, including user messages (from transcripts), assistant responses, and tool results. Hook providers can use this event for logging, monitoring, or implementing custom message processing logic.

Note: This event is only triggered for messages added by the framework itself, not for messages manually added by tools or external code.

**Attributes**:

-   `message` - The message that was added to the conversation history.

## BidiBeforeToolCallEvent

```python
@dataclass
class BidiBeforeToolCallEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:120](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L120)

Event triggered before BidiAgent executes a tool.

This event is fired just before the BidiAgent executes a tool during a streaming session, allowing hook providers to inspect, modify, or replace the tool that will be executed. The selected\_tool can be modified by hook callbacks to change which tool gets executed.

**Attributes**:

-   `selected_tool` - The tool that will be invoked. Can be modified by hooks to change which tool gets executed. This may be None if tool lookup failed.
-   `tool_use` - The tool parameters that will be passed to selected\_tool.
-   `invocation_state` - Keyword arguments that will be passed to the tool.
-   `cancel_tool` - A user defined message that when set, will cancel the tool call. The message will be placed into a tool result with an error status. If set to `True`, Strands will cancel the tool call and use a default cancel message.

## BidiAfterToolCallEvent

```python
@dataclass
class BidiAfterToolCallEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:148](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L148)

Event triggered after BidiAgent executes a tool.

This event is fired after the BidiAgent has finished executing a tool during a streaming session, regardless of whether the execution was successful or resulted in an error. Hook providers can use this event for cleanup, logging, or post-processing.

Note: This event uses reverse callback ordering, meaning callbacks registered later will be invoked first during cleanup.

**Attributes**:

-   `selected_tool` - The tool that was invoked. It may be None if tool lookup failed.
-   `tool_use` - The tool parameters that were passed to the tool invoked.
-   `invocation_state` - Keyword arguments that were passed to the tool.
-   `result` - The result of the tool invocation. Either a ToolResult on success or an Exception if the tool execution failed.
-   `exception` - Exception if the tool execution failed, None if successful.
-   `cancel_message` - The cancellation message if the user cancelled the tool call.
-   `duration` - Elapsed time in seconds spent executing the tool. Starts after BeforeToolCallEvent returns and stops before AfterToolCallEvent is constructed. None when the tool call was cancelled by a BeforeToolCallEvent hook before execution.

#### should\_reverse\_callbacks

```python
@property
def should_reverse_callbacks() -> bool
```

Defined in: [src/strands/experimental/hooks/events.py:185](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L185)

True to invoke callbacks in reverse order.

## BidiInterruptionEvent

```python
@dataclass
class BidiInterruptionEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:191](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L191)

Event triggered when model generation is interrupted.

This event is fired when the user interrupts the assistant (e.g., by speaking during the assistant’s response) or when an error causes interruption. This is specific to bidirectional streaming and doesn’t exist in standard agents.

Hook providers can use this event to log interruptions, implement custom interruption handling, or trigger cleanup logic.

**Attributes**:

-   `reason` - The reason for the interruption (“user\_speech” or “error”).
-   `interrupted_response_id` - Optional ID of the response that was interrupted.

## BidiBeforeConnectionRestartEvent

```python
@dataclass
class BidiBeforeConnectionRestartEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:211](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L211)

Event emitted before agent attempts to restart model connection after timeout.

**Attributes**:

-   `timeout_error` - Timeout error reported by the model.

## BidiAfterConnectionRestartEvent

```python
@dataclass
class BidiAfterConnectionRestartEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:222](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L222)

Event emitted after agent attempts to restart model connection after timeout.

Attribtues: exception: Populated if exception was raised during connection restart. None value means the restart was successful.