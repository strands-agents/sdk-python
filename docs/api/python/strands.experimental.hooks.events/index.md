Experimental hook events emitted as part of invoking Agents and BidiAgents.

This module defines the events that are emitted as Agents and BidiAgents run through the lifecycle of a request.

## BidiHookEvent

```python
@dataclass
class BidiHookEvent(BaseHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:43](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L43)

Base class for BidiAgent hook events.

**Attributes**:

-   `agent` - The BidiAgent instance that triggered this event.

## BidiAgentInitializedEvent

```python
@dataclass
class BidiAgentInitializedEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:54](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L54)

Event triggered when a BidiAgent has finished initialization.

This event is fired after the BidiAgent has been fully constructed and all built-in components have been initialized. Hook providers can use this event to perform setup tasks that require a fully initialized agent.

## BidiBeforeInvocationEvent

```python
@dataclass
class BidiBeforeInvocationEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:66](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L66)

Event triggered when BidiAgent starts a streaming session.

This event is fired before the BidiAgent begins a streaming session, before any model connection or audio processing occurs. Hook providers can use this event to perform session-level setup, logging, or validation.

This event is triggered at the beginning of agent.start().

## BidiAfterInvocationEvent

```python
@dataclass
class BidiAfterInvocationEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:80](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L80)

Event triggered when BidiAgent ends a streaming session.

This event is fired after the BidiAgent has completed a streaming session, regardless of whether it completed successfully or encountered an error. Hook providers can use this event for cleanup, logging, or state persistence.

Note: This event uses reverse callback ordering, meaning callbacks registered later will be invoked first during cleanup.

This event is triggered at the end of agent.stop().

#### should\_reverse\_callbacks

```python
@property
def should_reverse_callbacks() -> bool
```

Defined in: [src/strands/experimental/hooks/events.py:94](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L94)

True to invoke callbacks in reverse order.

## BidiMessageAddedEvent

```python
@dataclass
class BidiMessageAddedEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:100](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L100)

Event triggered when BidiAgent adds a message to the conversation.

This event is fired whenever the BidiAgent adds a new message to its internal message history, including user messages (from transcripts), assistant responses, and tool results. Hook providers can use this event for logging, monitoring, or implementing custom message processing logic.

Note: This event is only triggered for messages added by the framework itself, not for messages manually added by tools or external code.

**Attributes**:

-   `message` - The message that was added to the conversation history.

## BidiInterruptionEvent

```python
@dataclass
class BidiInterruptionEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:119](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L119)

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

Defined in: [src/strands/experimental/hooks/events.py:139](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L139)

Event emitted before the agent restarts the model connection.

A restart is triggered either reactively, after the model reports a timeout, or proactively, when the reconnect timer fires ahead of the provider’s limit.

**Attributes**:

-   `reason` - What triggered the restart (“timeout” reactively, “scheduled” proactively).
-   `timeout_error` - The model’s timeout error on the reactive path; None when scheduled.

## BidiAfterConnectionRestartEvent

```python
@dataclass
class BidiAfterConnectionRestartEvent(BidiHookEvent)
```

Defined in: [src/strands/experimental/hooks/events.py:155](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/hooks/events.py#L155)

Event emitted after the agent attempts to restart the model connection.

**Attributes**:

-   `reason` - What triggered the restart (“timeout” reactively, “scheduled” proactively).
-   `exception` - Populated if an exception was raised during the restart. None means success.