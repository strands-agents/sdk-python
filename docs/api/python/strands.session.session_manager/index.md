Session manager interface for agent session management.

## SessionManager

```python
class SessionManager(HookProvider, ABC)
```

Defined in: [src/strands/session/session\_manager.py:31](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py#L31)

Abstract interface for managing sessions.

A session manager is in charge of persisting the conversation and state of an agent across its interaction. Changes made to the agents conversation, state, or other attributes should be persisted immediately after they are changed. The different methods introduced in this class are called at important lifecycle events for an agent, and should be persisted in the session.

#### session\_id

The unique session identifier for this session manager.

#### register\_hooks

```python
def register_hooks(registry: HookRegistry, **kwargs: Any) -> None
```

Defined in: [src/strands/session/session\_manager.py:43](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py#L43)

Register hooks for persisting the agent to the session.

#### redact\_latest\_message

```python
@abstractmethod
def redact_latest_message(redact_message: Message, agent: "Agent",
                          **kwargs: Any) -> None
```

Defined in: [src/strands/session/session\_manager.py:68](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py#L68)

Redact the message most recently appended to the agent in the session.

**Arguments**:

-   `redact_message` - New message to use that contains the redact content
-   `agent` - Agent to apply the message redaction to
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### append\_message

```python
@abstractmethod
def append_message(message: Message, agent: "Agent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/session\_manager.py:78](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py#L78)

Append a message to the agent’s session.

**Arguments**:

-   `message` - Message to add to the agent in the session
-   `agent` - Agent to append the message to
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### sync\_agent

```python
@abstractmethod
def sync_agent(agent: "Agent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/session\_manager.py:88](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py#L88)

Serialize and sync the agent with the session storage.

**Arguments**:

-   `agent` - Agent who should be synchronized with the session storage
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### initialize

```python
@abstractmethod
def initialize(agent: "Agent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/session\_manager.py:97](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py#L97)

Initialize an agent with a session.

**Arguments**:

-   `agent` - Agent to initialize
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### sync\_multi\_agent

```python
def sync_multi_agent(source: "MultiAgentBase", **kwargs: Any) -> None
```

Defined in: [src/strands/session/session\_manager.py:105](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py#L105)

Serialize and sync multi-agent with the session storage.

**Arguments**:

-   `source` - Multi-agent source object to persist
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### initialize\_multi\_agent

```python
def initialize_multi_agent(source: "MultiAgentBase", **kwargs: Any) -> None
```

Defined in: [src/strands/session/session\_manager.py:118](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py#L118)

Read multi-agent state from persistent storage.

**Arguments**:

-   `**kwargs` - Additional keyword arguments for future extensibility.
-   `source` - Multi-agent state to initialize.

**Returns**:

Multi-agent state dictionary or empty dict if not found.

#### initialize\_bidi\_agent

```python
def initialize_bidi_agent(agent: "BidiAgent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/session\_manager.py:135](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py#L135)

Initialize a bidirectional agent with a session.

**Arguments**:

-   `agent` - BidiAgent to initialize
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### append\_bidi\_message

```python
def append_bidi_message(message: Message, agent: "BidiAgent",
                        **kwargs: Any) -> None
```

Defined in: [src/strands/session/session\_manager.py:148](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py#L148)

Append a message to the bidirectional agent’s session.

**Arguments**:

-   `message` - Message to add to the agent in the session
-   `agent` - BidiAgent to append the message to
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### sync\_bidi\_agent

```python
def sync_bidi_agent(agent: "BidiAgent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/session\_manager.py:162](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/session_manager.py#L162)

Serialize and sync the bidirectional agent with the session storage.

**Arguments**:

-   `agent` - BidiAgent who should be synchronized with the session storage
-   `**kwargs` - Additional keyword arguments for future extensibility.