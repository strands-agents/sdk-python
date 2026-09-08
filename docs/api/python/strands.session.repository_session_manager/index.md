Repository session manager implementation.

## RepositorySessionManager

```python
class RepositorySessionManager(SessionManager)
```

Defined in: [src/strands/session/repository\_session\_manager.py:28](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/repository_session_manager.py#L28)

Session manager for persisting agents in a SessionRepository.

This manager uses a :class:`SessionRepository` (a structured per-message CRUD interface), not the unified :class:`~strands.storage.storage.Storage` protocol. It does not resolve from the agent-level `storage` parameter. For snapshot-based persistence that integrates with agent-level storage, use :class:`~strands.session.snapshot_session_manager.SnapshotSessionManager`.

#### \_\_init\_\_

```python
def __init__(session_id: str, session_repository: SessionRepository,
             **kwargs: Any)
```

Defined in: [src/strands/session/repository\_session\_manager.py:40](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/repository_session_manager.py#L40)

Initialize the RepositorySessionManager.

If no session with the specified session\_id exists yet, it will be created in the session\_repository.

**Arguments**:

-   `session_id` - ID to use for the session. A new session with this id will be created if it does not exist in the repository yet
-   `session_repository` - Underlying session repository to use to store the sessions state.
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### append\_message

```python
def append_message(message: Message, agent: "Agent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/repository\_session\_manager.py:78](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/repository_session_manager.py#L78)

Append a message to the agent’s session.

**Arguments**:

-   `message` - Message to add to the agent in the session
-   `agent` - Agent to append the message to
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### redact\_latest\_message

```python
def redact_latest_message(redact_message: Message, agent: "Agent",
                          **kwargs: Any) -> None
```

Defined in: [src/strands/session/repository\_session\_manager.py:97](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/repository_session_manager.py#L97)

Redact the latest message appended to the session.

**Arguments**:

-   `redact_message` - New message to use that contains the redact content
-   `agent` - Agent to apply the message redaction to
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### sync\_agent

```python
def sync_agent(agent: "Agent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/repository\_session\_manager.py:111](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/repository_session_manager.py#L111)

Serialize and update the agent into the session repository.

Only updates the agent if state has been modified or internal state has changed. This optimization reduces unnecessary I/O operations when the agent processes messages without modifying its state.

**Arguments**:

-   `agent` - Agent to sync to the session.
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### initialize

```python
def initialize(agent: "Agent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/repository\_session\_manager.py:178](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/repository_session_manager.py#L178)

Initialize an agent with a session.

**Arguments**:

-   `agent` - Agent to initialize from the session
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### sync\_multi\_agent

```python
def sync_multi_agent(source: "MultiAgentBase", **kwargs: Any) -> None
```

Defined in: [src/strands/session/repository\_session\_manager.py:346](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/repository_session_manager.py#L346)

Serialize and update the multi-agent state into the session repository.

**Arguments**:

-   `source` - Multi-agent source object to sync to the session.
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### initialize\_multi\_agent

```python
def initialize_multi_agent(source: "MultiAgentBase", **kwargs: Any) -> None
```

Defined in: [src/strands/session/repository\_session\_manager.py:355](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/repository_session_manager.py#L355)

Initialize multi-agent state from the session repository.

**Arguments**:

-   `source` - Multi-agent source object to restore state into
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### initialize\_bidi\_agent

```python
def initialize_bidi_agent(agent: "BidiAgent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/repository\_session\_manager.py:376](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/repository_session_manager.py#L376)

Initialize a bidirectional agent with a session.

**Arguments**:

-   `agent` - BidiAgent to initialize from the session
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### append\_bidi\_message

```python
def append_bidi_message(message: Message, agent: "BidiAgent",
                        **kwargs: Any) -> None
```

Defined in: [src/strands/session/repository\_session\_manager.py:435](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/repository_session_manager.py#L435)

Append a message to the bidirectional agent’s session.

**Arguments**:

-   `message` - Message to add to the agent in the session
-   `agent` - BidiAgent to append the message to
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### sync\_bidi\_agent

```python
def sync_bidi_agent(agent: "BidiAgent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/repository\_session\_manager.py:454](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/repository_session_manager.py#L454)

Serialize and update the bidirectional agent into the session repository.

**Arguments**:

-   `agent` - BidiAgent to sync to the session.
-   `**kwargs` - Additional keyword arguments for future extensibility.