File-based session manager for local filesystem storage.

## FileSessionManager

```python
class FileSessionManager(RepositorySessionManager, SessionRepository)
```

Defined in: [src/strands/session/file\_session\_manager.py:28](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L28)

File-based session manager for local filesystem storage.

Creates the following filesystem structure for the session storage:

```bash
/<sessions_dir>/
└── session_<session_id>/
    ├── session.json                # Session metadata
    └── agents/
        └── agent_<agent_id>/
            ├── agent.json          # Agent metadata
            └── messages/
                ├── message_<id1>.json
                └── message_<id2>.json
```

#### \_\_init\_\_

```python
def __init__(session_id: str, storage_dir: str | None = None, **kwargs: Any)
```

Defined in: [src/strands/session/file\_session\_manager.py:45](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L45)

Initialize FileSession with filesystem storage.

**Arguments**:

-   `session_id` - ID for the session. ID is not allowed to contain path separators (e.g., a/b).
-   `storage_dir` - Directory for local filesystem storage. Defaults to a user-private `~/.strands/sessions/` directory.
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### create\_session

```python
def create_session(session: Session, **kwargs: Any) -> Session
```

Defined in: [src/strands/session/file\_session\_manager.py:164](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L164)

Create a new session.

#### read\_session

```python
def read_session(session_id: str, **kwargs: Any) -> Session | None
```

Defined in: [src/strands/session/file\_session\_manager.py:182](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L182)

Read session data.

#### delete\_session

```python
def delete_session(session_id: str, **kwargs: Any) -> None
```

Defined in: [src/strands/session/file\_session\_manager.py:191](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L191)

Delete session and all associated data.

#### create\_agent

```python
def create_agent(session_id: str, session_agent: SessionAgent,
                 **kwargs: Any) -> None
```

Defined in: [src/strands/session/file\_session\_manager.py:199](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L199)

Create a new agent in the session.

#### read\_agent

```python
def read_agent(session_id: str, agent_id: str,
               **kwargs: Any) -> SessionAgent | None
```

Defined in: [src/strands/session/file\_session\_manager.py:211](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L211)

Read agent data.

#### update\_agent

```python
def update_agent(session_id: str, session_agent: SessionAgent,
                 **kwargs: Any) -> None
```

Defined in: [src/strands/session/file\_session\_manager.py:220](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L220)

Update agent data.

#### create\_message

```python
def create_message(session_id: str, agent_id: str,
                   session_message: SessionMessage, **kwargs: Any) -> None
```

Defined in: [src/strands/session/file\_session\_manager.py:231](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L231)

Create a new message for the agent.

#### read\_message

```python
def read_message(session_id: str, agent_id: str, message_id: int,
                 **kwargs: Any) -> SessionMessage | None
```

Defined in: [src/strands/session/file\_session\_manager.py:241](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L241)

Read message data.

#### update\_message

```python
def update_message(session_id: str, agent_id: str,
                   session_message: SessionMessage, **kwargs: Any) -> None
```

Defined in: [src/strands/session/file\_session\_manager.py:249](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L249)

Update message data.

#### list\_messages

```python
def list_messages(session_id: str,
                  agent_id: str,
                  limit: int | None = None,
                  offset: int = 0,
                  **kwargs: Any) -> list[SessionMessage]
```

Defined in: [src/strands/session/file\_session\_manager.py:261](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L261)

List messages for an agent with pagination.

#### create\_multi\_agent

```python
def create_multi_agent(session_id: str, multi_agent: "MultiAgentBase",
                       **kwargs: Any) -> None
```

Defined in: [src/strands/session/file\_session\_manager.py:301](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L301)

Create a new multiagent state in the session.

#### read\_multi\_agent

```python
def read_multi_agent(session_id: str, multi_agent_id: str,
                     **kwargs: Any) -> dict[str, Any] | None
```

Defined in: [src/strands/session/file\_session\_manager.py:311](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L311)

Read multi-agent state from filesystem.

#### update\_multi\_agent

```python
def update_multi_agent(session_id: str, multi_agent: "MultiAgentBase",
                       **kwargs: Any) -> None
```

Defined in: [src/strands/session/file\_session\_manager.py:318](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/file_session_manager.py#L318)

Update multi-agent state from filesystem.