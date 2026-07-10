Data models for session management.

## SessionType

```python
class SessionType(str, Enum)
```

Defined in: [src/strands/types/session.py:18](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L18)

Enumeration of session types.

As sessions are expanded to support new use cases like multi-agent patterns, new types will be added here.

#### encode\_bytes\_values

```python
def encode_bytes_values(obj: Any) -> Any
```

Defined in: [src/strands/types/session.py:28](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L28)

Recursively encode any bytes values in an object to base64.

Handles dictionaries, lists, and nested structures.

#### decode\_bytes\_values

```python
def decode_bytes_values(obj: Any) -> Any
```

Defined in: [src/strands/types/session.py:43](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L43)

Recursively decode any base64-encoded bytes values in an object.

Handles dictionaries, lists, and nested structures.

## SessionMessage

```python
@dataclass
class SessionMessage()
```

Defined in: [src/strands/types/session.py:59](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L59)

Message within a SessionAgent.

**Attributes**:

-   `message` - Message content
-   `message_id` - Index of the message in the conversation history
-   `redact_message` - If the original message is redacted, this is the new content to use
-   `created_at` - ISO format timestamp for when this message was created
-   `updated_at` - ISO format timestamp for when this message was last updated

#### from\_message

```python
@classmethod
def from_message(cls, message: Message, index: int) -> "SessionMessage"
```

Defined in: [src/strands/types/session.py:77](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L77)

Convert from a Message, base64 encoding bytes values.

#### to\_message

```python
def to_message() -> Message
```

Defined in: [src/strands/types/session.py:86](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L86)

Convert SessionMessage back to a Message, decoding any bytes values.

If the message was redacted, return the redact content instead.

#### from\_dict

```python
@classmethod
def from_dict(cls, env: dict[str, Any]) -> "SessionMessage"
```

Defined in: [src/strands/types/session.py:97](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L97)

Initialize a SessionMessage from a dictionary, ignoring keys that are not class parameters.

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Defined in: [src/strands/types/session.py:102](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L102)

Convert the SessionMessage to a dictionary representation.

## SessionAgent

```python
@dataclass
class SessionAgent()
```

Defined in: [src/strands/types/session.py:108](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L108)

Agent that belongs to a Session.

**Attributes**:

-   `agent_id` - Unique id for the agent.
-   `state` - User managed state.
-   `conversation_manager_state` - State for conversation management.
-   `created_at` - Created at time.
-   `updated_at` - Updated at time.

#### from\_agent

```python
@classmethod
def from_agent(cls, agent: "Agent") -> "SessionAgent"
```

Defined in: [src/strands/types/session.py:127](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L127)

Convert an Agent to a SessionAgent.

#### from\_bidi\_agent

```python
@classmethod
def from_bidi_agent(cls, agent: "BidiAgent") -> "SessionAgent"
```

Defined in: [src/strands/types/session.py:142](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L142)

Convert a BidiAgent to a SessionAgent.

**Arguments**:

-   `agent` - BidiAgent to convert

**Returns**:

SessionAgent with empty conversation\_manager\_state (BidiAgent doesn’t use conversation manager)

#### from\_dict

```python
@classmethod
def from_dict(cls, env: dict[str, Any]) -> "SessionAgent"
```

Defined in: [src/strands/types/session.py:167](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L167)

Initialize a SessionAgent from a dictionary, ignoring keys that are not class parameters.

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Defined in: [src/strands/types/session.py:172](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L172)

Convert the SessionAgent to a dictionary representation.

#### initialize\_internal\_state

```python
def initialize_internal_state(agent: "Agent") -> None
```

Defined in: [src/strands/types/session.py:176](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L176)

Initialize internal state of agent.

#### initialize\_bidi\_internal\_state

```python
def initialize_bidi_internal_state(agent: "BidiAgent") -> None
```

Defined in: [src/strands/types/session.py:183](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L183)

Initialize internal state of BidiAgent.

**Arguments**:

-   `agent` - BidiAgent to initialize internal state for

## Session

```python
@dataclass
class Session()
```

Defined in: [src/strands/types/session.py:196](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L196)

Session data model.

#### from\_dict

```python
@classmethod
def from_dict(cls, env: dict[str, Any]) -> "Session"
```

Defined in: [src/strands/types/session.py:205](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L205)

Initialize a Session from a dictionary, ignoring keys that are not class parameters.

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Defined in: [src/strands/types/session.py:209](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/session.py#L209)

Convert the Session to a dictionary representation.