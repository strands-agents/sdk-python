Human-in-the-loop interrupt system for agent workflows.

## Interrupt

```python
@dataclass
class Interrupt()
```

Defined in: [src/strands/interrupt.py:15](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L15)

Represents an interrupt that can pause agent execution for human-in-the-loop workflows.

**Attributes**:

-   `id` - Unique identifier.
-   `name` - User defined name.
-   `reason` - User provided reason for raising the interrupt.
-   `response` - Human response provided when resuming the agent after an interrupt.

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Defined in: [src/strands/interrupt.py:30](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L30)

Serialize to dict for session management.

## InterruptException

```python
class InterruptException(Exception)
```

Defined in: [src/strands/interrupt.py:35](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L35)

Exception raised when human input is required.

#### \_\_init\_\_

```python
def __init__(interrupt: Interrupt) -> None
```

Defined in: [src/strands/interrupt.py:38](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L38)

Set the interrupt.

## \_InterruptState

```python
@dataclass
class _InterruptState()
```

Defined in: [src/strands/interrupt.py:44](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L44)

Track the state of interrupt events raised by the user.

Note, unanswered interrupts are cleared after resuming; an answered invocation-scoped response is retained for the rest of its interrupt cycle.

**Attributes**:

-   `interrupts` - Interrupts raised by the user. May be non-empty even when `activated` is False because retained responses persist until their cycle ends.
-   `context` - Additional context associated with an interrupt event.
-   `activated` - True if agent is in an interrupt state, False otherwise.

#### has\_pending\_tool\_execution

```python
@property
def has_pending_tool_execution() -> bool
```

Defined in: [src/strands/interrupt.py:63](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L63)

Whether a tool execution is pending resume.

#### activate

```python
def activate() -> None
```

Defined in: [src/strands/interrupt.py:67](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L67)

Activate the interrupt state.

#### deactivate

```python
def deactivate() -> None
```

Defined in: [src/strands/interrupt.py:72](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L72)

Deacitvate the interrupt state.

Interrupts and context are cleared.

#### end\_tool\_cycle

```python
def end_tool_cycle() -> None
```

Defined in: [src/strands/interrupt.py:82](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L82)

Clear a completed tool cycle’s state, keeping answered invocation-scoped responses.

#### end\_interrupt\_cycle

```python
def end_interrupt_cycle() -> None
```

Defined in: [src/strands/interrupt.py:93](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L93)

Release invocation-scoped interrupts once their interrupt cycle is over.

#### resume

```python
def resume(prompt: "AgentInput") -> None
```

Defined in: [src/strands/interrupt.py:106](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L106)

Configure the interrupt state if resuming from an interrupt event.

**Arguments**:

-   `prompt` - User responses if resuming from interrupt.

**Raises**:

-   `TypeError` - If in interrupt state but user did not provide responses.

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Defined in: [src/strands/interrupt.py:155](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L155)

Serialize to dict for session management.

Exclude deactivated invocation-scoped responses — persisting them would give a restored agent a standing approval.

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "_InterruptState"
```

Defined in: [src/strands/interrupt.py:176](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interrupt.py#L176)

Initiailize interrupt state from serialized interrupt state.

Interrupt state can be serialized with the `to_dict` method.