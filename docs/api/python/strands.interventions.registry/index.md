Bridges InterventionHandler instances to the Strands hook system.

Registers one hook callback per lifecycle event type, dispatches to all handlers that override that method in registration order, with short-circuiting on Deny (and denied Confirms) and accumulation for Guide.

## InterventionRegistry

```python
class InterventionRegistry()
```

Defined in: [src/strands/interventions/registry.py:28](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interventions/registry.py#L28)

Bridges InterventionHandler instances and the Strands hook system.

Registers one hook callback per lifecycle event type, dispatches to all handlers that override that method in registration order.

#### \_\_init\_\_

```python
def __init__(handlers: list[InterventionHandler],
             hook_registry: HookRegistry) -> None
```

Defined in: [src/strands/interventions/registry.py:35](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interventions/registry.py#L35)

Initialize the registry and wire handlers into the hook system.

**Arguments**:

-   `handlers` - Intervention handlers in evaluation order.
-   `hook_registry` - The agent’s hook registry to attach callbacks to.

**Raises**:

-   `ValueError` - If two handlers share the same name.

#### handlers

```python
@property
def handlers() -> list[InterventionHandler]
```

Defined in: [src/strands/interventions/registry.py:55](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/interventions/registry.py#L55)

Registered handlers in registration order.