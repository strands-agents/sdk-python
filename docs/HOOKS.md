# Hooks System

The hooks system enables extensible agent functionality through strongly-typed event callbacks.

## Terminology

- **Paired events**: Events that denote the beginning and end of an operation
- **Hook callback**: A function that receives a strongly-typed event argument
- **Hook provider**: An object implementing `HookProvider` that registers callbacks via `register_hooks()`

## Naming Conventions

- All hook events have a suffix of `Event`
- Paired events follow `Before{Action}Event` and `After{Action}Event`
- Action words come after the lifecycle indicator (e.g., `BeforeToolCallEvent` not `BeforeToolEvent`)

## Paired Events

- For every `Before` event there is a corresponding `After` event, even if an exception occurs
- `After` events invoke callbacks in reverse registration order (for proper cleanup)

## Writable Properties

Some events have writable properties that modify agent behavior. Values are re-read after callbacks complete. For example, `BeforeToolCallEvent.selected_tool` is writable - after invoking the callback, the modified `selected_tool` takes effect for the tool call.

## Input Guardrails with BeforeInvocationEvent

The `BeforeInvocationEvent` provides access to input messages through its `messages` attribute, enabling hooks to implement input-side guardrails that run before messages are added to the agent's conversation history.

### Use Cases

- **PII Detection/Redaction**: Scan and redact sensitive information before processing
- **Content Moderation**: Filter toxic or inappropriate content
- **Prompt Attack Prevention**: Detect and block malicious prompt injection attempts

### Example: Input Redaction Hook

```python
from strands import Agent
from strands.hooks import BeforeInvocationEvent, HookProvider, HookRegistry

class InputGuardrailHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self.check_input)
    
    async def check_input(self, event: BeforeInvocationEvent) -> None:
        if event.messages is None:
            return
        
        for message in event.messages:
            if message.get("role") == "user":
                content = message.get("content", [])
                for block in content:
                    if "text" in block:
                        # Option 1: Redact in-place
                        block["text"] = redact_pii(block["text"])
                        
                        # Option 2: Abort invocation by raising an exception
                        # if contains_malicious_content(block["text"]):
                        #     raise ValueError("Malicious content detected")

agent = Agent(hooks=[InputGuardrailHook()])
agent("Process this message")  # Guardrail runs before message is added to memory
```

### Key Behaviors

- `messages` defaults to `None` for backward compatibility (e.g., when invoked from deprecated methods)
- `messages` is writable, allowing hooks to modify content in-place
- The `AfterInvocationEvent` is always triggered even if a hook raises an exception, maintaining the paired event guarantee
