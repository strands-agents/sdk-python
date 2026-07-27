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

`AfterToolsEvent` fires after all results for a tool batch are assembled. Its `message` contains the aggregate user-role tool-result message, and callbacks run in reverse registration order. Set `end_turn` to `True` to end the agent loop with the default assistant message, or to a non-empty string to use that string as the final assistant message. The aggregate tool results and final assistant message are added to conversation history, and the result uses `stop_reason="end_turn"` without another model call.

The event also runs when a tool batch exits through an interrupt or exception so cleanup callbacks can observe the results collected so far. Those outcomes remain authoritative: setting `end_turn` does not replace an interrupt or exception.
