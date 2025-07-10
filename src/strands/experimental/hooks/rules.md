# Hook System Rules

## Terminology

- **Paired events**: Events that denote the beginning and end of an operation
- **Hook callback**: A function that receives a strongly-typed event argument and performs some action in response

## Naming Conventions

- All hook events have a suffix of `Event`
- Paired events follow the naming convention of `Before{Item}Event` and `After{Item}Event`

## Paired Events

- The final event in a pair returns `True` for `should_reverse_callbacks`
- For every `Before` event there is a corresponding `After` event, even if an exception occurs

## Writable Properties

For events with writable properties, those values are re-read after invoking the hook callbacks and used in subsequent processing. For example, `BeforeToolInvocationEvent.selected_tool` is writable - after invoking the callback for `BeforeToolInvocationEvent`, the `selected_tool` takes effect for the tool call.