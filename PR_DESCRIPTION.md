## Description

This PR adds a `@hook` decorator that transforms Python functions into `HookProvider` implementations with automatic event type detection from type hints.

## Motivation

Defining hooks currently requires implementing the `HookProvider` protocol with a class, which is verbose for simple use cases:

```python
# Current approach - verbose
class LoggingHooks(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self.on_tool_call)
    
    def on_tool_call(self, event: BeforeToolCallEvent) -> None:
        print(f"Tool: {event.tool_use}")

agent = Agent(hooks=[LoggingHooks()])
```

The `@hook` decorator provides a simpler function-based approach that reduces boilerplate while maintaining full compatibility with the existing hooks system.

Resolves: #1483

## Public API Changes

New `@hook` decorator exported from `strands` and `strands.hooks`:

```python
# After - concise
from strands import Agent, hook
from strands.hooks import BeforeToolCallEvent

@hook
def log_tool_calls(event: BeforeToolCallEvent) -> None:
    print(f"Tool: {event.tool_use}")

agent = Agent(hooks=[log_tool_calls])
```

The decorator supports multiple usage patterns:

```python
# Type hint detection
@hook
def my_hook(event: BeforeToolCallEvent) -> None: ...

# Explicit event type
@hook(event=BeforeToolCallEvent)
def my_hook(event) -> None: ...

# Multiple events via parameter
@hook(events=[BeforeToolCallEvent, AfterToolCallEvent])
def my_hook(event) -> None: ...

# Multiple events via Union type
@hook
def my_hook(event: BeforeToolCallEvent | AfterToolCallEvent) -> None: ...

# Async hooks
@hook
async def my_hook(event: BeforeToolCallEvent) -> None: ...

# Class methods
class MyHooks:
    @hook
    def my_hook(self, event: BeforeToolCallEvent) -> None: ...
```

## Related Issues

Fixes #1483

## Documentation PR

No documentation changes required.

## Type of Change

New feature

## Testing

- Added comprehensive unit tests (35 test cases)
- Tests cover: basic usage, explicit events, multi-events, union types, async, class methods, error handling
- [x] I ran `hatch run prepare`

## Checklist
- [x] I have read the CONTRIBUTING document
- [x] I have added any necessary tests that prove my fix is effective or my feature works
- [x] I have updated the documentation accordingly
- [x] I have added an appropriate example to the documentation to outline the feature, or no new docs are needed
- [x] My changes generate no new warnings
- [x] Any dependent changes have been merged and published

----

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.
