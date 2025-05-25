# PR Title: Implement dynamic system prompt override functionality

## Summary

This PR implements dynamic system prompt override functionality for the Agent class, resolving issue #103. The implementation allows users to override the system prompt on a per-call basis without modifying the agent's default configuration.

## Problem

Previously, the `Agent._execute_event_loop_cycle()` method was discarding kwargs parameters (including `system_prompt`) by using `.pop()` without storing the values. This made it impossible to dynamically override the system prompt when calling the agent, limiting flexibility for use cases requiring different behaviors per call.

## Solution

Modified the `_execute_event_loop_cycle()` method to:

1. **Extract parameters with fallbacks** instead of discarding them:
   ```python
   # Before: kwargs.pop("system_prompt", None)  # Discarded!
   # After: system_prompt = kwargs.pop("system_prompt", self.system_prompt)
   ```

2. **Use extracted variables** in the event loop cycle call:
   ```python
   # Before: system_prompt=self.system_prompt,  # Always instance value
   # After: system_prompt=system_prompt,        # Uses override or fallback
   ```

3. **Apply same pattern** to all other parameters (model, tool_execution_handler, etc.)

## Usage

```python
# Create agent with default system prompt
agent = Agent(system_prompt="You are a helpful assistant.", model=model)

# Normal call (uses default)
agent("Hello")  # Uses: "You are a helpful assistant."

# Override system prompt for this call only
agent("Hello", system_prompt="You are a pirate.")  # Uses: "You are a pirate."

# Next call reverts to default
agent("Hello")  # Uses: "You are a helpful assistant."
```

## Use Cases Enabled

- **Multi-purpose agents**: Single agent, different behaviors per call
- **Dynamic conversation management**: Changing system behavior mid-conversation  
- **A/B testing**: Testing different system prompts with the same agent
- **Context-specific behavior**: Adapting agent behavior based on user context
- **Development & debugging**: Easy testing of different system prompts

## Changes Made

### Core Implementation
- **File**: `src/strands/agent/agent.py`
- **Method**: `_execute_event_loop_cycle()` (lines ~460-490)
- **Change**: Parameter extraction with fallbacks instead of discarding

### Tests Added
- **File**: `tests/strands/agent/test_system_prompt_override.py`
- **Coverage**: 8 comprehensive test cases covering:
  - Default system prompt usage
  - System prompt override functionality
  - Reversion to default after override
  - Multiple different overrides
  - Edge cases (None, empty string)
  - Agents without default system prompts

## Testing

All tests pass and verify:
- ✅ Override functionality works correctly
- ✅ Default behavior is preserved
- ✅ Edge cases are handled properly
- ✅ Backward compatibility is maintained

## Backward Compatibility

This change is **100% backward compatible**:
- Existing code continues to work unchanged
- No breaking changes to the API
- Default behavior is preserved
- Performance impact is negligible

## Edge Cases Handled

- `None` system prompt override
- Empty string system prompt override
- Agents without default system prompts
- Special characters and unicode in system prompts
- Very long system prompts

## Related

- Closes #103
- Addresses the core issue described in the GitHub issue
- Enables the requested dynamic system prompt functionality

---

**Review Focus Areas:**
1. Parameter extraction logic in `_execute_event_loop_cycle()`
2. Test coverage for various scenarios
3. Backward compatibility verification
4. Edge case handling

**Testing Instructions:**
1. Run the new tests: `python -m pytest tests/strands/agent/test_system_prompt_override.py -v`
2. Verify existing tests still pass
3. Test the usage examples in the description
