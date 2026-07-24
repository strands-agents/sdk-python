Shared types and constants for the sleep tool.

#### DEFAULT\_MAX\_DURATION

Default upper bound on `duration` (seconds) accepted by :func:`make_sleep`.

#### sleep\_description

```python
def sleep_description(max_duration: float) -> str
```

Defined in: [src/strands/vended\_tools/sleep/types.py:7](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/sleep/types.py#L7)

Build the model-facing description with the configured max interpolated.

#### SLEEP\_DESCRIPTION

Description for the default sleep tool (60-second cap).