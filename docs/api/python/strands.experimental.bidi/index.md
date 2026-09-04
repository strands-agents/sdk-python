Bidirectional streaming package.

#### \_\_getattr\_\_

```python
def __getattr__(name: str) -> Any
```

Defined in: [src/strands/experimental/bidi/**init**.py:88](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/__init__.py#L88)

Lazy load IO implementations only when accessed.

This defers the import of optional dependencies until actually needed.