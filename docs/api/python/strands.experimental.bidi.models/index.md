Bidirectional model interfaces and implementations.

#### \_\_getattr\_\_

```python
def __getattr__(name: str) -> Any
```

Defined in: [src/strands/experimental/bidi/models/**init**.py:18](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/__init__.py#L18)

Lazy load bidi model implementations only when accessed.

This defers the import of optional dependencies until actually needed.