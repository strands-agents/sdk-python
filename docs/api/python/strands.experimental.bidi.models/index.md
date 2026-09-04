Bidirectional model interfaces and implementations.

#### \_\_getattr\_\_

```python
def __getattr__(name: str) -> Any
```

Defined in: [src/strands/experimental/bidi/models/**init**.py:14](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/__init__.py#L14)

Lazy load bidi model implementations only when accessed.

This defers the import of optional dependencies until actually needed.