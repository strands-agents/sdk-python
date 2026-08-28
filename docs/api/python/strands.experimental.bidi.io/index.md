IO channel implementations for bidirectional streaming.

#### \_\_getattr\_\_

```python
def __getattr__(name: str) -> Any
```

Defined in: [src/strands/experimental/bidi/io/**init**.py:10](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/__init__.py#L10)

Lazy load the audio IO implementation only when accessed.