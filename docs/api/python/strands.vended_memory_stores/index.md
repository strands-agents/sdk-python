Vended memory stores for Strands Agents.

Concrete :class:`~strands.memory.types.MemoryStore` backends shipped with the SDK. A store may be imported from here or from its subpackage, e.g. `from strands.vended_memory_stores import BedrockKnowledgeBaseStore`.

#### \_\_getattr\_\_

```python
def __getattr__(name: str) -> Any
```

Defined in: [src/strands/vended\_memory\_stores/**init**.py:16](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/__init__.py#L16)

Lazy load store implementations only when accessed.

This defers the import of optional dependencies until actually needed.