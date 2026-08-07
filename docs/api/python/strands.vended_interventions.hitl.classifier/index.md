Built-in LLM risk classifier for the HumanInTheLoop handler.

Uses an inner agent with structured output to evaluate whether a tool call requires human approval based on risk criteria.

## ClassifierResult

```python
@dataclass
class ClassifierResult()
```

Defined in: [src/strands/vended\_interventions/hitl/classifier.py:21](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_interventions/hitl/classifier.py#L21)

Result from a classifier evaluation.

## HumanInTheLoopClassifier

```python
@runtime_checkable
class HumanInTheLoopClassifier(Protocol)
```

Defined in: [src/strands/vended\_interventions/hitl/classifier.py:29](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_interventions/hitl/classifier.py#L29)

Callable (sync or async) that decides whether a tool call requires human approval.

#### \_\_call\_\_

```python
def __call__(event: BeforeToolCallEvent,
             **kwargs: Any) -> ClassifierResult | Awaitable[ClassifierResult]
```

Defined in: [src/strands/vended\_interventions/hitl/classifier.py:32](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_interventions/hitl/classifier.py#L32)

Evaluate whether a tool call requires human approval.

**Arguments**:

-   `event` - The tool call event under evaluation.
-   `**kwargs` - Additional keyword arguments for future extensibility.

**Returns**:

ClassifierResult indicating whether approval is required.

## LLMClassifierConfig

```python
@dataclass
class LLMClassifierConfig()
```

Defined in: [src/strands/vended\_interventions/hitl/classifier.py:46](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_interventions/hitl/classifier.py#L46)

Configuration for the built-in LLM risk classifier.

**Arguments**:

-   `system_prompt` - Risk criteria prompt. Defaults to a general-purpose risk prompt.
-   `model` - Model for risk evaluation. Defaults to the parent agent’s model.