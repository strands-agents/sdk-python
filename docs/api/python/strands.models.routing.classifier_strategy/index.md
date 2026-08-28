Route among configured candidates using model classification.

## ClassifierStrategy

```python
class ClassifierStrategy()
```

Defined in: [src/strands/models/routing/classifier\_strategy.py:102](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/classifier_strategy.py#L102)

Choose a candidate by applying a configurable policy with a classifier model.

Classification adds one call to the explicitly configured model. Candidate declaration order does not inform classification. Candidate names, descriptions, metadata, the latest request, and textual parent-agent instructions may cross the classifier provider boundary and must not contain secrets. Structured parent-system-prompt blocks such as cache points are omitted because the classifier receives rebuilt, bounded context rather than the original prompt.

Classification failures warn and decline selection, so `ModelRouter` serves candidate zero. If the selected candidate later fails, this strategy declines further selection and lets the original model error surface without switching. Nested routers are treated as opaque candidates using only their wrapper evidence.

#### \_\_init\_\_

```python
def __init__(
        model: Model,
        *,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        timeout: float = 30.0,
        max_message_chars: int = _DEFAULT_MESSAGE_CHARACTER_LIMIT,
        max_agent_instructions_chars:
    int = _DEFAULT_AGENT_INSTRUCTIONS_CHARACTER_LIMIT,
        max_candidate_chars: int = _DEFAULT_CANDIDATE_CHARACTER_LIMIT) -> None
```

Defined in: [src/strands/models/routing/classifier\_strategy.py:116](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/classifier_strategy.py#L116)

Initialize the strategy.

**Arguments**:

-   `model` - Model used for classification. It must support structured output.
-   `system_prompt` - Routing policy for the classifier, sent verbatim and never truncated. The SDK appends mandatory isolation, candidate-index, and structured-output rules that the policy cannot override. Defaults to the SDK input-complexity policy.
-   `timeout` - Maximum seconds to wait for classification.
-   `max_message_chars` - Maximum characters copied from the latest request into the classifier’s user message.
-   `max_agent_instructions_chars` - Maximum characters copied from the parent agent’s system prompt text into the untrusted classification context.
-   `max_candidate_chars` - Maximum aggregate characters for the serialized evidence (names, descriptions, and metadata) of all candidates. Evidence is never truncated; selection raises `ValueError` when the budget is exceeded.

**Raises**:

-   `TypeError` - If an argument has the wrong type.
-   `ValueError` - If `timeout` is not finite and greater than zero or a character limit is not positive.

#### select

```python
async def select(context: RoutingContext,
                 **kwargs: Any) -> RoutingCandidate | None
```

Defined in: [src/strands/models/routing/classifier\_strategy.py:165](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/classifier_strategy.py#L165)

Select one opening candidate, declining on classification or serving-time failure.

**Raises**:

-   `ValueError` - If the candidates’ serialized evidence exceeds `max_candidate_chars`. This misconfiguration is permanent, so it propagates instead of declining.