An agent treats its message history as trusted input. When you build that history from a source you do not control (a request body, a loaded snapshot) treat it as untrusted, because message content carries more than text: it can carry tool-call and tool-result blocks.

Forged tool content is a concern in both SDKs. A tool-result block you did not produce, placed in history that reaches the model, can misrepresent what a tool returned and steer the model’s next step.

In the Python SDK, invoking an agent with content other than a string is a pointed example of this: the input is considered trusted, and a tool-call block as the most recent message causes the agent to run that tool directly on its next invocation, with no model call in between. The block’s author chooses the tool and its arguments outright.

## Do not populate history from an untrusted source

The reliable control is the trust boundary. Build an agent’s message history from your own application, not from input a caller can shape. History your application produces, or persists to storage only it can write, is trusted; load it as-is.

## Clearing a trailing tool-call block

If a Python application must accept caller-influenced history, strip `toolUse` content from the tail before the agent uses it. The dispatch check inspects only the last message, so keep stripping until the final message carries no `toolUse` block: dropping one `toolUse`\-only message can expose another beneath it.

```python
from strands import Agent


def strip_trailing_tool_use(messages):
    """Strip toolUse blocks from the tail until the last message has none."""
    messages = list(messages)
    while messages:
        last = messages[-1]
        content = [block for block in last.get("content", []) if "toolUse" not in block]
        if len(content) == len(last.get("content", [])):
            break  # no toolUse in the final message
        if content:
            messages[-1] = {**last, "content": content}
            break
        messages.pop()  # message was toolUse-only; drop it and re-check the tail
    return messages


# untrusted_messages came from a request body, queue, or shared store.
agent = Agent()
agent(strip_trailing_tool_use(untrusted_messages))
```

Stripping the trailing tool-call content closes the direct-dispatch path. It does not make untrusted content safe: injected text and forged tool-result content elsewhere in the history still reach the model. Prefer the trust boundary above.

## Related

-   [Prompts](/docs/user-guide/concepts/agents/prompts/index.md) for the input shapes an agent accepts.
-   [Snapshots](/docs/user-guide/concepts/agents/snapshots/index.md) for capturing and loading agent state.

## Related pages

- [Prompt Engineering](/docs/user-guide/safety-security/prompt-engineering/index.md) (2 shared tags)
- [Prompts](/docs/user-guide/concepts/agents/prompts/index.md) (2 shared tags)
- [Coherence Evaluator](/docs/user-guide/evals-sdk/evaluators/coherence_evaluator/index.md) (1 shared tag)
- [Conciseness Evaluator](/docs/user-guide/evals-sdk/evaluators/conciseness_evaluator/index.md) (1 shared tag)
- [Goal Success Rate Evaluator](/docs/user-guide/evals-sdk/evaluators/goal_success_rate_evaluator/index.md) (1 shared tag)
- [Helpfulness Evaluator](/docs/user-guide/evals-sdk/evaluators/helpfulness_evaluator/index.md) (1 shared tag)
- [Interactions Evaluator](/docs/user-guide/evals-sdk/evaluators/interactions_evaluator/index.md) (1 shared tag)
- [Output Evaluator](/docs/user-guide/evals-sdk/evaluators/output_evaluator/index.md) (1 shared tag)
- [Attack Strategies](/docs/user-guide/evals-sdk/red-teaming/strategies/index.md) (1 shared tag)
- [Harmfulness Evaluator](/docs/user-guide/evals-sdk/evaluators/harmfulness_evaluator/index.md) (1 shared tag)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/event_loop/event_loop.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/event_loop/event_loop.py)
- [harness-sdk/strands-py/src/strands/agent/agent.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/agent/agent.py)

### TypeScript

- [harness-sdk/strands-ts/src/agent/agent.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/agent/agent.ts)
