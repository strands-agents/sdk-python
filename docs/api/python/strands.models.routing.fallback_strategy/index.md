The default routing strategy: ordered failover that prefers the candidates failing least.

## FallbackStrategy

```python
class FallbackStrategy()
```

Defined in: [src/strands/models/routing/fallback\_strategy.py:13](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/fallback_strategy.py#L13)

Picks the healthiest candidate not yet tried since the last success.

Candidates already tried since the last success are excluded, then the fewest recorded failures wins, ties going to the earlier declaration. So an invocation with no failures behind it is plain declaration order, and a model that keeps failing sinks below healthier ones rather than being re-tried in its declared slot.

A success re-arms every candidate, since exclusion looks only at attempts since the last success, and clears the succeeding candidate’s own failure count. Every other candidate keeps its failure history, so a model that fails repeatedly stays below healthier ones across successes. Returns `None` once every candidate has been tried since the last success.

#### select

```python
async def select(context: RoutingContext,
                 **kwargs: Any) -> RoutingCandidate | None
```

Defined in: [src/strands/models/routing/fallback\_strategy.py:27](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/fallback_strategy.py#L27)

Return the least-failed candidate not yet tried since the last success, else `None`.