ModelRouter: a reusable, immutable set of candidate models with a routing strategy.

A router is a `Plugin`, so an agent accepts it through `model=`. Its `RoutingStrategy` makes every routing decision: the router asks for a candidate before the first model call, and again after a call fails without a hook claiming the retry, passing the attempts so far.

The router orchestrates only. It resolves a candidate to a concrete model, applies it to the call, gives each new candidate a fresh retry budget, and holds per-invocation state. It has no failover policy, so a strategy can change routing behavior without changing the router.

The strategy defaults to `FallbackStrategy`, which makes `ModelRouter([a, b])` ordered failover until a candidate starts failing repeatedly; see `fallback_strategy` for what it decides and when it departs from declaration order. `max_switches` caps switches per invocation.

What an answer does depends on whether a failed model call is pending. A candidate is resolved and applied. `None` declines: the opening choice then runs the router’s default model, the first declared candidate resolved without consulting any strategy, and a later decline ends routing so the model’s error surfaces. A strategy that raises propagates on the opening choice and ends routing after a failure, where the pending model error stays the one that surfaces. A candidate that will not resolve to a model propagates on the opening choice; after a failure it takes its slot in the round and the strategy is asked again, so one unusable candidate does not strand the healthy ones declared after it.

A failure round uses each candidate at most once, whether it was switched to or found unusable, so naming one the round already used ends routing; a success starts a new round on the candidate that succeeded, which counts as that round’s first use just as the opening choice does.

A nested `ModelRouter` contributes **one** candidate: it is asked with its own candidates and no attempts, and performs no internal failover, so when a nested pick fails the outer router moves off the whole nested candidate rather than advancing within it. A nested strategy that declines serves that router’s default model; one that raises makes that candidate unusable, which propagates on the opening choice and costs it its slot in the round after a failure.

Known limitation: a model that fails after streaming part of a response has already emitted those events, so a streaming consumer sees that partial output followed by the replacement’s full response. `AfterModelCallEvent` documents this for any hook-requested retry; routing reaches it more often because it advances on any failure the retry strategy declines, not only throttling.

Known limitation: routing applies to `InvokeModelStage`, so `agent.model` stays the first declared candidate and subsystems reading it reason about that model rather than the one running. Proactive compression sizes against `agent.model`’s context window, so routing among candidates with materially different windows can under-compress and overflow the routed model; the agent span reports the first candidate’s model id; and `Agent.structured_output()` calls the model directly, bypassing routing entirely. Prefer candidates with comparable context windows and tokenizers until the selected model is threaded to those consumers.

## RoutingCandidate

```python
@dataclass(frozen=True)
class RoutingCandidate()
```

Defined in: [src/strands/models/routing/router.py:78](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/router.py#L78)

A model or model group with optional classifier-facing evidence.

`model` may be a nested `ModelRouter`, which contributes one opaque candidate. Its strategy selects from its own candidates, and the group performs no internal failover. Metadata must be a JSON-serializable mapping; classifier-based strategies may send it across provider boundaries, so it must not contain secrets. The candidate stores the caller’s mapping without copying, so it must not be mutated after construction.

**Raises**:

-   `TypeError` - If metadata is not a mapping with string keys.
-   `ValueError` - If metadata is not JSON-serializable.

#### \_\_post\_init\_\_

```python
def __post_init__() -> None
```

Defined in: [src/strands/models/routing/router.py:96](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/router.py#L96)

Validate caller-owned metadata.

## ModelRouter

```python
class ModelRouter(Plugin)
```

Defined in: [src/strands/models/routing/router.py:130](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/router.py#L130)

A reusable set of candidate models routed in strategy-defined preference order.

#### \_\_init\_\_

```python
def __init__(models: Sequence[CandidateInput],
             *,
             strategy: RoutingStrategy | None = None,
             max_switches: int | None = None) -> None
```

Defined in: [src/strands/models/routing/router.py:135](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/router.py#L135)

Initialize the router.

**Arguments**:

-   `models` - The models to route among, as a sequence. Each is a `Model`, a nested `ModelRouter`, or a `RoutingCandidate` wrapping one with a name, description, and metadata. The first is the router’s default, used when a strategy declines, and each is normalized into the `RoutingCandidate` a strategy chooses from.
-   `strategy` - Chooses the candidate for each model call, and is asked again after a failed call. Defaults to `FallbackStrategy`, which prefers the candidate with the fewest recorded failures and breaks ties by declaration order, so an invocation with no failures behind it is ordered failover. A success re-arms every candidate.
-   `max_switches` - Cap on model switches within one invocation, after which the router stops asking and lets the error surface. Selection is asked once per invocation, but every failed model call can switch, so an invocation running a long tool loop has many chances to switch. Defaults to `None`, leaving the stop decision to the strategy.

**Raises**:

-   `TypeError` - If `models` is not a sequence, a candidate is not a `Model` or `ModelRouter`, or `strategy` does not implement `RoutingStrategy`.
-   `ValueError` - If `models` is empty, candidate names collide, a model is routed to more than
-   `once` - including through a nested router — any candidate is a stateful model, or `max_switches` is negative.

#### candidates

```python
@property
def candidates() -> tuple[RoutingCandidate, ...]
```

Defined in: [src/strands/models/routing/router.py:182](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/router.py#L182)

The normalized candidates, in declaration order.

#### default\_model

```python
@property
def default_model() -> Model
```

Defined in: [src/strands/models/routing/router.py:187](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/router.py#L187)

The first declared candidate resolved to a concrete model, without consulting a strategy.

#### init\_agent

```python
def init_agent(agent: Agent) -> None
```

Defined in: [src/strands/models/routing/router.py:196](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/routing/router.py#L196)

Register routing middleware and hooks; reject attachment through `plugins=[...]`.

**Arguments**:

-   `agent` - The agent the router is attached to.

**Raises**:

-   `ValueError` - If the router was not attached through `Agent(model=...)`.

#### CandidateInput

What `ModelRouter(models=...)` accepts for each entry.