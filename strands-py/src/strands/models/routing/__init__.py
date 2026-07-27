"""Model routing primitives.

``ModelRouter`` holds an ordered set of candidate models and, per model call, selects one via a
``RoutingStrategy`` (default: ``FallbackStrategy``). When the selected model's retries are
exhausted, the router advances to the next candidate in declaration order. ``ContextFitStrategy``
selects by context-window capacity. The API is provisional and may change before it is finalized.
"""

from .router import CandidateInput, FallbackStrategy, ModelRouter, RoutingCandidate
from .strategies import ContextFitStrategy
from .strategy import RoutingContext, RoutingStrategy

__all__ = [
    "CandidateInput",
    "ContextFitStrategy",
    "FallbackStrategy",
    "ModelRouter",
    "RoutingCandidate",
    "RoutingContext",
    "RoutingStrategy",
]
