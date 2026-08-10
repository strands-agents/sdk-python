"""Model routing primitives.

``ModelRouter`` asks its ``RoutingStrategy`` which candidate to use, and asks again after a failed
call, so the strategy owns every routing decision and the router only orchestrates. Declining with
``None`` after a failure ends routing and lets the error surface; declining the opening choice still
serves the request on the first declared candidate it has not already tried. The default
``FallbackStrategy`` prefers the candidate with the fewest recorded failures, breaking ties by
declaration order, and re-arms a candidate once a later call succeeds. The API is provisional and may
change before it is finalized.
"""

from .fallback_strategy import FallbackStrategy
from .router import CandidateInput, ModelRouter, RoutingCandidate
from .strategy import RoutingAttempt, RoutingContext, RoutingStrategy

__all__ = [
    "CandidateInput",
    "FallbackStrategy",
    "ModelRouter",
    "RoutingAttempt",
    "RoutingCandidate",
    "RoutingContext",
    "RoutingStrategy",
]
