"""Model routing primitives.

``ModelRouter`` asks its ``RoutingStrategy`` which candidate to use, and asks again after a failed
call, so the strategy owns every routing decision and the router only orchestrates. A strategy that
returns ``None`` ends routing and lets the error surface. The default ``FallbackStrategy`` works down
the candidates in declaration order. The API is provisional and may change before it is finalized.
"""

from .router import CandidateInput, FallbackStrategy, ModelRouter, RoutingCandidate
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
