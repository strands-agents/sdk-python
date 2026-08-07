"""Model routing primitives.

``ModelRouter`` asks its ``RoutingStrategy`` which candidate to use, and asks again after a failed
call, so the strategy owns every routing decision and the router only orchestrates. A strategy that
returns ``None`` ends routing and lets the error surface. The default ``FallbackStrategy`` follows
declaration order, re-arming a candidate once a later call succeeds and trying repeatedly failing
candidates after healthier ones. The API is provisional and may change before it is finalized.

These symbols are re-exported from ``strands.models``.
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

# Module layout, following ``agent/conversation_manager``: ``strategy`` holds the contract,
# ``router`` the orchestration, and each concrete strategy its own module.
