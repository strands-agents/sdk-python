"""Model routing primitives.

``ModelRouter`` holds an ordered set of candidate models and resolves a concrete default. The
API is provisional and may change before it is finalized.
"""

from .router import CandidateInput, ModelRouter

__all__ = [
    "CandidateInput",
    "ModelRouter",
]
