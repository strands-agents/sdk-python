"""Multiagent capabilities for Strands Agents.

This module provides support for multiagent systems, including agent-to-agent (A2A)
communication protocols and coordination mechanisms.

Submodules:
    a2a: Implementation of the Agent-to-Agent (A2A) protocol, which enables
         standardized communication between agents.
"""

from .base import MultiAgentBase, MultiAgentResult
from .graph import GraphBuilder, GraphResult
from .swarm import Swarm, SwarmResult

__all__ = [
    "GraphBuilder",
    "GraphResult",
    "MultiAgentBase",
    "MultiAgentResult",
    "Swarm",
    "SwarmResult",
]
