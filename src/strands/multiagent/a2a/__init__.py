"""Agent-to-Agent (A2A) communication protocol implementation for Strands Agents.

This module provides classes and utilities for enabling Strands Agents to communicate
with other agents using the Agent-to-Agent (A2A) protocol.

Docs: https://google-a2a.github.io/A2A/latest/

Classes:
    A2AServer: A wrapper that adapts a Strands Agent to be an A2A server.
    A2AClient: A client for communicating with remote A2A agents.
    StrandsA2AExecutor: The executor that handles A2A requests for Strands Agents.
"""

from .client import A2AClient, A2AError, build_agentcore_url, extract_region_from_arn
from .executor import StrandsA2AExecutor
from .server import A2AServer

__all__ = [
    "A2AServer",
    "A2AClient",
    "A2AError",
    "StrandsA2AExecutor",
    "build_agentcore_url",
    "extract_region_from_arn",
]
