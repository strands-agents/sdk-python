"""Multiagent capabilities for Strands Agents.

This module provides support for multiagent systems, including agent-to-agent (A2A)
communication protocols and coordination mechanisms.

Submodules:
    a2a: Implementation of the Agent-to-Agent (A2A) protocol, which enables
         standardized communication between agents.
"""

from . import a2a

__all__ = ["a2a"]
