"""Agent Composition — Declarative Multi-Agent Coordination.

This module provides declarative composition for multi-agent systems. Agents declare
what they need and what they produce via manifests. The composition engine resolves
dependencies, derives event routing, and manages agent lifecycle automatically.

Key components:
    CompositionRegistry: Local catalog of agents and their manifests.
    BindingResolver: Evaluates manifests against available resources.
    EventRouter: Derives event flow graph from manifest declarations.
    LifecycleManager: Manages dormant/active transitions reactively.

Example:
    >>> from strands.composition import CompositionRegistry, BindingResolver, EventRouter
    >>> registry = CompositionRegistry()
    >>> registry.register(my_agent)
    >>> resolver = BindingResolver(available_events=["DataEvent"], available_tools=["query"])
    >>> status = resolver.evaluate(my_agent.manifest)
    >>> print(status.gap_report())
"""

from strands.composition.lifecycle import AgentState, LifecycleManager, StateTransition
from strands.composition.registry import CompositionRegistry
from strands.composition.resolver import BindingResolver, BindingStatus, Gap
from strands.composition.router import EventRouter, Route

__all__ = [
    "AgentState",
    "BindingResolver",
    "BindingStatus",
    "CompositionRegistry",
    "EventRouter",
    "Gap",
    "LifecycleManager",
    "Route",
    "StateTransition",
]
