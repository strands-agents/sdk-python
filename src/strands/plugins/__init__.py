"""Plugin system for Strands Agents SDK.

This package exposes the :class:`Plugin` base class that allows tool and hook
methods to be bundled together and registered with an agent in one step.

See :mod:`strands.plugins.plugin` for full documentation and examples.
"""

from .plugin import Plugin

__all__ = [
    "Plugin",
]
