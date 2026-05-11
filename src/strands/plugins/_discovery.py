"""Shared utility for discovering decorated methods on plugin instances.

This module provides helper functions used by both Plugin and MultiAgentPlugin
to scan for @hook (and optionally @tool) decorated methods, and shared registry
utilities for plugin initialization and hook registration.
"""

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from .._async import run_async
from ..hooks.registry import HookCallback, HookRegistry
from ..tools.decorator import DecoratedFunctionTool

logger = logging.getLogger(__name__)


def discover_hooks(instance: object, plugin_name: str) -> list[HookCallback]:
    """Scan an instance's class hierarchy for @hook decorated methods.

    Walks the MRO in reverse so parent class hooks come first, but child
    overrides win (only the child's version is included).

    Args:
        instance: The plugin instance to scan.
        plugin_name: The plugin name (used for debug logging).

    Returns:
        List of bound hook callback methods in declaration order.
    """
    hooks: list[HookCallback] = []
    seen: set[str] = set()

    for cls in reversed(type(instance).__mro__):
        for attr_name in cls.__dict__:
            if attr_name in seen:
                continue
            seen.add(attr_name)

            try:
                bound = getattr(instance, attr_name)
            except Exception:
                continue

            if hasattr(bound, "_hook_event_types") and callable(bound):
                hooks.append(bound)
                logger.debug("plugin=<%s>, hook=<%s> | discovered hook method", plugin_name, attr_name)

    return hooks


def discover_tools(instance: object, plugin_name: str) -> list[DecoratedFunctionTool]:
    """Scan an instance's class hierarchy for @tool decorated methods.

    Walks the MRO in reverse so parent class tools come first, but child
    overrides win (only the child's version is included).

    Args:
        instance: The plugin instance to scan.
        plugin_name: The plugin name (used for debug logging).

    Returns:
        List of DecoratedFunctionTool instances in declaration order.
    """
    tools: list[DecoratedFunctionTool] = []
    seen: set[str] = set()

    for cls in reversed(type(instance).__mro__):
        for attr_name in cls.__dict__:
            if attr_name in seen:
                continue
            seen.add(attr_name)

            try:
                bound = getattr(instance, attr_name)
            except Exception:
                continue

            if isinstance(bound, DecoratedFunctionTool):
                tools.append(bound)
                logger.debug("plugin=<%s>, tool=<%s> | discovered tool method", plugin_name, attr_name)

    return tools


def call_init_method(init_method: Callable[..., Any], target: Any) -> None:
    """Call a plugin's init method, handling both sync and async implementations.

    Args:
        init_method: The init_agent or init_multi_agent method to call.
        target: The agent or orchestrator instance to pass to the init method.
    """
    if inspect.iscoroutinefunction(init_method):
        async_init = cast(Callable[..., Awaitable[None]], init_method)
        run_async(lambda: async_init(target))
    else:
        init_method(target)


def register_hooks(plugin_name: str, hooks: list[HookCallback], registry: HookRegistry) -> None:
    """Register discovered hook callbacks with a hook registry.

    Args:
        plugin_name: The plugin name (used for debug logging).
        hooks: List of hook callbacks to register.
        registry: The HookRegistry to register callbacks with.
    """
    for hook_callback in hooks:
        event_types = getattr(hook_callback, "_hook_event_types", [])
        for event_type in event_types:
            registry.add_callback(event_type, hook_callback)
            logger.debug(
                "plugin=<%s>, hook=<%s>, event_type=<%s> | registered hook",
                plugin_name,
                getattr(hook_callback, "__name__", repr(hook_callback)),
                event_type.__name__,
            )
