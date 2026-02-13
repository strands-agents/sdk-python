"""Plugin base class for Strands Agents SDK.

This module provides the ``Plugin`` base class that enables self-contained bundles
of tools and hooks to be registered with an agent in a single step.  Decorated
methods are auto-discovered so that plugin authors only need to apply the
standard ``@tool`` and ``@hook`` decorators to their methods.

Example:
    ```python
    from strands import Agent, Plugin, tool
    from strands.hooks import BeforeInvocationEvent, hook

    class GreeterPlugin(Plugin):
        name = "greeter"

        @tool
        def greet(self, who: str) -> str:
            '''Say hello.'''
            return f"Hello, {who}!"

        @hook
        def log_invocation(self, event: BeforeInvocationEvent) -> None:
            '''Log every invocation.'''
            print("Invocation starting")

    agent = Agent(plugins=[GreeterPlugin()])
    ```
"""

import logging
from typing import TYPE_CHECKING, Any

from ..hooks.decorator import DecoratedFunctionHook
from ..hooks.registry import HookProvider
from ..tools.decorator import DecoratedFunctionTool

if TYPE_CHECKING:
    from ..agent.agent import Agent

logger = logging.getLogger(__name__)


class Plugin:
    """Base class for agent plugins with auto-discovery of ``@tool`` and ``@hook`` methods.

    Subclasses declare tools and hooks by decorating methods with ``@tool`` and
    ``@hook``.  When a ``Plugin`` instance is constructed, it scans its own
    methods and collects the decorated ones into the :pyattr:`tools` and
    :pyattr:`hooks` lists.  These lists can be filtered or replaced before
    the plugin is handed to an ``Agent``.

    Attributes:
        name: A human-readable identifier for the plugin.  Subclasses should
            override this with a meaningful value.
    """

    name: str = ""

    def __init__(self) -> None:
        """Initialize the plugin and auto-discover ``@tool`` / ``@hook`` methods."""
        self._tools: list[DecoratedFunctionTool[..., Any]] = []
        self._hooks: list[HookProvider] = []
        self._discover_tools_and_hooks()

    # -- public properties ---------------------------------------------------

    @property
    def tools(self) -> list[DecoratedFunctionTool[..., Any]]:
        """The list of auto-discovered (or manually set) tools for this plugin."""
        return self._tools

    @tools.setter
    def tools(self, value: list[DecoratedFunctionTool[..., Any]]) -> None:
        self._tools = value

    @property
    def hooks(self) -> list[HookProvider]:
        """The list of auto-discovered (or manually set) hooks for this plugin."""
        return self._hooks

    @hooks.setter
    def hooks(self, value: list[HookProvider]) -> None:
        self._hooks = value

    # -- lifecycle -----------------------------------------------------------

    def init_plugin(self, agent: "Agent", **kwargs: Any) -> None:
        """Optional hook called after the plugin's tools and hooks are registered.

        Override this method to perform additional setup that requires the
        fully-constructed agent (e.g. mutating ``agent.system_prompt``).

        Subclasses should always accept ``**kwargs`` so that additional keyword
        arguments can be introduced in future SDK versions without breaking
        existing plugins.

        Args:
            agent: The agent that this plugin has been registered with.
            **kwargs: Reserved for future use.
        """

    # -- internals -----------------------------------------------------------

    def _discover_tools_and_hooks(self) -> None:
        """Scan instance methods for ``@tool`` and ``@hook`` decorators.

        The scan iterates over the *class* attributes (via ``type(self)``) so
        that descriptors are seen in their raw form.  The *instance* attribute
        is then fetched to obtain a properly bound version.
        """
        seen: set[str] = set()
        for cls in type(self).__mro__:
            for attr_name, cls_attr in vars(cls).items():
                if attr_name.startswith("_") or attr_name in seen:
                    continue
                seen.add(attr_name)

                if isinstance(cls_attr, DecoratedFunctionTool):
                    # Accessing through the instance triggers __get__ which
                    # returns a properly bound DecoratedFunctionTool.
                    bound_tool: DecoratedFunctionTool[..., Any] = getattr(self, attr_name)
                    self._tools.append(bound_tool)
                    logger.debug(
                        "plugin=%s | discovered tool: %s",
                        self.name or type(self).__name__,
                        bound_tool.tool_name,
                    )

                elif isinstance(cls_attr, DecoratedFunctionHook):
                    # Accessing through the instance triggers __get__ which
                    # returns a properly bound DecoratedFunctionHook.
                    bound_hook: DecoratedFunctionHook[Any] = getattr(self, attr_name)
                    self._hooks.append(bound_hook)
                    logger.debug(
                        "plugin=%s | discovered hook: %s",
                        self.name or type(self).__name__,
                        bound_hook.name,
                    )

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"Plugin(name={self.name!r}, "
            f"tools=[{', '.join(t.tool_name for t in self._tools)}], "
            f"hooks=[{', '.join(h.name if hasattr(h, 'name') else type(h).__name__ for h in self._hooks)}])"
        )
