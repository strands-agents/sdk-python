"""Base class for intervention handlers.

Handlers override the lifecycle methods they care about. Default implementations
return Proceed. The framework detects which methods are overridden and only
registers hook callbacks for those.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from ..hooks.events import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)
from .actions import Confirm, Deny, Guide, Proceed, Transform

OnError = Literal["throw", "proceed", "deny"]
"""What to do when a handler throws during evaluation.

- ``'throw'`` — rethrow the error (default, safest: a broken policy check blocks execution)
- ``'proceed'`` — log the error and continue as if the handler returned Proceed.
  **This mode is fail-open**: a broken handler silently stops enforcing its policy.
  Use only when availability matters more than enforcement.
- ``'deny'`` — log the error and treat it as a Deny (fail-closed)
"""


class InterventionHandler(ABC):
    """Base class for intervention handlers.

    Subclasses must define a ``name`` attribute and override the lifecycle
    methods they care about at the **class level**. The framework detects which
    methods are overridden and only calls those. Instance-level assignments
    (e.g., ``handler.before_tool_call = my_func``) are not detected.

    Example:
        ```python
        class CedarAuth(InterventionHandler):
            name = "cedar-auth"

            def before_tool_call(self, event):
                if not self.is_authorized(event):
                    return Deny(reason="not authorized")
                return Proceed()
        ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this handler."""
        ...

    @property
    def on_error(self) -> OnError:
        """What to do when this handler throws. Defaults to 'throw'."""
        return "throw"

    def before_invocation(self, event: BeforeInvocationEvent, **kwargs: Any) -> Proceed | Deny | Guide | Transform:
        """Called before an agent invocation begins."""
        return Proceed()

    def before_tool_call(
        self, event: BeforeToolCallEvent, **kwargs: Any
    ) -> Proceed | Deny | Guide | Confirm | Transform:
        """Called before a tool is executed."""
        return Proceed()

    def after_tool_call(self, event: AfterToolCallEvent, **kwargs: Any) -> Proceed | Transform:
        """Called after a tool execution completes."""
        return Proceed()

    def before_model_call(self, event: BeforeModelCallEvent, **kwargs: Any) -> Proceed | Deny | Guide | Transform:
        """Called before the model is invoked."""
        return Proceed()

    def after_model_call(self, event: AfterModelCallEvent, **kwargs: Any) -> Proceed | Guide | Transform:
        """Called after the model invocation completes."""
        return Proceed()
