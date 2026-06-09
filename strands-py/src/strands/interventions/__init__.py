"""First-class intervention primitive for agent control.

The intervention system provides a composable way to add authorization, steering,
guardrails, and other control layers to agents. Each control layer is an
InterventionHandler that intercepts lifecycle events and returns typed decisions.

Example:
    ```python
    from strands import Agent
    from strands.interventions import InterventionHandler, deny, proceed

    class MyAuth(InterventionHandler):
        name = "my-auth"

        async def before_tool_call(self, event):
            if not self.is_authorized(event):
                return deny("not authorized")
            return proceed()

    agent = Agent(interventions=[MyAuth()])
    ```
"""

from .actions import (
    Confirm,
    Deny,
    Guide,
    InterventionAction,
    Proceed,
    Transform,
    confirm,
    deny,
    guide,
    proceed,
    transform,
)
from .handler import InterventionHandler, OnError
from .registry import InterventionRegistry

__all__ = [
    "Confirm",
    "Deny",
    "Guide",
    "InterventionAction",
    "InterventionHandler",
    "InterventionRegistry",
    "OnError",
    "Proceed",
    "Transform",
    "confirm",
    "deny",
    "guide",
    "proceed",
    "transform",
]
