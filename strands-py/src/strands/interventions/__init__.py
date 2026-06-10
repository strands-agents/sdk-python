"""First-class intervention primitive for agent control.

The intervention system provides a composable way to add authorization, steering,
guardrails, and other control layers to agents. Each control layer is an
InterventionHandler that intercepts lifecycle events and returns typed decisions.

Example:
    ```python
    from strands import Agent, InterventionHandler, InterventionActions

    class MyAuth(InterventionHandler):
        name = "my-auth"

        async def before_tool_call(self, event):
            if not self.is_authorized(event):
                return InterventionActions.deny("not authorized")
            return InterventionActions.proceed()

    agent = Agent(interventions=[MyAuth()])
    ```
"""

from .actions import confirm, deny, guide, proceed, transform
from .handler import InterventionHandler as InterventionHandler


class InterventionActions:
    """Namespaced factory functions for intervention actions.

    Usage:
        ```python
        from strands import InterventionActions

        InterventionActions.proceed()
        InterventionActions.deny("not authorized")
        InterventionActions.guide("try a different approach")
        InterventionActions.confirm("approve this action?")
        InterventionActions.transform(lambda e: None)
        ```
    """

    def __new__(cls) -> "InterventionActions":
        """Prevent instantiation — this is a namespace, not a class."""
        raise TypeError("InterventionActions is a namespace, not instantiable")

    proceed = staticmethod(proceed)
    deny = staticmethod(deny)
    guide = staticmethod(guide)
    confirm = staticmethod(confirm)
    transform = staticmethod(transform)
