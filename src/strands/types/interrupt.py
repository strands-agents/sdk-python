"""Interrupt related type definitions for human-in-the-loop workflows.

Interrupt Flow:
    ┌─────────────────┐
    │ Agent Invoke    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Hook Calls      │
    | on Event        |
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐    No     ┌─────────────────┐
    │ Interrupts      │ ────────► │ Continue        │
    │ Raised?         │           │ Execution       │
    └────────┬────────┘           └─────────────────┘
             │ Yes
             ▼
    ┌─────────────────┐
    │ Stop Event Loop │◄───────────────────┐
    └────────┬────────┘                    |
             │                             |
             ▼                             |
    ┌─────────────────┐                    |
    │ Return          |                    |
    | Interrupts      │                    |
    └────────┬────────┘                    |
             │                             |
             ▼                             |
    ┌─────────────────┐                    |
    │ Agent Invoke    │                    |
    │ with Responses  │                    |
    └────────┬────────┘                    |
             │                             |
             ▼                             |
    ┌─────────────────┐                    |
    │ Hook Calls      │                    |
    | on Event        |                    |
    | with Responses  |                    |
    └────────┬────────┘                    |
             │                             |
             ▼                             |
    ┌─────────────────┐    Yes    ┌────────┴────────┐
    │ New Interrupts  │ ────────► │ Store State     │
    │ Raised?         │           │                 │
    └────────┬────────┘           └─────────────────┘
             │ No
             ▼
    ┌─────────────────┐
    │ Continue        │
    │ Execution       │
    └─────────────────┘

Example:
    ```
    from typing import Any

    from strands import Agent, tool
    from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry


    @tool
    def delete_tool(key: str) -> bool:
        print("DELETE_TOOL | deleting")
        return True


    class ToolInterruptHook(HookProvider):
        def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
            registry.add_callback(BeforeToolCallEvent, self.approve)

        def approve(self, event: BeforeToolCallEvent) -> None:
            if event.tool_use["name"] != "delete_tool":
                return

            approval = event.interrupt("for_delete_tool", reason="APPROVAL")
            if approval != "A":
                event.cancel_tool = "approval was not granted"

    agent = Agent(
        hooks=[ToolInterruptHook()],
        tools=[delete_tool],
        system_prompt="You delete objects given their keys.",
        callback_handler=None,
    )
    result = agent(f"delete object with key 'X'")

    if result.stop_reason == "interrupt":
        responses = []
        for interrupt in result.interrupts:
            if interrupt.name == "for_delete_tool":
                responses.append({"interruptResponse": {"interruptId": interrupt.id, "response": "A"})

        result = agent(responses)

    ...
    ```

Details:

    - User raises interrupt on their hook event by calling `event.interrupt()`.
    - User can raise one interrupt per hook callback.
    - Interrupts stop the agent event loop.
    - Interrupts are returned to the user in AgentResult.
    - User resumes by invoking agent with interrupt responses.
    - Second call to `event.interrupt()` returns user response.
    - Process repeats if user raises additional interrupts.
    - Interrupts are session managed in-between return and user response.
"""

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from ..interrupt import Interrupt, InterruptException

if TYPE_CHECKING:
    from ..agent import Agent


class _Interruptible(Protocol):
    """Interface that adds interrupt support to hook events and tools."""

    agent: "Agent"

    def interrupt(self, name: str, reason: Any = None, response: Any = None) -> Any:
        """Trigger the interrupt with a reason.

        Args: name: User defined name for the interrupt.
                Must be unique across hook callbacks.
            reason: User provided reason for the interrupt.
            response: Preemptive response from user if available.

        Returns:
            The response from a human user when resuming from an interrupt state.

        Raises:
            InterruptException: If human input is required.
        """
        id = self._interrupt_id(name)
        state = self.agent._interrupt_state

        interrupt_ = state.interrupts.setdefault(id, Interrupt(id, name, reason, response))
        if interrupt_.response:
            return interrupt_.response

        raise InterruptException(interrupt_)

    def _interrupt_id(self, name: str) -> str:
        """Unique id for the interrupt.

        Args:
            name: User defined name for the interrupt.
            reason: User provided reason for the interrupt.

        Returns:
            Interrupt id.
        """
        ...


class InterruptResponse(TypedDict):
    """User response to an interrupt.

    Attributes:
        interruptId: Unique identifier for the interrupt.
        response: User response to the interrupt.
    """

    interruptId: str
    response: Any


class InterruptResponseContent(TypedDict):
    """Content block containing a user response to an interrupt.

    Attributes:
        interruptResponse: User response to an interrupt event.
    """

    interruptResponse: InterruptResponse
