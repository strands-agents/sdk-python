"""Steering handler base class for providing contextual guidance to agents.

Provides modular prompting through contextual guidance that appears when relevant,
rather than front-loading all instructions. Handlers integrate with the Strands hook
system to intercept tool calls and provide just-in-time feedback based on local context.

Architecture:
    BeforeToolCallEvent → Context Callbacks → Update steering_context → steer() → SteeringAction
            ↓                    ↓                      ↓                ↓           ↓
    Hook triggered      Populate context      Handler evaluates    Handler decides   Action taken

Lifecycle:
    1. Context callbacks update handler's steering_context on hook events
    2. BeforeToolCallEvent triggers steering evaluation via steer() method
    3. Handler accesses self.steering_context for guidance decisions
    4. SteeringAction determines tool execution: Proceed/Guide/Interrupt

Implementation:
    Subclass SteeringHandler and implement steer() method.
    Pass context_callbacks in constructor to register context update functions.
    Each handler maintains isolated steering_context that persists across calls.

SteeringAction handling:
    Proceed: Tool executes immediately
    Guide: Tool cancelled, agent receives contextual feedback to explore alternatives
    Interrupt: Tool execution paused for human input via interrupt system
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ....hooks.events import BeforeToolCallEvent
from ....hooks.registry import HookProvider, HookRegistry
from ....types.tools import ToolUse
from .action import Guide, Interrupt, Proceed, SteeringAction
from .context import SteeringContext, SteeringContextProvider

if TYPE_CHECKING:
    from ....agent import Agent

logger = logging.getLogger(__name__)


class SteeringHandler(HookProvider, ABC):
    """Base class for steering handlers that provide contextual guidance to agents.

    Steering handlers maintain local context and register hook callbacks
    to populate context data as needed for guidance decisions.
    """

    def __init__(self, context_providers: list[SteeringContextProvider] | None = None):
        """Initialize the steering handler.

        Args:
            context_providers: List of context providers for context updates
        """
        super().__init__()
        self.steering_context = SteeringContext()
        self._context_callbacks = []

        # Collect callbacks from all providers
        for provider in context_providers or []:
            self._context_callbacks.extend(provider.context_providers())

        logger.debug("handler_class=<%s> | initialized", self.__class__.__name__)

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks for steering guidance and context updates."""
        # Register context update callbacks
        for callback in self._context_callbacks:
            registry.add_callback(
                callback.event_type, lambda event, callback=callback: callback(event, self.steering_context)
            )

        # Register steering guidance
        registry.add_callback(BeforeToolCallEvent, self._provide_steering_guidance)

    async def _provide_steering_guidance(self, event: BeforeToolCallEvent) -> None:
        """Provide steering guidance for tool call."""
        tool_name = event.tool_use["name"]
        logger.debug("tool_name=<%s> | providing steering guidance", tool_name)

        try:
            action = await self.steer(event.agent, event.tool_use)
        except Exception as e:
            logger.debug("tool_name=<%s>, error=<%s> | steering handler guidance failed", tool_name, e)
            return

        self._handle_steering_action(action, event, tool_name)

    def _handle_steering_action(self, action: SteeringAction, event: BeforeToolCallEvent, tool_name: str) -> None:
        """Handle the steering action by modifying tool execution flow.

        Proceed: Tool executes normally
        Guide: Tool cancelled with contextual feedback for agent to consider alternatives
        Interrupt: Tool execution paused for human input via interrupt system
        """
        if isinstance(action, Proceed):
            logger.debug("tool_name=<%s> | tool call proceeding", tool_name)
        elif isinstance(action, Guide):
            logger.debug("tool_name=<%s> | tool call guided: %s", tool_name, action.reason)
            event.cancel_tool = (
                f"Tool call cancelled given new guidance. {action.reason}. Consider this approach and continue"
            )
        elif isinstance(action, Interrupt):
            logger.debug("tool_name=<%s> | tool call requires human input: %s", tool_name, action.reason)
            can_proceed: bool = event.interrupt(name=f"steering_input_{tool_name}", reason={"message": action.reason})
            logger.debug("tool_name=<%s> | received human input for tool call", tool_name)

            if not can_proceed:
                event.cancel_tool = f"Manual approval denied: {action.reason}"
                logger.debug("tool_name=<%s> | tool call denied by manual approval", tool_name)
            else:
                logger.debug("tool_name=<%s> | tool call approved manually", tool_name)
        else:
            raise ValueError(f"Unknown steering action type: {action}")

    @abstractmethod
    async def steer(self, agent: "Agent", tool_use: ToolUse, **kwargs: Any) -> SteeringAction:
        """Provide contextual guidance to help agent navigate complex workflows.

        Args:
            agent: The agent instance
            tool_use: The tool use object with name and arguments
            **kwargs: Additional keyword arguments for guidance evaluation

        Returns:
            SteeringAction indicating how to guide the agent's next action

        Note:
            Access steering context via self.steering_context
        """
        ...
