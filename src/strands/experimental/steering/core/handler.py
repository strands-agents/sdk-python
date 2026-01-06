"""Steering handler base class for providing contextual guidance to agents.

Provides modular prompting through contextual guidance that appears when relevant,
rather than front-loading all instructions. Handlers integrate with the Strands hook
system to intercept actions and provide just-in-time feedback based on local context.

Architecture:
    Hook Event → Context Callbacks → Update steering_context → steer_*() → SteeringAction
         ↓                ↓                      ↓                   ↓            ↓
    Hook triggered  Populate context    Handler evaluates    Handler decides  Action taken

Lifecycle:
    1. Context callbacks update handler's steering_context on hook events
    2. BeforeToolCallEvent triggers steer_before_tool() for tool steering
    3. AfterModelCallEvent triggers steer_after_model() for model steering
    4. Handler accesses self.steering_context for guidance decisions
    5. SteeringAction determines execution flow

Implementation:
    Subclass SteeringHandler and override steer_before_tool() and/or steer_after_model().
    Both methods have default implementations that return Proceed, so you only need to
    override the methods you want to customize.
    Pass context_providers in constructor to register context update functions.
    Each handler maintains isolated steering_context that persists across calls.

SteeringAction handling for steer_before_tool:
    Proceed: Tool executes immediately
    Guide: Tool cancelled, agent receives contextual feedback to explore alternatives
    Interrupt: Tool execution paused for human input via interrupt system

SteeringAction handling for steer_after_model:
    Proceed: Model response accepted without modification
    Guide: Discard model response and retry (message is dropped, model is called again)
    Interrupt: Model response handling paused for human input via interrupt system
"""

import logging
import warnings
from abc import ABC
from typing import TYPE_CHECKING, Any

from ....hooks.events import AfterModelCallEvent, BeforeToolCallEvent
from ....hooks.registry import HookProvider, HookRegistry
from ....types.content import Message
from ....types.streaming import StopReason
from ....types.tools import ToolUse
from .action import Guide, Interrupt, ModelSteeringAction, Proceed, SteeringAction, ToolSteeringAction
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

        # Register tool steering guidance
        registry.add_callback(BeforeToolCallEvent, self._provide_tool_steering_guidance)

        # Register model steering guidance
        registry.add_callback(AfterModelCallEvent, self._provide_model_steering_guidance)

    async def _provide_tool_steering_guidance(self, event: BeforeToolCallEvent) -> None:
        """Provide steering guidance for tool call."""
        tool_name = event.tool_use["name"]
        logger.debug("tool_name=<%s> | providing tool steering guidance", tool_name)

        try:
            action = await self.steer_before_tool(event.agent, event.tool_use)
        except Exception as e:
            logger.debug("tool_name=<%s>, error=<%s> | tool steering handler guidance failed", tool_name, e)
            return

        self._handle_tool_steering_action(action, event, tool_name)

    def _handle_tool_steering_action(self, action: SteeringAction, event: BeforeToolCallEvent, tool_name: str) -> None:
        """Handle the steering action for tool calls by modifying tool execution flow.

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
            raise ValueError(f"Unknown steering action type for tool call: {action}")

    async def _provide_model_steering_guidance(self, event: AfterModelCallEvent) -> None:
        """Provide steering guidance for model response."""
        logger.debug("providing model steering guidance")

        # Only steer on successful model responses
        if event.stop_response is None:
            logger.debug("no stop response available | skipping model steering")
            return

        try:
            action = await self.steer_after_model(
                event.agent, event.stop_response.message, event.stop_response.stop_reason
            )
        except Exception as e:
            logger.debug("error=<%s> | model steering handler guidance failed", e)
            return

        await self._handle_model_steering_action(action, event)

    async def _handle_model_steering_action(self, action: ModelSteeringAction, event: AfterModelCallEvent) -> None:
        """Handle the steering action for model responses by modifying response handling flow.

        Proceed: Model response accepted without modification
        Guide: Discard model response and retry with guidance message added to conversation
        """
        if isinstance(action, Proceed):
            logger.debug("model response proceeding")
        elif isinstance(action, Guide):
            logger.debug("model response guided (retrying): %s", action.reason)
            # Set retry flag to discard current response
            event.retry = True
            # Add guidance message to agent's conversation so model sees it on retry
            await event.agent._append_messages({"role": "user", "content": [{"text": action.reason}]})
            logger.debug("added guidance message to conversation for model retry")
        else:
            raise ValueError(f"Unknown steering action type for model response: {action}")

    async def steer_before_tool(self, agent: "Agent", tool_use: ToolUse, **kwargs: Any) -> ToolSteeringAction:
        """Provide contextual guidance before tool execution.

        This method is called before a tool is executed, allowing the handler to:
        - Proceed: Allow tool execution to continue
        - Guide: Cancel tool and provide feedback for alternative approaches
        - Interrupt: Pause for human input before tool execution

        Args:
            agent: The agent instance
            tool_use: The tool use object with name and arguments
            **kwargs: Additional keyword arguments for guidance evaluation

        Returns:
            ToolSteeringAction indicating how to guide the tool execution

        Note:
            Access steering context via self.steering_context
            Default implementation returns Proceed (allow tool execution)
            Override this method to implement custom tool steering logic
        """
        return Proceed(reason="Default implementation: allowing tool execution")

    async def steer_after_model(
        self, agent: "Agent", message: Message, stop_reason: StopReason, **kwargs: Any
    ) -> ModelSteeringAction:
        """Provide contextual guidance after model response.

        This method is called after the model generates a response, allowing the handler to:
        - Proceed: Accept the model response without modification
        - Guide: Discard the response and retry (message is dropped, model is called again)

        Note: Interrupt is not supported for model steering as the model has already responded.

        Args:
            agent: The agent instance
            message: The model's generated message
            stop_reason: The reason the model stopped generating
            **kwargs: Additional keyword arguments for guidance evaluation

        Returns:
            ModelSteeringAction indicating how to handle the model response

        Note:
            Access steering context via self.steering_context
            Default implementation returns Proceed (accept response as-is)
            Override this method to implement custom model steering logic
        """
        return Proceed(reason="Default implementation: accepting model response")

    async def steer(self, agent: "Agent", tool_use: ToolUse, **kwargs: Any) -> SteeringAction:
        """Provide contextual guidance to help agent navigate complex workflows.

        .. deprecated::
            Use `steer_before_tool` instead. This method will be removed in a future version.

        Args:
            agent: The agent instance
            tool_use: The tool use object with name and arguments
            **kwargs: Additional keyword arguments for guidance evaluation

        Returns:
            SteeringAction indicating how to guide the agent's next action

        Note:
            Access steering context via self.steering_context
        """
        warnings.warn(
            "steer() is deprecated and will be removed in a future version. Use steer_before_tool() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.steer_before_tool(agent, tool_use, **kwargs)
