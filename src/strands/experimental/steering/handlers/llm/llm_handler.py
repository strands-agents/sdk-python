"""LLM-based steering handler with support for both tool and model output steering.

Extends the LLMSteeringHandler to support steer_after_model in addition to steer_before_tool.
Users opt in to each hook by providing the corresponding prompt mapper.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field

from .....models import Model
from .....types.content import Message
from .....types.streaming import StopReason
from .....types.tools import ToolUse
from ...context_providers.ledger_provider import LedgerProvider
from ...core.action import Guide, Interrupt, ModelSteeringAction, Proceed, ToolSteeringAction
from ...core.context import SteeringContextProvider
from ...core.handler import SteeringHandler
from .mappers import DefaultPromptMapper as DefaultToolPromptMapper
from .mappers import LLMModelPromptMapper, LLMPromptMapper as LLMToolPromptMapper
from .mappers import DefaultModelPromptMapper

if TYPE_CHECKING:
    from .....agent import Agent

logger = logging.getLogger(__name__)


class _LLMToolSteering(BaseModel):
    """Structured output model for tool steering decisions."""

    decision: Literal["proceed", "guide", "interrupt"] = Field(
        description="Steering decision: 'proceed' to continue, 'guide' to provide feedback, 'interrupt' for human input"
    )
    reason: str = Field(description="Clear explanation of the steering decision and any guidance provided")


class _LLMModelSteering(BaseModel):
    """Structured output model for model steering decisions.

    Only 'proceed' and 'guide' are valid since the model has already
    responded (interrupt doesn't apply).
    """

    decision: Literal["proceed", "guide"] = Field(
        description="Steering decision: 'proceed' to accept the output, 'guide' to reject and retry with feedback"
    )
    reason: str = Field(description="Clear explanation of the steering decision and any guidance provided")


class LLMSteeringHandler(SteeringHandler):
    """Steering handler that uses an LLM to provide contextual guidance.

    Supports both tool steering and model output steering. Opt in to each
    by providing the corresponding prompt mapper:

        # Tool steering only (same as existing SDK behavior)
        handler = LLMSteeringHandler(
            system_prompt="...",
            tool_prompt_mapper=DefaultToolPromptMapper(),
        )

        # Model output steering only
        handler = LLMSteeringHandler(
            system_prompt="...",
            model_prompt_mapper=DefaultModelPromptMapper(),
        )

        # Both
        handler = LLMSteeringHandler(
            system_prompt="...",
            tool_prompt_mapper=DefaultToolPromptMapper(),
            model_prompt_mapper=DefaultModelPromptMapper(),
        )

    If neither mapper is provided, both hooks default to Proceed.
    """

    def __init__(
        self,
        system_prompt: str,
        model: Model | None = None,
        tool_prompt_mapper: LLMToolPromptMapper | None = None,
        model_prompt_mapper: LLMModelPromptMapper | None = None,
        context_providers: list[SteeringContextProvider] | None = None,
    ):
        """Initialize the LLMSteeringHandler.

        Args:
            system_prompt: System prompt defining steering guidance rules
            model: Optional model override for steering evaluation
            tool_prompt_mapper: Prompt mapper for tool steering evaluation
            model_prompt_mapper: Prompt mapper for model output steering evaluation
            context_providers: List of context providers for populating steering context.
                Defaults to [LedgerProvider()] if None. Pass an empty list to disable
                context providers.
        """
        providers: list[SteeringContextProvider] = (
            [LedgerProvider()] if context_providers is None else context_providers
        )
        super().__init__(context_providers=providers)
        self.system_prompt = system_prompt
        self.model = model
        self.tool_prompt_mapper = tool_prompt_mapper
        self.model_prompt_mapper = model_prompt_mapper

    def _create_steering_agent(self, agent: Agent) -> Agent:
        """Create a fresh, isolated steering agent for evaluation.

        Args:
            agent: The parent agent whose model to use as fallback

        Returns:
            A new Agent instance configured for steering evaluation
        """
        from .....agent import Agent

        return Agent(
            system_prompt=self.system_prompt,
            model=self.model or agent.model,
            callback_handler=None,
        )

    async def steer_before_tool(self, *, agent: Agent, tool_use: ToolUse, **kwargs: Any) -> ToolSteeringAction:
        """Provide contextual guidance for tool usage.

        Args:
            agent: The agent instance
            tool_use: The tool use object with name and arguments
            **kwargs: Additional keyword arguments for steering evaluation

        Returns:
            ToolSteeringAction indicating how to guide the tool execution
        """
        if self.tool_prompt_mapper is None:
            return Proceed(reason="Tool steering not configured")

        prompt = self.tool_prompt_mapper.create_steering_prompt(self.steering_context, tool_use=tool_use)

        steering_agent = self._create_steering_agent(agent)
        llm_result: _LLMToolSteering = cast(
            _LLMToolSteering,
            steering_agent(prompt, structured_output_model=_LLMToolSteering).structured_output,
        )

        match llm_result.decision:
            case "proceed":
                return Proceed(reason=llm_result.reason)
            case "guide":
                return Guide(reason=llm_result.reason)
            case "interrupt":
                return Interrupt(reason=llm_result.reason)
            case _:
                logger.warning("decision=<%s> | unknown llm decision, defaulting to proceed", llm_result.decision)
                return Proceed(reason="Unknown LLM decision, defaulting to proceed")

    async def steer_after_model(
        self, *, agent: Agent, message: Message, stop_reason: StopReason, **kwargs: Any
    ) -> ModelSteeringAction:
        """Provide contextual guidance for model output.

        Args:
            agent: The agent instance
            message: The model's generated message
            stop_reason: The reason the model stopped generating
            **kwargs: Additional keyword arguments for steering evaluation

        Returns:
            ModelSteeringAction indicating how to guide the model output
        """
        if self.model_prompt_mapper is None:
            return Proceed(reason="Model steering not configured")

        agent_system_prompt = getattr(agent, "system_prompt", None)

        prompt = self.model_prompt_mapper.create_steering_prompt(
            self.steering_context,
            message=message,
            stop_reason=stop_reason,
            agent_system_prompt=agent_system_prompt,
        )

        steering_agent = self._create_steering_agent(agent)
        llm_result: _LLMModelSteering = cast(
            _LLMModelSteering,
            steering_agent(prompt, structured_output_model=_LLMModelSteering).structured_output,
        )

        match llm_result.decision:
            case "proceed":
                return Proceed(reason=llm_result.reason)
            case "guide":
                return Guide(reason=llm_result.reason)
            case _:
                logger.warning("decision=<%s> | unknown llm decision, defaulting to proceed", llm_result.decision)
                return Proceed(reason="Unknown LLM decision, defaulting to proceed")
