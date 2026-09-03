"""ContextManager: first-class agent component for strategy-driven context management.

On overflow, runs the strategy pipeline (including an emergency truncation as the final step).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..hooks.events import AfterModelCallEvent, BeforeModelCallEvent
from ..plugins.plugin import Plugin
from ..types.exceptions import ContextWindowOverflowException
from .strategies.offload import Offload
from .strategies.offload.base import EmergencyTruncateStrategy
from .types import ContextState, ContextStrategy

if TYPE_CHECKING:
    from ..agent.agent import Agent

logger = logging.getLogger(__name__)


class ContextManager(Plugin):
    """Manages context reduction for an agent's conversation.

    On context overflow, runs the strategy pipeline (offload, summarize, emergency truncate).
    The emergency truncation is always appended as the final strategy — it recomputes
    utilization and only fires if the window is still overflowing after user strategies.

    Pass via the ``context_manager`` parameter on the Agent constructor. When present,
    it owns overflow recovery — no separate ConversationManager is needed.
    """

    @property
    def name(self) -> str:
        """Plugin name."""
        return "strands:context-manager"

    def __init__(self, *, strategies: list[ContextStrategy] | None = None) -> None:
        """Initialize with an optional ordered list of strategies (defaults provided)."""
        super().__init__()
        self._strategies: list[ContextStrategy] = [
            *(
                strategies
                if strategies is not None
                else [
                    Offload.truncate("tool_results").when(threshold=2500),
                    Offload.summarize("*").when(threshold=1000, utilization=0.85),
                ]
            ),
            EmergencyTruncateStrategy(),
        ]

    def init_agent(self, agent: Agent) -> None:
        """Register strategy hooks for proactive compression and overflow recovery."""
        for strategy in self._strategies:
            init = getattr(strategy, "init", None)
            if init is not None:
                init(agent)

        async def _on_before_model_call(event: BeforeModelCallEvent) -> None:
            await self._run_strategies(event.agent, event.projected_input_tokens)

        agent.hooks.add_callback(BeforeModelCallEvent, _on_before_model_call)

        overflow_retries = 0

        async def _on_after_model_call(event: AfterModelCallEvent) -> None:
            nonlocal overflow_retries

            if not isinstance(event.exception, ContextWindowOverflowException):
                overflow_retries = 0
                return

            if overflow_retries >= 3:
                logger.warning("agent_id=<%s> | overflow retry limit reached, giving up", event.agent.agent_id)
                overflow_retries = 0
                return

            acted = await self._run_strategies(event.agent)
            if not acted:
                logger.warning("agent_id=<%s> | no strategy made progress, skipping retry", event.agent.agent_id)
                return

            overflow_retries += 1
            event.retry = True

        agent.hooks.add_callback(AfterModelCallEvent, _on_after_model_call)

    async def _run_strategies(self, agent: Agent, precomputed_input_tokens: int | None = None) -> bool:
        """Run the strategy pipeline, recomputing utilization after each acting strategy."""
        messages = agent.messages
        if precomputed_input_tokens is not None:
            input_tokens = precomputed_input_tokens
        else:
            input_tokens = await agent.model.count_tokens(messages)

        context = ContextState(
            messages=messages,
            agent=agent,
            utilization=agent.model.estimate_utilization(input_tokens),
        )

        any_acted = False
        for strategy in self._strategies:
            try:
                acted = await strategy.apply(context)
                if acted:
                    any_acted = True
                    new_tokens = await agent.model.count_tokens(messages)
                    context.utilization = agent.model.estimate_utilization(new_tokens)
                    logger.debug("strategy=<%s>, agent_id=<%s> | strategy applied", strategy.name, agent.agent_id)
            except Exception:
                logger.warning(
                    "strategy=<%s>, agent_id=<%s> | strategy failed, continuing",
                    strategy.name,
                    agent.agent_id,
                    exc_info=True,
                )
        return any_acted
