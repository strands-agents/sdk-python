"""Bundled routing strategies.

``ContextFitStrategy`` is a local (no extra model call) strategy that routes by context capacity,
reusing the conversation manager's window default and threshold so routing and proactive compression
share one notion of "how full is too full". ``FallbackStrategy`` lives in ``router.py`` alongside the
router-owned ordered-fallback mechanism it pairs with.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ...agent.conversation_manager.conversation_manager import (
    DEFAULT_COMPRESSION_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW_LIMIT,
)
from ..model import Model
from .router import ModelRouter, RoutingCandidate
from .strategy import RoutingContext

logger = logging.getLogger(__name__)


class ContextFitStrategy:
    """Routes by context capacity: the smallest window that holds the request within ``threshold``.

    For each candidate it estimates the request's token count with that candidate's own
    ``count_tokens`` (tokenization differs by provider), then treats the candidate as fitting when
    the estimate stays within ``threshold`` of its context window. ``threshold`` defaults to the
    conversation manager's ``DEFAULT_COMPRESSION_THRESHOLD``, so a fitting candidate is one whose
    context would not immediately trip proactive compression. It returns the smallest fitting window
    and the largest window when none fit. A model with no declared (or unreadable) window is treated
    as ``DEFAULT_CONTEXT_WINDOW_LIMIT``. Token counting is best-effort: a candidate whose count fails
    is treated as not fitting.
    """

    def __init__(self, *, threshold: float = DEFAULT_COMPRESSION_THRESHOLD) -> None:
        """Initialize the strategy.

        Args:
            threshold: Fraction of a candidate's context window the request may occupy and still be
                considered a fit. Defaults to the conversation manager's compression threshold.

        Raises:
            ValueError: If ``threshold`` is not in ``(0.0, 1.0]``.
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0.0, 1.0]")
        self._threshold = threshold

    async def select(self, context: RoutingContext, **kwargs: Any) -> RoutingCandidate:
        """Return the smallest-window candidate whose window holds the request within ``threshold``."""
        windows = [_candidate_window(candidate) for candidate in context.candidates]
        counts = [await _count(candidate, context) for candidate in context.candidates]
        fitting = [index for index, window in enumerate(windows) if counts[index] <= window * self._threshold]
        if not fitting:
            logger.warning("no candidate window fits the request; falling back to the largest window")
        chosen = (
            min(fitting, key=lambda index: windows[index])
            if fitting
            else max(range(len(windows)), key=lambda index: windows[index])
        )
        return context.candidates[chosen]


async def _count(candidate: RoutingCandidate, context: RoutingContext) -> float:
    """Estimate the request's token count with the candidate's tokenizer; non-finite or errors are unfit."""
    model = _concrete_model(candidate)
    system_prompt = context.system_prompt if isinstance(context.system_prompt, str) else None
    system_prompt_content = context.system_prompt if isinstance(context.system_prompt, list) else None
    try:
        value = float(
            await model.count_tokens(
                list(context.messages),
                list(context.tool_specs),
                system_prompt=system_prompt,
                system_prompt_content=system_prompt_content,
            )
        )
    except Exception as error:
        logger.debug("model=<%s>, error=<%s> | token count failed, treating candidate as not fitting", model, error)
        return math.inf
    return value if math.isfinite(value) and value >= 0 else math.inf


def _concrete_model(candidate: RoutingCandidate) -> Model:
    """Resolve a candidate to a concrete model, using a nested router's default."""
    model = candidate.model
    return model.default_model if isinstance(model, ModelRouter) else model


def _candidate_window(candidate: RoutingCandidate) -> int:
    """Return a candidate's context window, using the shared default when it is undeclared or unreadable."""
    try:
        limit = _concrete_model(candidate).context_window_limit
    except Exception as error:
        logger.debug("candidate=<%s>, error=<%s> | window read failed, using default", candidate.name, error)
        return DEFAULT_CONTEXT_WINDOW_LIMIT
    return limit if limit is not None else DEFAULT_CONTEXT_WINDOW_LIMIT
