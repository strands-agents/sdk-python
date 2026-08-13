"""A local routing strategy that selects a model whose context window fits the request."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...types.content import split_system_prompt
from .._defaults import DEFAULT_COMPRESSION_THRESHOLD, DEFAULT_CONTEXT_WINDOW_LIMIT
from ..model import Model
from .strategy import RoutingContext, RoutingStrategy

if TYPE_CHECKING:
    from .router import RoutingCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CandidateFit:
    candidate: RoutingCandidate
    input_tokens: float | None = None
    context_window_limit: float | None = None
    fits: bool | None = None


class ContextFitStrategy:
    """Select the smallest direct model whose context window fits the request.

    Candidate models count the request with their own tokenizer. Models without a configured context
    window use :data:`DEFAULT_CONTEXT_WINDOW_LIMIT`. A nested ``ModelRouter`` or a candidate whose
    measurement fails has unknown fit: a known fitting model wins, otherwise the first unknown candidate
    is preserved rather than excluded. When every fit is known and none fits, the largest model wins.

    After a model failure, selection is delegated to ``fallback`` when configured. With no fallback,
    returning ``None`` ends routing and preserves the model error.
    """

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_COMPRESSION_THRESHOLD,
        fallback: RoutingStrategy | None = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            threshold: Maximum context-window utilization considered a fit, in ``(0, 1]``.
            fallback: Strategy that handles selection after a model failure. Defaults to no failover.

        Raises:
            ValueError: If ``threshold`` is not finite or is outside ``(0, 1]``.
        """
        if not math.isfinite(threshold) or threshold <= 0 or threshold > 1:
            raise ValueError(f"threshold must be between 0 (exclusive) and 1 (inclusive), got {threshold}")
        self._threshold = threshold
        self._fallback = fallback

    async def select(self, context: RoutingContext, **kwargs: Any) -> RoutingCandidate | None:
        """Return the best-fitting candidate, or delegate selection after a failure."""
        if context.attempts:
            if self._fallback is None:
                return None
            return await self._fallback.select(context)

        measured = await _measure_candidate_fit(context, self._threshold)
        fitting = [measurement for measurement in measured if measurement.fits]
        if fitting:
            return min(fitting, key=lambda measurement: measurement.context_window_limit or 0).candidate

        unknown = [measurement for measurement in measured if measurement.fits is None]
        if unknown:
            return unknown[0].candidate

        if measured:
            return max(measured, key=lambda measurement: measurement.context_window_limit or 0).candidate
        return None


async def _measure_candidate_fit(context: RoutingContext, threshold: float) -> list[_CandidateFit]:
    """Measure direct candidates sequentially, preserving unmeasurable candidates as unknown."""
    system_prompt, system_prompt_content = split_system_prompt(context.system_prompt)
    measurements: list[_CandidateFit] = []

    for candidate in context.candidates:
        model = candidate.model
        if not isinstance(model, Model):
            measurements.append(_CandidateFit(candidate=candidate))
            continue

        try:
            configured_limit = model.context_window_limit
            limit = DEFAULT_CONTEXT_WINDOW_LIMIT if configured_limit is None else _nonnegative_number(configured_limit)
            tokens = _nonnegative_number(
                await model.count_tokens(
                    context.messages,
                    tool_specs=list(context.tool_specs),
                    system_prompt=system_prompt,
                    system_prompt_content=system_prompt_content,
                )
            )
        except (TypeError, ValueError) as error:
            logger.debug(
                "candidate=<%s>, error=<%s> | context fit could not be measured",
                candidate.name or type(model).__name__,
                error,
            )
            measurements.append(_CandidateFit(candidate=candidate))
            continue
        except Exception as error:
            logger.debug(
                "candidate=<%s>, error=<%s> | context fit token counting failed",
                candidate.name or type(model).__name__,
                error,
            )
            measurements.append(_CandidateFit(candidate=candidate))
            continue

        measurements.append(
            _CandidateFit(
                candidate=candidate,
                input_tokens=tokens,
                context_window_limit=limit,
                fits=tokens <= limit * threshold,
            )
        )

    return measurements


def _nonnegative_number(value: Any) -> float:
    """Return a finite non-negative number, raising for invalid provider metadata."""
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"expected a finite non-negative number, got {value!r}")
    return number
