"""Agent execution state machine.

This module provides a formal state machine representation of the agent execution
lifecycle. Making states explicit enables:

- **Durable Agents**: Serialize/restore agent state at safe checkpoint states
  (``INTERRUPTED``, ``COMPLETED``) to survive process restarts or failures.
- **Observability**: Inspect ``agent.state_machine.state`` at any point to know
  exactly which execution phase the agent is in.
- **Hook integration**: React to state changes via ``AgentStateTransitionEvent``
  to implement custom logic at lifecycle boundaries.

Example::

    from strands import Agent
    from strands.agent.state_machine import AgentExecutionState

    agent = Agent()

    # Observe current state
    print(agent.state_machine.state)  # AgentExecutionState.IDLE

    # Listen for transitions
    def on_transition(old: AgentExecutionState, new: AgentExecutionState) -> None:
        print(f"Agent moved from {old.value!r} -> {new.value!r}")

    agent.state_machine.add_listener(on_transition)

    # Serialize for durable checkpointing
    snapshot = agent.state_machine.to_dict()
    # ... store snapshot ...
    agent.state_machine = AgentStateMachine.from_dict(snapshot)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AgentExecutionState(str, Enum):
    """Formal states in the agent execution lifecycle.

    The state machine follows this high-level flow::

        IDLE
          │  (invocation begins, lock acquired)
          ▼
        INITIALIZING
          │  (prompt converted, messages appended)
          ▼
        MODEL_CALL ◄──────────────────────────────────────────┐
          │  (model response received)                         │
          ▼                                                     │
        TOOL_EXECUTION ─── (tools done, recurse) ─────────────┘
          │  (interrupt raised by tool)
          ▼
        INTERRUPTED  (checkpoint — safe to serialize here)
          │  (resume called on next invocation)
          └──► INITIALIZING

        MODEL_CALL / TOOL_EXECUTION
          │  (normal end_turn / stop_sequence)
          ▼
        COMPLETED  (checkpoint — safe to serialize here)
          │  (AfterInvocationEvent.resume set → loop again)
          ├──► INITIALIZING
          └──► IDLE

        MODEL_CALL / TOOL_EXECUTION / INITIALIZING
          │  (agent.cancel() called)
          ▼
        CANCELLED
          └──► IDLE

        Any running state
          │  (unhandled exception)
          ▼
        ERROR
          └──► IDLE
    """

    IDLE = "idle"
    """Agent exists but no invocation is in progress."""

    INITIALIZING = "initializing"
    """Invocation lock acquired; prompt is being converted to messages."""

    MODEL_CALL = "model_call"
    """Model API call in flight; streaming response chunks."""

    TOOL_EXECUTION = "tool_execution"
    """One or more tools requested by the model are being executed."""

    INTERRUPTED = "interrupted"
    """Paused for human-in-the-loop input. Safe checkpoint state."""

    COMPLETED = "completed"
    """Invocation finished successfully. Safe checkpoint state."""

    CANCELLED = "cancelled"
    """Invocation was cancelled via ``agent.cancel()``."""

    ERROR = "error"
    """Unhandled exception occurred during execution."""


# ---------------------------------------------------------------------------
# Valid transitions
# ---------------------------------------------------------------------------

_TRANSITIONS: dict[AgentExecutionState, frozenset[AgentExecutionState]] = {
    AgentExecutionState.IDLE: frozenset(
        [AgentExecutionState.INITIALIZING]
    ),
    AgentExecutionState.INITIALIZING: frozenset(
        [
            AgentExecutionState.MODEL_CALL,
            AgentExecutionState.TOOL_EXECUTION,  # resuming from interrupt skips model call
            AgentExecutionState.COMPLETED,       # short-circuit: mocked/overridden event loops
            AgentExecutionState.INTERRUPTED,     # short-circuit: mocked/overridden event loops
            AgentExecutionState.CANCELLED,
            AgentExecutionState.ERROR,
        ]
    ),
    AgentExecutionState.MODEL_CALL: frozenset(
        [
            AgentExecutionState.TOOL_EXECUTION,
            AgentExecutionState.INITIALIZING,   # context window overflow retry
            AgentExecutionState.COMPLETED,
            AgentExecutionState.CANCELLED,
            AgentExecutionState.ERROR,
        ]
    ),
    AgentExecutionState.TOOL_EXECUTION: frozenset(
        [
            AgentExecutionState.MODEL_CALL,  # recurse for next turn
            AgentExecutionState.INTERRUPTED,
            AgentExecutionState.COMPLETED,
            AgentExecutionState.CANCELLED,
            AgentExecutionState.ERROR,
        ]
    ),
    AgentExecutionState.INTERRUPTED: frozenset(
        [
            AgentExecutionState.IDLE,        # waiting for external resume
            AgentExecutionState.INITIALIZING, # resume called in same session
        ]
    ),
    AgentExecutionState.COMPLETED: frozenset(
        [
            AgentExecutionState.IDLE,
            AgentExecutionState.INITIALIZING,  # AfterInvocationEvent.resume set
        ]
    ),
    AgentExecutionState.CANCELLED: frozenset(
        [AgentExecutionState.IDLE]
    ),
    AgentExecutionState.ERROR: frozenset(
        [AgentExecutionState.IDLE]
    ),
}

# States where it is safe to snapshot agent data for durability
CHECKPOINT_STATES: frozenset[AgentExecutionState] = frozenset(
    [AgentExecutionState.IDLE, AgentExecutionState.INTERRUPTED, AgentExecutionState.COMPLETED]
)


class InvalidStateTransitionError(Exception):
    """Raised when an invalid state transition is attempted.

    Attributes:
        from_state: The state the machine was in.
        to_state: The state the transition was attempted to.
        allowed: The set of valid target states from ``from_state``.
    """

    def __init__(
        self,
        from_state: AgentExecutionState,
        to_state: AgentExecutionState,
        allowed: frozenset[AgentExecutionState],
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.allowed = allowed
        super().__init__(
            f"Invalid state transition: {from_state.value!r} -> {to_state.value!r}. "
            f"Allowed targets: {sorted(s.value for s in allowed)}"
        )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

TransitionListener = Callable[[AgentExecutionState, AgentExecutionState], None]
"""Callable invoked synchronously on every successful state transition.

Args:
    old_state: The state before the transition.
    new_state: The state after the transition.
"""


@dataclass
class AgentStateMachine:
    """Tracks the current execution state of an :class:`~strands.agent.Agent`.

    The machine validates every transition against the allowed transition table
    and notifies registered listeners synchronously before returning.

    Attributes:
        state: The current execution state.
    """

    state: AgentExecutionState = AgentExecutionState.IDLE
    _listeners: list[TransitionListener] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def transition(self, new_state: AgentExecutionState) -> None:
        """Transition to *new_state*, validating the transition first.

        Args:
            new_state: The target state.

        Raises:
            InvalidStateTransitionError: If the transition from the current
                state to *new_state* is not permitted.
        """
        allowed = _TRANSITIONS.get(self.state, frozenset())
        if new_state not in allowed:
            raise InvalidStateTransitionError(self.state, new_state, allowed)

        old_state = self.state
        self.state = new_state
        logger.debug("state_machine | %s -> %s", old_state.value, new_state.value)

        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except Exception:
                logger.exception(
                    "state_machine | listener raised an exception during transition %s -> %s",
                    old_state.value,
                    new_state.value,
                )

    def reset(self) -> None:
        """Force-reset the state machine to IDLE, bypassing transition validation.

        This is intended exclusively for cleanup/error-recovery paths (e.g., the
        ``finally`` block of an invocation) where the agent must return to a usable
        state regardless of which phase it was in when an exception occurred.
        """
        old_state = self.state
        self.state = AgentExecutionState.IDLE
        if old_state != AgentExecutionState.IDLE:
            logger.debug("state_machine | reset %s -> idle", old_state.value)
            for listener in self._listeners:
                try:
                    listener(old_state, AgentExecutionState.IDLE)
                except Exception:
                    logger.exception(
                        "state_machine | listener raised an exception during reset from %s",
                        old_state.value,
                    )

    def try_transition(self, new_state: AgentExecutionState) -> bool:
        """Attempt a transition, returning *False* instead of raising on failure.

        Useful for "best-effort" transitions in error paths where the exact
        current state may be uncertain.

        Args:
            new_state: The target state.

        Returns:
            True if the transition succeeded, False otherwise.
        """
        try:
            self.transition(new_state)
            return True
        except InvalidStateTransitionError:
            logger.debug(
                "state_machine | ignoring invalid transition %s -> %s",
                self.state.value,
                new_state.value,
            )
            return False

    # ------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------

    def add_listener(self, listener: TransitionListener) -> None:
        """Register a callable to be invoked on every state transition.

        Args:
            listener: Callable with signature ``(old_state, new_state) -> None``.
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: TransitionListener) -> None:
        """Remove a previously registered listener.

        Args:
            listener: The listener to remove.
        """
        self._listeners.remove(listener)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def is_checkpoint(self) -> bool:
        """True if the current state is safe for durable snapshots."""
        return self.state in CHECKPOINT_STATES

    @property
    def is_running(self) -> bool:
        """True if an invocation is currently in progress."""
        return self.state not in (
            AgentExecutionState.IDLE,
            AgentExecutionState.INTERRUPTED,
            AgentExecutionState.COMPLETED,
            AgentExecutionState.CANCELLED,
            AgentExecutionState.ERROR,
        )

    # ------------------------------------------------------------------
    # Serialization (for durable agents)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state machine to a JSON-safe dict.

        Only :attr:`state` is serialized; listeners are not persisted.

        Returns:
            ``{"state": "<state-value>"}``
        """
        return {"state": self.state.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentStateMachine":
        """Restore a state machine from a serialized dict.

        Args:
            data: Dict previously produced by :meth:`to_dict`.

        Returns:
            A new :class:`AgentStateMachine` in the restored state.
        """
        return cls(state=AgentExecutionState(data["state"]))
