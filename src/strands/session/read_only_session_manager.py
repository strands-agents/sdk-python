"""Read-only session manager wrapper."""

import logging
from typing import TYPE_CHECKING, Any

from ..hooks.registry import HookRegistry
from ..types.content import Message
from .session_manager import SessionManager

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from ..experimental.bidi.agent.agent import BidiAgent
    from ..multiagent.base import MultiAgentBase

logger = logging.getLogger(__name__)


class ReadOnlySessionManager(SessionManager):
    """A wrapper that delegates read operations to an inner session manager and no-ops all writes.

    Read-only enforcement happens at the SessionManager level — all write methods are no-ops regardless
    of whether they are called by the Agent, custom hooks, or user code.

    Attribute access is forwarded to the inner session manager, so properties like ``session_id``
    and ``bucket`` are available directly on the wrapper.

    Note:
        The wrapper protects writes because the Agent holds a reference to this wrapper instance.
        Bypassing the wrapper by obtaining the inner session manager and passing it directly to an
        Agent will lose read-only protection.

    Usage::

        from strands import Agent
        from strands.session import ReadOnlySessionManager, S3SessionManager

        inner = S3SessionManager(session_id="tenant-123", bucket="my-bucket")
        agent = Agent(session_manager=ReadOnlySessionManager(inner))
    """

    def __init__(self, inner: SessionManager) -> None:
        """Initialize the ReadOnlySessionManager.

        Args:
            inner: The session manager to delegate read operations to.
        """
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the inner session manager."""
        return getattr(self._inner, name)

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks preserving the inner session manager's custom hooks.

        Patches the inner's write methods to point to this wrapper's no-ops, then
        delegates to the inner's register_hooks. This preserves any custom read-path
        hooks (e.g., LTM retrieval) while ensuring write-path lambdas resolve to
        no-ops at call time via Python's late binding.
        """
        write_methods = [
            "append_message", "sync_agent", "redact_latest_message",
            "sync_multi_agent", "append_bidi_message", "sync_bidi_agent",
        ]
        for name in write_methods:
            setattr(self._inner, name, getattr(self, name))

        self._inner.register_hooks(registry, **kwargs)

    def initialize(self, agent: "Agent", **kwargs: Any) -> None:
        """Delegate to inner session manager to restore agent state."""
        self._inner.initialize(agent, **kwargs)

    def initialize_multi_agent(self, source: "MultiAgentBase", **kwargs: Any) -> None:
        """Delegate to inner session manager to restore multi-agent state."""
        self._inner.initialize_multi_agent(source, **kwargs)

    def initialize_bidi_agent(self, agent: "BidiAgent", **kwargs: Any) -> None:
        """Delegate to inner session manager to restore bidi agent state."""
        self._inner.initialize_bidi_agent(agent, **kwargs)

    def append_message(self, message: Message, agent: "Agent", **kwargs: Any) -> None:
        """No-op: read-only mode skips message persistence."""
        logger.debug("skipping append_message in read-only mode")

    def redact_latest_message(self, redact_message: Message, agent: "Agent", **kwargs: Any) -> None:
        """No-op: read-only mode skips message redaction persistence."""
        logger.debug("skipping redact_latest_message in read-only mode")

    def sync_agent(self, agent: "Agent", **kwargs: Any) -> None:
        """No-op: read-only mode skips agent sync."""
        logger.debug("skipping sync_agent in read-only mode")

    def sync_multi_agent(self, source: "MultiAgentBase", **kwargs: Any) -> None:
        """No-op: read-only mode skips multi-agent sync."""
        logger.debug("skipping sync_multi_agent in read-only mode")

    def append_bidi_message(self, message: Message, agent: "BidiAgent", **kwargs: Any) -> None:
        """No-op: read-only mode skips bidi message persistence."""
        logger.debug("skipping append_bidi_message in read-only mode")

    def sync_bidi_agent(self, agent: "BidiAgent", **kwargs: Any) -> None:
        """No-op: read-only mode skips bidi agent sync."""
        logger.debug("skipping sync_bidi_agent in read-only mode")
