"""Data models for session management with lazy loading and pagination."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, cast
from uuid import uuid4

from ..types.content import ContentBlock, Message

if TYPE_CHECKING:
    pass


class SessionType(str, Enum):
    """Enumeration of session types."""

    AGENT = "AGENT"


@dataclass
class SessionMessage:
    """Message within a SessionAgent."""

    role: str
    content: List[ContentBlock]
    message_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_message(self) -> Message:
        """Convert to a message."""
        message_dict = {key: value for key, value in self.to_dict().items() if key in Message.__annotations__.keys()}
        return cast(Message, message_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMessage":
        """Create a session message from a dictionary."""
        return cls(**data)


@dataclass
class SessionAgent:
    """Agent within a Session."""

    agent_id: str
    session_id: str
    event_loop_metrics: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionAgent":
        """Create a session agent from a dictionary."""
        return cls(**data)


@dataclass
class Session:
    """Session data model."""

    session_id: str
    session_type: SessionType
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create a session from a dictionary."""
        return cls(**data)
