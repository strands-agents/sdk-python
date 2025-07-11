"""Data models for session management."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, TypedDict, cast
from uuid import uuid4

from ..agent.agent import Agent
from .content import Message


class SessionType(str, Enum):
    """Enumeration of session types."""

    AGENT = "AGENT"


class SessionMessage(Message):
    """Message within a SessionAgent."""

    message_id: str
    created_at: str
    updated_at: str


def session_message_to_message(session_message: SessionMessage) -> Message:
    """Convert to a message."""
    message_dict = {key: value for key, value in session_message.items() if key in Message.__annotations__.keys()}
    return cast(Message, message_dict)


def session_message_from_message(message: Message) -> SessionMessage:
    """Convert from a Message."""
    now = datetime.now(timezone.utc).isoformat()
    message_dict = dict(message)
    return SessionMessage(message_id=str(uuid4()), created_at=now, updated_at=now, **message_dict)  # type: ignore


class SessionAgent(TypedDict):
    """Agent within a Session."""

    agent_id: str
    event_loop_metrics: Dict[str, Any]
    state: Dict[str, Any]
    created_at: str
    updated_at: str


def session_agent_from_agent(agent: Agent) -> SessionAgent:
    """Convert an Agent to a SessionAgent."""
    now = datetime.now(timezone.utc).isoformat()
    if agent.agent_id is None:
        raise ValueError("agent_id needs to be defined.")
    return SessionAgent(
        agent_id=agent.agent_id,
        event_loop_metrics=agent.event_loop_metrics.to_dict(),
        state=agent.state.get(),
        created_at=now,
        updated_at=now,
    )


class Session(TypedDict):
    """Session data model."""

    session_id: str
    session_type: SessionType
    created_at: str
    updated_at: str


def create_session(session_id: str, session_type: SessionType) -> Session:
    """Create a new Session with the given ID and type."""
    now = datetime.now(timezone.utc).isoformat()
    return Session(
        session_id=session_id,
        session_type=session_type,
        created_at=now,
        updated_at=now,
    )
