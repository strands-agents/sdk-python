"""SQLAlchemy models for DatabaseSessionManager."""

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKeyConstraint, Integer, String
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class StrandsSession(Base):
    """Top-level session table.

    Stores session metadata. Can be updated (e.g., updated_at timestamp).
    """

    __tablename__ = "strands_sessions"

    # Primary key
    session_id = Column(String(255), primary_key=True)

    # Metadata
    session_type = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    agents = relationship(
        "StrandsAgent",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class StrandsAgent(Base):
    """Per-session agent table.

    Stores agent state and conversation manager state. Can be updated.
    """

    __tablename__ = "strands_agents"

    # Composite primary key
    session_id = Column(String(255), primary_key=True)
    agent_id = Column(String(255), primary_key=True)

    # State (stored as JSON)
    state = Column(JSON, nullable=False)
    conversation_manager_state = Column(JSON, nullable=False)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Foreign key to session
    __table_args__ = (ForeignKeyConstraint(["session_id"], ["strands_sessions.session_id"], ondelete="CASCADE"),)

    # Relationships
    session = relationship("StrandsSession", back_populates="agents")
    messages = relationship(
        "StrandsMessage",
        back_populates="agent",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class StrandsMessage(Base):
    """Append-only message table.

    Stores conversation messages. Should only INSERT, not UPDATE (except redaction).
    """

    __tablename__ = "strands_messages"

    # Composite primary key
    session_id = Column(String(255), primary_key=True)
    agent_id = Column(String(255), primary_key=True)
    message_id = Column(Integer, primary_key=True)

    # Message content (stored as JSON)
    message = Column(JSON, nullable=False)
    redact_message = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Foreign key to agent
    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "agent_id"],
            ["strands_agents.session_id", "strands_agents.agent_id"],
            ondelete="CASCADE",
        ),
    )

    # Relationships
    agent = relationship("StrandsAgent", back_populates="messages")
