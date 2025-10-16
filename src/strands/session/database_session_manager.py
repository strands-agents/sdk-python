"""Database-based session manager for PostgreSQL/MySQL/SQLite storage."""

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, List, Optional, cast

from .. import _identifier
from ..types.exceptions import SessionException
from ..types.session import Session, SessionAgent, SessionMessage
from .database_models import Base, StrandsAgent, StrandsMessage, StrandsSession
from .repository_session_manager import RepositorySessionManager
from .session_repository import SessionRepository

logger = logging.getLogger(__name__)


class DatabaseSessionManager(RepositorySessionManager, SessionRepository):
    """Database-based session manager using SQLAlchemy.

    Supports PostgreSQL, MySQL, and SQLite databases for session persistence.

    Examples:
        # With connection string (creates own engine)
        >>> manager = DatabaseSessionManager(
        ...     session_id="user-123",
        ...     connection_string="postgresql://user:pass@localhost/mydb"
        ... )

        # With shared engine (recommended for FastAPI)
        >>> from sqlalchemy import create_engine
        >>> engine = create_engine("postgresql://...", pool_size=20)
        >>> manager = DatabaseSessionManager(
        ...     session_id="user-123",
        ...     engine=engine
        ... )
    """

    def __init__(
        self,
        session_id: str,
        connection_string: Optional[str] = None,
        engine: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize DatabaseSessionManager.

        Args:
            session_id: ID for the session (required).
            connection_string: Database connection string (PostgreSQL, MySQL, or SQLite).
                Required if engine is None. Example: "postgresql://user:pass@localhost:5432/dbname"
            engine: Pre-configured SQLAlchemy Engine. If provided, connection_string is ignored.
                Use this to share a connection pool across multiple session managers.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            ValueError: If neither engine nor connection_string is provided.
            ImportError: If SQLAlchemy is not installed.
        """
        # Validation
        if engine is None and connection_string is None:
            raise ValueError(
                "Must provide either 'engine' or 'connection_string'. "
                "Example: DatabaseSessionManager(session_id='test', "
                "connection_string='postgresql://localhost/db')"
            )

        # Import SQLAlchemy (optional dependency)
        try:
            from sqlalchemy import create_engine as sqlalchemy_create_engine
            from sqlalchemy.orm import sessionmaker
        except ImportError as e:
            raise ImportError(
                "DatabaseSessionManager requires SQLAlchemy. Install with: pip install 'strands-agents[database]'"
            ) from e

        # Create or use engine
        if engine is None:
            # At this point, connection_string is guaranteed to be non-None due to validation above
            assert connection_string is not None
            self.engine = sqlalchemy_create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            self._owns_engine = True
        else:
            self.engine = engine
            self._owns_engine = False

        # Create session factory (thread-safe)
        self.Session = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

        # Initialize parent
        super().__init__(session_id=session_id, session_repository=self)

    def __del__(self) -> None:
        """Clean up engine if we own it."""
        if hasattr(self, "_owns_engine") and self._owns_engine and hasattr(self, "engine"):
            self.engine.dispose()

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        """Validate session ID for security."""
        return _identifier.validate(session_id, _identifier.Identifier.SESSION)

    @staticmethod
    def _validate_agent_id(agent_id: str) -> str:
        """Validate agent ID for security."""
        return _identifier.validate(agent_id, _identifier.Identifier.AGENT)

    @staticmethod
    def _validate_ids(session_id: str, agent_id: str) -> tuple[str, str]:
        """Validate both session ID and agent ID for security."""
        return (
            DatabaseSessionManager._validate_session_id(session_id=session_id),
            DatabaseSessionManager._validate_agent_id(agent_id=agent_id),
        )

    @contextmanager
    def _db_session(self, error_prefix: Optional[str] = None, auto_commit: bool = True) -> Iterator[Any]:
        """Context manager for database sessions with automatic cleanup and rollback on error.

        Args:
            error_prefix: Optional prefix for error messages. If provided, non-SessionException
                         errors will be wrapped in SessionException with this prefix.
            auto_commit: If True, automatically commit on success. Default is True.
        """
        session = self.Session()
        try:
            yield session
            if auto_commit:
                session.commit()
        except SessionException:
            session.rollback()
            raise
        except Exception as e:
            session.rollback()
            if error_prefix:
                raise SessionException(f"{error_prefix}: {e}") from e
            raise
        finally:
            session.close()

    def create_session(self, session: Session, **kwargs: Any) -> Session:
        """Create a new session in the database."""
        session_id = self._validate_session_id(session_id=session.session_id)

        with self._db_session(error_prefix="Failed to create session") as db_session:
            # Check if session already exists
            existing = db_session.query(StrandsSession).filter_by(session_id=session_id).first()
            if existing:
                raise SessionException(f"Session {session_id} already exists")

            # Create new session
            db_row = StrandsSession(
                session_id=session_id,
                session_type=session.session_type,
                created_at=datetime.fromisoformat(session.created_at),
                updated_at=datetime.fromisoformat(session.updated_at),
            )
            db_session.add(db_row)
            return session

    def read_session(self, session_id: str, **kwargs: Any) -> Optional[Session]:
        """Read session data from the database."""
        session_id = self._validate_session_id(session_id=session_id)

        with self._db_session(error_prefix="Failed to read session", auto_commit=False) as db_session:
            db_row = db_session.query(StrandsSession).filter_by(session_id=session_id).first()
            if db_row is None:
                return None

            return Session(
                session_id=db_row.session_id,
                session_type=db_row.session_type,
                created_at=db_row.created_at.isoformat(),
                updated_at=db_row.updated_at.isoformat(),
            )

    def delete_session(self, session_id: str, **kwargs: Any) -> None:
        """Delete session and all associated data from the database."""
        session_id = self._validate_session_id(session_id=session_id)

        with self._db_session(error_prefix="Failed to delete session") as db_session:
            db_row = db_session.query(StrandsSession).filter_by(session_id=session_id).first()
            if db_row is None:
                raise SessionException(f"Session {session_id} does not exist")

            db_session.delete(db_row)

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Create a new agent in the database."""
        session_id, agent_id = self._validate_ids(session_id=session_id, agent_id=session_agent.agent_id)

        with self._db_session(error_prefix="Failed to create agent") as db_session:
            db_row = StrandsAgent(
                session_id=session_id,
                agent_id=agent_id,
                state=session_agent.state,
                conversation_manager_state=session_agent.conversation_manager_state,
                created_at=datetime.fromisoformat(session_agent.created_at),
                updated_at=datetime.fromisoformat(session_agent.updated_at),
            )
            db_session.add(db_row)

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> Optional[SessionAgent]:
        """Read agent data from the database."""
        session_id, agent_id = self._validate_ids(session_id=session_id, agent_id=agent_id)

        with self._db_session(error_prefix="Failed to read agent", auto_commit=False) as db_session:
            db_row = db_session.query(StrandsAgent).filter_by(session_id=session_id, agent_id=agent_id).first()
            if db_row is None:
                return None

            return SessionAgent(
                agent_id=db_row.agent_id,
                state=db_row.state,
                conversation_manager_state=db_row.conversation_manager_state,
                created_at=db_row.created_at.isoformat(),
                updated_at=db_row.updated_at.isoformat(),
            )

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Update agent data in the database."""
        session_id, agent_id = self._validate_ids(session_id=session_id, agent_id=session_agent.agent_id)

        with self._db_session(error_prefix="Failed to update agent") as db_session:
            db_row = db_session.query(StrandsAgent).filter_by(session_id=session_id, agent_id=agent_id).first()
            if db_row is None:
                raise SessionException(f"Agent {agent_id} in session {session_id} does not exist")

            # Update fields (preserving created_at)
            db_row.state = session_agent.state
            db_row.conversation_manager_state = session_agent.conversation_manager_state
            db_row.updated_at = datetime.now(timezone.utc)

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Create a new message in the database."""
        session_id, agent_id = self._validate_ids(session_id=session_id, agent_id=agent_id)

        if not isinstance(session_message.message_id, int):
            raise ValueError(f"message_id=<{session_message.message_id}> | message id must be an integer")

        with self._db_session(error_prefix="Failed to create message") as db_session:
            db_row = StrandsMessage(
                session_id=session_id,
                agent_id=agent_id,
                message_id=session_message.message_id,
                message=cast(dict, session_message.message),
                redact_message=cast(dict, session_message.redact_message) if session_message.redact_message else None,
                created_at=datetime.fromisoformat(session_message.created_at),
                updated_at=datetime.fromisoformat(session_message.updated_at),
            )
            db_session.add(db_row)

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs: Any) -> Optional[SessionMessage]:
        """Read message data from the database."""
        session_id, agent_id = self._validate_ids(session_id=session_id, agent_id=agent_id)

        if not isinstance(message_id, int):
            raise ValueError(f"message_id=<{message_id}> | message id must be an integer")

        with self._db_session(error_prefix="Failed to read message", auto_commit=False) as db_session:
            db_row = (
                db_session.query(StrandsMessage)
                .filter_by(session_id=session_id, agent_id=agent_id, message_id=message_id)
                .first()
            )
            if db_row is None:
                return None

            return SessionMessage(
                message=db_row.message,
                message_id=db_row.message_id,
                redact_message=db_row.redact_message,
                created_at=db_row.created_at.isoformat(),
                updated_at=db_row.updated_at.isoformat(),
            )

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Update message data in the database (typically for redaction)."""
        session_id, agent_id = self._validate_ids(session_id=session_id, agent_id=agent_id)
        message_id = session_message.message_id

        if not isinstance(message_id, int):
            raise ValueError(f"message_id=<{message_id}> | message id must be an integer")

        with self._db_session(error_prefix="Failed to update message") as db_session:
            db_row = (
                db_session.query(StrandsMessage)
                .filter_by(session_id=session_id, agent_id=agent_id, message_id=message_id)
                .first()
            )
            if db_row is None:
                raise SessionException(f"Message {message_id} does not exist")

            # Update redact_message only (preserving created_at)
            db_row.redact_message = (
                cast(dict, session_message.redact_message) if session_message.redact_message else None
            )
            db_row.updated_at = datetime.now(timezone.utc)

    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0, **kwargs: Any
    ) -> List[SessionMessage]:
        """List messages for an agent with pagination."""
        session_id, agent_id = self._validate_ids(session_id=session_id, agent_id=agent_id)

        with self._db_session(error_prefix="Failed to list messages", auto_commit=False) as db_session:
            query = (
                db_session.query(StrandsMessage)
                .filter_by(session_id=session_id, agent_id=agent_id)
                .order_by(StrandsMessage.message_id)
                .offset(offset)
            )

            if limit is not None:
                query = query.limit(limit)

            db_rows = query.all()

            messages: List[SessionMessage] = []
            for db_row in db_rows:
                messages.append(
                    SessionMessage(
                        message=db_row.message,
                        message_id=db_row.message_id,
                        redact_message=db_row.redact_message,
                        created_at=db_row.created_at.isoformat(),
                        updated_at=db_row.updated_at.isoformat(),
                    )
                )

            return messages
