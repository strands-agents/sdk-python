"""File-based session manager for local filesystem storage."""

import json
import os
import shutil
import tempfile
from typing import Any, Optional, cast

from ..types.exceptions import SessionException
from ..types.session import Session, SessionAgent, SessionMessage
from .agent_session_manager import AgentSessionManager
from .session_repository import SessionRepository

SESSION_PREFIX = "session_"
AGENT_PREFIX = "agent_"
MESSAGE_PREFIX = "message_"


class FileSessionManager(AgentSessionManager, SessionRepository):
    """File-based session manager for local filesystem storage."""

    def __init__(self, session_id: str, storage_dir: Optional[str] = None):
        """Initialize FileSession with filesystem storage.

        Args:
            session_id: ID for the session
            storage_dir: Directory for local filesystem storage (defaults to temp dir)
        """
        self.storage_dir = storage_dir or os.path.join(tempfile.gettempdir(), "strands/sessions")
        os.makedirs(self.storage_dir, exist_ok=True)

        super().__init__(session_id=session_id, session_repository=self)

    def _get_session_path(self, session_id: str) -> str:
        """Get session directory path."""
        return os.path.join(self.storage_dir, f"{SESSION_PREFIX}{session_id}")

    def _get_agent_path(self, session_id: str, agent_id: str) -> str:
        """Get agent directory path."""
        session_path = self._get_session_path(session_id)
        return os.path.join(session_path, "agents", f"{AGENT_PREFIX}{agent_id}")

    def _get_message_path(self, session_id: str, agent_id: str, message_id: str) -> str:
        """Get message file path."""
        agent_path = self._get_agent_path(session_id, agent_id)
        return os.path.join(agent_path, "messages", f"{MESSAGE_PREFIX}{message_id}.json")

    def _read_file(self, path: str) -> dict[str, Any]:
        """Read JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cast(dict[str, Any], json.load(f))
        except json.JSONDecodeError as e:
            raise SessionException(f"Invalid JSON in file {path}: {e}") from e

    def _write_file(self, path: str, data: dict[str, Any]) -> None:
        """Write JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_session(self, session: Session) -> Session:
        """Create a new session."""
        session_dir = self._get_session_path(session["session_id"])
        if os.path.exists(session_dir):
            raise SessionException(f"Session {session['session_id']} already exists")

        # Create directory structure
        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(os.path.join(session_dir, "agents"), exist_ok=True)

        # Write session file
        session_file = os.path.join(session_dir, "session.json")
        session_dict = cast(dict, session)
        self._write_file(session_file, session_dict)

        return session

    def read_session(self, session_id: str) -> Optional[Session]:
        """Read session data."""
        session_file = os.path.join(self._get_session_path(session_id), "session.json")
        if not os.path.exists(session_file):
            return None

        session_data = self._read_file(session_file)
        return Session(**session_data)  # type: ignore

    def create_agent(self, session_id: str, session_agent: SessionAgent) -> None:
        """Create a new agent in the session."""
        agent_id = session_agent["agent_id"]

        agent_dir = self._get_agent_path(session_id, agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        os.makedirs(os.path.join(agent_dir, "messages"), exist_ok=True)

        agent_file = os.path.join(agent_dir, "agent.json")
        session_data = cast(dict, session_agent)
        self._write_file(agent_file, session_data)

    def delete_session(self, session_id: str) -> None:
        """Delete session and all associated data."""
        session_dir = self._get_session_path(session_id)
        if not os.path.exists(session_dir):
            raise SessionException(f"Session {session_id} does not exist")

        shutil.rmtree(session_dir)

    def read_agent(self, session_id: str, agent_id: str) -> Optional[SessionAgent]:
        """Read agent data."""
        agent_file = os.path.join(self._get_agent_path(session_id, agent_id), "agent.json")
        if not os.path.exists(agent_file):
            return None

        agent_data = self._read_file(agent_file)
        return SessionAgent(**agent_data)  # type: ignore

    def update_agent(self, session_id: str, session_agent: SessionAgent) -> None:
        """Update agent data."""
        agent_id = session_agent["agent_id"]
        agent_file = os.path.join(self._get_agent_path(session_id, agent_id), "agent.json")
        agent_dict = cast(dict, session_agent)
        self._write_file(agent_file, agent_dict)

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        """Create a new message for the agent."""
        message_file = self._get_message_path(session_id, agent_id, session_message["message_id"])
        session_dict = cast(dict, session_message)
        self._write_file(message_file, session_dict)

    def read_message(self, session_id: str, agent_id: str, message_id: str) -> Optional[SessionMessage]:
        """Read message data."""
        message_file = self._get_message_path(session_id, agent_id, message_id)
        if not os.path.exists(message_file):
            return None
        message_data = self._read_file(message_file)
        return SessionMessage(**message_data)  # type: ignore

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        """Update message data."""
        message_id = session_message["message_id"]
        message_file = self._get_message_path(session_id, agent_id, message_id)
        message_dict = cast(dict, session_message)
        self._write_file(message_file, message_dict)

    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> list[SessionMessage]:
        """List messages for an agent with pagination."""
        messages_dir = os.path.join(self._get_agent_path(session_id, agent_id), "messages")
        if not os.path.exists(messages_dir):
            raise SessionException("messages directory missing from agent: %s in session %s", agent_id, session_id)

        # Get all message files and sort by creation time (newest first)
        message_files = []
        for filename in os.listdir(messages_dir):
            if filename.startswith(MESSAGE_PREFIX) and filename.endswith(".json"):
                file_path = os.path.join(messages_dir, filename)
                message_files.append((file_path, os.path.getctime(file_path)))

        # Sort by creation time (newest first)
        message_files.sort(key=lambda x: x[1], reverse=True)

        # Apply pagination
        if limit is not None:
            message_files = message_files[offset : offset + limit]
        else:
            message_files = message_files[offset:]

        # Read message data
        messages: list[SessionMessage] = []
        for file_path, _ in message_files:
            if not os.path.exists(file_path):
                continue
            message_data = self._read_file(file_path)
            messages.append(SessionMessage(**message_data))  # type: ignore

        return messages
