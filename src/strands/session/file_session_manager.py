"""File-based session manager for local filesystem storage."""

import json
import logging
import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Optional, cast

from .. import _identifier
from ..types.exceptions import SessionException
from ..types.session import Session, SessionAgent, SessionMessage
from .repository_session_manager import RepositorySessionManager
from .session_repository import SessionRepository

if TYPE_CHECKING:
    from ..multiagent.base import MultiAgentBase

logger = logging.getLogger(__name__)

SESSION_PREFIX = "session_"
AGENT_PREFIX = "agent_"
MESSAGE_PREFIX = "message_"
MULTI_AGENT_PREFIX = "multi_agent_"


class FileSessionManager(RepositorySessionManager, SessionRepository):
    """File-based session manager for local filesystem storage.

    Creates the following filesystem structure for the session storage:
    ```bash
    /<sessions_dir>/
    └── session_<session_id>/
        ├── session.json                # Session metadata
        └── agents/
            └── agent_<agent_id>/
                ├── agent.json          # Agent metadata
                └── messages/
                    ├── message_<id1>.json
                    └── message_<id2>.json
    ```
    """

    def __init__(
        self,
        session_id: str,
        storage_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize FileSession with filesystem storage.

        Args:
            session_id: ID for the session.
                ID is not allowed to contain path separators (e.g., a/b).
            storage_dir: Directory for local filesystem storage (defaults to temp dir).
            **kwargs: Additional keyword arguments for future extensibility.
        """
        self.storage_dir = storage_dir or os.path.join(tempfile.gettempdir(), "strands/sessions")
        os.makedirs(self.storage_dir, exist_ok=True)

        super().__init__(session_id=session_id, session_repository=self)

    def _get_session_path(self, session_id: str) -> str:
        """Get session directory path.

        Args:
            session_id: ID for the session.

        Raises:
            ValueError: If session id contains a path separator.
        """
        session_id = _identifier.validate(session_id, _identifier.Identifier.SESSION)
        return os.path.join(self.storage_dir, f"{SESSION_PREFIX}{session_id}")

    def _get_agent_path(self, session_id: str, agent_id: str) -> str:
        """Get agent directory path.

        Args:
            session_id: ID for the session.
            agent_id: ID for the agent.

        Raises:
            ValueError: If session id or agent id contains a path separator.
        """
        session_path = self._get_session_path(session_id)
        agent_id = _identifier.validate(agent_id, _identifier.Identifier.AGENT)
        return os.path.join(session_path, "agents", f"{AGENT_PREFIX}{agent_id}")

    def _get_message_path(self, session_id: str, agent_id: str, message_id: int) -> str:
        """Get message file path.

        Args:
            session_id: ID of the session
            agent_id: ID of the agent
            message_id: Index of the message
        Returns:
            The filename for the message

        Raises:
            ValueError: If message_id is not an integer.
        """
        if not isinstance(message_id, int):
            raise ValueError(f"message_id=<{message_id}> | message id must be an integer")

        agent_path = self._get_agent_path(session_id, agent_id)
        return os.path.join(agent_path, "messages", f"{MESSAGE_PREFIX}{message_id}.json")

    def _read_file(self, path: str) -> dict[str, Any]:
        """Read JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cast(dict[str, Any], json.load(f))
        except json.JSONDecodeError as e:
            raise SessionException(f"Invalid JSON in file {path}: {str(e)}") from e

    def _write_file(self, path: str, data: dict[str, Any]) -> None:
        """Write JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # This automic write ensure the completeness of session files in both single agent/ multi agents
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    def create_session(self, session: Session, **kwargs: Any) -> Session:
        """Create a new session."""
        session_dir = self._get_session_path(session.session_id)
        if os.path.exists(session_dir):
            raise SessionException(f"Session {session.session_id} already exists")

        # Create directory structure
        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(os.path.join(session_dir, "agents"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "multi_agents"), exist_ok=True)

        # Write session file
        session_file = os.path.join(session_dir, "session.json")
        session_dict = session.to_dict()
        self._write_file(session_file, session_dict)

        return session

    def read_session(self, session_id: str, **kwargs: Any) -> Optional[Session]:
        """Read session data."""
        session_file = os.path.join(self._get_session_path(session_id), "session.json")
        if not os.path.exists(session_file):
            return None

        session_data = self._read_file(session_file)
        return Session.from_dict(session_data)

    def delete_session(self, session_id: str, **kwargs: Any) -> None:
        """Delete session and all associated data."""
        session_dir = self._get_session_path(session_id)
        if not os.path.exists(session_dir):
            raise SessionException(f"Session {session_id} does not exist")

        shutil.rmtree(session_dir)

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Create a new agent in the session."""
        agent_id = session_agent.agent_id

        agent_dir = self._get_agent_path(session_id, agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        os.makedirs(os.path.join(agent_dir, "messages"), exist_ok=True)

        agent_file = os.path.join(agent_dir, "agent.json")
        session_data = session_agent.to_dict()
        self._write_file(agent_file, session_data)

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> Optional[SessionAgent]:
        """Read agent data."""
        agent_file = os.path.join(self._get_agent_path(session_id, agent_id), "agent.json")
        if not os.path.exists(agent_file):
            return None

        agent_data = self._read_file(agent_file)
        return SessionAgent.from_dict(agent_data)

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Update agent data."""
        agent_id = session_agent.agent_id
        previous_agent = self.read_agent(session_id=session_id, agent_id=agent_id)
        if previous_agent is None:
            raise SessionException(f"Agent {agent_id} in session {session_id} does not exist")

        session_agent.created_at = previous_agent.created_at
        agent_file = os.path.join(self._get_agent_path(session_id, agent_id), "agent.json")
        self._write_file(agent_file, session_agent.to_dict())

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Create a new message for the agent."""
        message_file = self._get_message_path(
            session_id,
            agent_id,
            session_message.message_id,
        )
        session_dict = session_message.to_dict()
        self._write_file(message_file, session_dict)

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs: Any) -> Optional[SessionMessage]:
        """Read message data."""
        message_path = self._get_message_path(session_id, agent_id, message_id)
        if not os.path.exists(message_path):
            return None
        message_data = self._read_file(message_path)
        return SessionMessage.from_dict(message_data)

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Update message data."""
        message_id = session_message.message_id
        previous_message = self.read_message(session_id=session_id, agent_id=agent_id, message_id=message_id)
        if previous_message is None:
            raise SessionException(f"Message {message_id} does not exist")

        # Preserve the original created_at timestamp
        session_message.created_at = previous_message.created_at
        message_file = self._get_message_path(session_id, agent_id, message_id)
        self._write_file(message_file, session_message.to_dict())

    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0, **kwargs: Any
    ) -> list[SessionMessage]:
        """List messages for an agent with pagination."""
        messages_dir = os.path.join(self._get_agent_path(session_id, agent_id), "messages")
        if not os.path.exists(messages_dir):
            raise SessionException(f"Messages directory missing from agent: {agent_id} in session {session_id}")

        # Read all message files, and record the index
        message_index_files: list[tuple[int, str]] = []
        for filename in os.listdir(messages_dir):
            if filename.startswith(MESSAGE_PREFIX) and filename.endswith(".json"):
                # Extract index from message_<index>.json format
                index = int(filename[len(MESSAGE_PREFIX) : -5])  # Remove prefix and .json suffix
                message_index_files.append((index, filename))

        # Sort by index and extract just the filenames
        message_files = [f for _, f in sorted(message_index_files)]

        # Apply pagination to filenames
        if limit is not None:
            message_files = message_files[offset : offset + limit]
        else:
            message_files = message_files[offset:]

        # Load only the message files
        messages: list[SessionMessage] = []
        for filename in message_files:
            file_path = os.path.join(messages_dir, filename)
            message_data = self._read_file(file_path)
            messages.append(SessionMessage.from_dict(message_data))

        return messages

    def _get_multi_agent_path(self, session_id: str, multi_agent_id: str) -> str:
        """Get multi-agent state file path."""
        session_path = self._get_session_path(session_id)
        multi_agent_id = _identifier.validate(multi_agent_id, _identifier.Identifier.AGENT)
        return os.path.join(session_path, "multi_agents", f"{MULTI_AGENT_PREFIX}{multi_agent_id}")

    def create_multi_agent(self, session_id: str, multi_agent: "MultiAgentBase", **kwargs: Any) -> None:
        """Create a new multiagent state in the session."""
        multi_agent_id = multi_agent.id
        multi_agent_dir = self._get_multi_agent_path(session_id, multi_agent_id)
        os.makedirs(multi_agent_dir, exist_ok=True)

        multi_agent_file = os.path.join(multi_agent_dir, "multi_agent.json")
        session_data = multi_agent.serialize_state()
        self._write_file(multi_agent_file, session_data)

    def read_multi_agent(self, session_id: str, multi_agent_id: str, **kwargs: Any) -> Optional[dict[str, Any]]:
        """Read multi-agent state from filesystem."""
        multi_agent_file = os.path.join(self._get_multi_agent_path(session_id, multi_agent_id), "multi_agent.json")
        if not os.path.exists(multi_agent_file):
            return None
        return self._read_file(multi_agent_file)

    def update_multi_agent(self, session_id: str, multi_agent: "MultiAgentBase", **kwargs: Any) -> None:
        """Update multi-agent state from filesystem."""
        multi_agent_state = multi_agent.serialize_state()
        previous_multi_agent_state = self.read_multi_agent(session_id=session_id, multi_agent_id=multi_agent.id)
        if previous_multi_agent_state is None:
            raise SessionException(f"MultiAgent state {multi_agent.id} in session {session_id} does not exist")

        multi_agent_file = os.path.join(self._get_multi_agent_path(session_id, multi_agent.id), "multi_agent.json")
        self._write_file(multi_agent_file, multi_agent_state)
