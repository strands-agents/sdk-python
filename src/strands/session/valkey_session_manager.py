"""Valkey-based session manager for Redis-compatible storage."""

import json
import logging
from typing import Any, Dict, List, Optional, Union, cast

import valkey

from ..types.exceptions import SessionException
from ..types.session import Session, SessionAgent, SessionMessage
from .repository_session_manager import RepositorySessionManager
from .session_repository import SessionRepository

logger = logging.getLogger(__name__)

SESSION_PREFIX = "session"
AGENT_PREFIX = "agent"
MESSAGE_PREFIX = "message"


class ValkeySessionManager(RepositorySessionManager, SessionRepository):
    """Valkey-based session manager for Redis-compatible storage.

    Creates the following key structure for the session storage:
    ```
    session:<session_id>                           # Session metadata (JSON)
    session:<session_id>:agent:<agent_id>        # Agent metadata (JSON)
    session:<session_id>:agent:<agent_id>:message:<message_id>  # Message data (JSON)
    ```
    """

    def __init__(self, session_id: str, client: Union[valkey.Valkey, valkey.ValkeyCluster], **kwargs: Any):
        """Initialize ValkeySessionManager with Valkey storage.

        Args:
            session_id: ID for the session
            client: Pre-configured Valkey client (Valkey or ValkeyCluster)
            **kwargs: Additional keyword arguments for future extensibility.
        """
        self.client = client
        super().__init__(session_id=session_id, session_repository=self)

    def _get_session_key(self, session_id: str) -> str:
        """Get session key.

        Args:
            session_id: ID for the session.

        Raises:
            ValueError: If session_id contains colon characters.
        """
        if ":" in session_id:
            raise ValueError(f"session_id cannot contain ':' characters: {session_id}")
        return f"{SESSION_PREFIX}:{session_id}"

    def _get_agent_key(self, session_id: str, agent_id: str) -> str:
        """Get agent key.

        Args:
            session_id: ID for the session.
            agent_id: ID for the agent.

        Raises:
            ValueError: If agent_id contains colon characters.
        """
        if ":" in agent_id:
            raise ValueError(f"agent_id cannot contain ':' characters: {agent_id}")
        session_key = self._get_session_key(session_id)
        return f"{session_key}:{AGENT_PREFIX}:{agent_id}"

    def _get_message_key(self, session_id: str, agent_id: str, message_id: int) -> str:
        """Get message key.

        Args:
            session_id: ID of the session
            agent_id: ID of the agent
            message_id: Index of the message

        Returns:
            The key for the message

        Raises:
            ValueError: If message_id is not an integer.
        """
        if not isinstance(message_id, int):
            raise ValueError(f"message_id=<{message_id}> | message id must be an integer")

        agent_key = self._get_agent_key(session_id, agent_id)
        return f"{agent_key}:{MESSAGE_PREFIX}:{message_id}"

    def _read_json_object(self, key: str) -> Optional[Dict[str, Any]]:
        """Read JSON object from Valkey."""
        try:
            data = self.client.execute_command("JSON.GET", key)
            if data is None:
                return None
            return cast(dict[str, Any], json.loads(data))
        except Exception as e:
            raise SessionException(f"Valkey error reading {key}: {e}") from e

    def _write_json_object(self, key: str, data: Dict[str, Any]) -> None:
        """Write JSON object to Valkey."""
        try:
            json_data = json.dumps(data, ensure_ascii=False)
            self.client.execute_command("JSON.SET", key, ".", json_data)
        except Exception as e:
            raise SessionException(f"Failed to write Valkey object {key}: {e}") from e

    def create_session(self, session: Session, **kwargs: Any) -> Session:
        """Create a new session in Valkey."""
        session_key = self._get_session_key(session.session_id)

        # Check if session already exists
        if self.client.exists(session_key):
            raise SessionException(f"Session {session.session_id} already exists")

        # Write session object
        session_dict = session.to_dict()
        self._write_json_object(session_key, session_dict)
        return session

    def read_session(self, session_id: str, **kwargs: Any) -> Optional[Session]:
        """Read session data from Valkey."""
        session_key = self._get_session_key(session_id)
        session_data = self._read_json_object(session_key)
        if session_data is None:
            return None
        return Session.from_dict(session_data)

    def delete_session(self, session_id: str, **kwargs: Any) -> None:
        """Delete session and all associated data from Valkey."""
        session_key = self._get_session_key(session_id)

        # Find all keys related to this session using SCAN
        pattern = f"{session_key}*"
        keys = []
        cursor = 0
        while True:
            cursor, batch = self.client.scan(cursor=cursor, match=pattern, count=100)  # type: ignore[misc]
            keys.extend(batch)
            if cursor == 0:
                break

        if not keys:
            raise SessionException(f"Session {session_id} does not exist")

        # Delete keys individually to avoid CROSSSLOT errors in clustered mode
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            self.client.delete(key_str)

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Create a new agent in Valkey."""
        agent_id = session_agent.agent_id
        agent_dict = session_agent.to_dict()
        agent_key = self._get_agent_key(session_id, agent_id)
        self._write_json_object(agent_key, agent_dict)

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> Optional[SessionAgent]:
        """Read agent data from Valkey."""
        agent_key = self._get_agent_key(session_id, agent_id)
        agent_data = self._read_json_object(agent_key)
        if agent_data is None:
            return None
        return SessionAgent.from_dict(agent_data)

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Update agent data in Valkey."""
        agent_id = session_agent.agent_id
        previous_agent = self.read_agent(session_id=session_id, agent_id=agent_id)
        if previous_agent is None:
            raise SessionException(f"Agent {agent_id} in session {session_id} does not exist")

        # Preserve creation timestamp
        session_agent.created_at = previous_agent.created_at
        agent_key = self._get_agent_key(session_id, agent_id)
        self._write_json_object(agent_key, session_agent.to_dict())

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Create a new message in Valkey."""
        message_id = session_message.message_id
        message_dict = session_message.to_dict()
        message_key = self._get_message_key(session_id, agent_id, message_id)
        self._write_json_object(message_key, message_dict)

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs: Any) -> Optional[SessionMessage]:
        """Read message data from Valkey."""
        message_key = self._get_message_key(session_id, agent_id, message_id)
        message_data = self._read_json_object(message_key)
        if message_data is None:
            return None
        return SessionMessage.from_dict(message_data)

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Update message data in Valkey."""
        message_id = session_message.message_id
        previous_message = self.read_message(session_id=session_id, agent_id=agent_id, message_id=message_id)
        if previous_message is None:
            raise SessionException(f"Message {message_id} does not exist")

        # Preserve creation timestamp
        session_message.created_at = previous_message.created_at
        message_key = self._get_message_key(session_id, agent_id, message_id)
        self._write_json_object(message_key, session_message.to_dict())

    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0, **kwargs: Any
    ) -> List[SessionMessage]:
        """List messages for an agent with pagination from Valkey."""
        agent_key = self._get_agent_key(session_id, agent_id)
        messages_pattern = f"{agent_key}:{MESSAGE_PREFIX}:*"

        try:
            # Use SCAN instead of KEYS (KEYS not supported in ElastiCache Serverless)
            message_keys = []
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor=cursor, match=messages_pattern, count=100)  # type: ignore[misc]
                message_keys.extend(keys)
                if cursor == 0:
                    break

            # Extract message indices and sort
            message_index_keys: list[tuple[int, str]] = []
            for key in message_keys:
                # Decode bytes to string if needed
                key_str = key.decode() if isinstance(key, bytes) else key
                # Extract index from key format: session:id:agent:id:message:index
                index = int(key_str.split(":")[-1])
                message_index_keys.append((index, key_str))

            # Sort by index and extract just the keys
            sorted_keys = [k for _, k in sorted(message_index_keys)]

            # Apply pagination to keys before loading content
            if limit is not None:
                sorted_keys = sorted_keys[offset : offset + limit]
            else:
                sorted_keys = sorted_keys[offset:]

            # Load only the required message objects
            messages: List[SessionMessage] = []
            for key in sorted_keys:
                message_data = self._read_json_object(key)
                if message_data:
                    messages.append(SessionMessage.from_dict(message_data))

            return messages

        except Exception as e:
            raise SessionException(f"Valkey error reading messages: {e}") from e
