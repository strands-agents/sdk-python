"""Dapr state store session manager for distributed storage."""

import json
import logging
from typing import Any, Dict, List, Literal, Optional, cast

from dapr.clients import DaprClient
from dapr.clients.grpc._state import Consistency, StateOptions

from .. import _identifier
from ..types.exceptions import SessionException
from ..types.session import Session, SessionAgent, SessionMessage
from .repository_session_manager import RepositorySessionManager
from .session_repository import SessionRepository

logger = logging.getLogger(__name__)

# Type-safe consistency constants
ConsistencyLevel = Literal["eventual", "strong"]
DAPR_CONSISTENCY_EVENTUAL: ConsistencyLevel = "eventual"
DAPR_CONSISTENCY_STRONG: ConsistencyLevel = "strong"


class DaprSessionManager(RepositorySessionManager, SessionRepository):
    """Dapr state store session manager for distributed storage.

    Stores session data in Dapr state stores (Redis, PostgreSQL, MongoDB, Cosmos DB, etc.)
    with support for TTL and consistency levels.

    Key structure:
    - `{session_id}:session` - Session metadata
    - `{session_id}:agents:{agent_id}` - Agent metadata
    - `{session_id}:messages:{agent_id}` - Message list (JSON array)
    """

    def __init__(
        self,
        session_id: str,
        state_store_name: str,
        dapr_client: DaprClient,
        ttl: Optional[int] = None,
        consistency: ConsistencyLevel = DAPR_CONSISTENCY_EVENTUAL,
        **kwargs: Any,
    ):
        """Initialize DaprSessionManager.

        Args:
            session_id: ID for the session.
                ID is not allowed to contain path separators (e.g., a/b).
            state_store_name: Name of the Dapr state store component.
            dapr_client: DaprClient instance for state operations.
            ttl: Optional time-to-live in seconds for state items.
            consistency: Consistency level for state operations ("eventual" or "strong").
            **kwargs: Additional keyword arguments for future extensibility.
        """
        self._state_store_name = state_store_name
        self._dapr_client = dapr_client
        self._ttl = ttl
        self._consistency = consistency
        self._owns_client = False

        super().__init__(session_id=session_id, session_repository=self)

    @classmethod
    def from_address(
        cls,
        session_id: str,
        state_store_name: str,
        dapr_address: str = "localhost:50001",
        **kwargs: Any,
    ) -> "DaprSessionManager":
        """Create DaprSessionManager from Dapr address.

        Args:
            session_id: ID for the session.
            state_store_name: Name of the Dapr state store component.
            dapr_address: Dapr gRPC endpoint (default: localhost:50001).
            **kwargs: Additional arguments passed to __init__ (ttl, consistency, etc.).

        Returns:
            DaprSessionManager instance with owned client.
        """
        dapr_client = DaprClient(address=dapr_address)
        manager = cls(session_id, state_store_name=state_store_name, dapr_client=dapr_client, **kwargs)
        manager._owns_client = True
        return manager

    def _get_session_key(self, session_id: str) -> str:
        """Get session state key.

        Args:
            session_id: ID for the session.

        Returns:
            State store key for the session.

        Raises:
            ValueError: If session id contains a path separator.
        """
        session_id = _identifier.validate(session_id, _identifier.Identifier.SESSION)
        return f"{session_id}:session"

    def _get_agent_key(self, session_id: str, agent_id: str) -> str:
        """Get agent state key.

        Args:
            session_id: ID for the session.
            agent_id: ID for the agent.

        Returns:
            State store key for the agent.

        Raises:
            ValueError: If session id or agent id contains a path separator.
        """
        session_id = _identifier.validate(session_id, _identifier.Identifier.SESSION)
        agent_id = _identifier.validate(agent_id, _identifier.Identifier.AGENT)
        return f"{session_id}:agents:{agent_id}"

    def _get_messages_key(self, session_id: str, agent_id: str) -> str:
        """Get messages list state key.

        Args:
            session_id: ID for the session.
            agent_id: ID for the agent.

        Returns:
            State store key for the messages list.

        Raises:
            ValueError: If session id or agent id contains a path separator.
        """
        session_id = _identifier.validate(session_id, _identifier.Identifier.SESSION)
        agent_id = _identifier.validate(agent_id, _identifier.Identifier.AGENT)
        return f"{session_id}:messages:{agent_id}"

    def _get_read_metadata(self) -> Dict[str, str]:
        """Get metadata for read operations (consistency).

        Returns:
            Metadata dictionary for state reads.
        """
        metadata: Dict[str, str] = {}
        if self._consistency:
            metadata["consistency"] = self._consistency
        return metadata

    def _get_write_metadata(self) -> Dict[str, str]:
        """Get metadata for write operations (TTL).

        Returns:
            Metadata dictionary for state writes.
        """
        metadata: Dict[str, str] = {}
        if self._ttl is not None:
            metadata["ttlInSeconds"] = str(self._ttl)
        return metadata

    def _get_state_options(self) -> Optional[StateOptions]:
        """Get state options for write/delete operations (consistency).

        Returns:
            StateOptions for consistency or None.
        """
        if self._consistency == DAPR_CONSISTENCY_STRONG:
            return StateOptions(consistency=Consistency.strong)
        elif self._consistency == DAPR_CONSISTENCY_EVENTUAL:
            return StateOptions(consistency=Consistency.eventual)
        return None

    def _read_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Read and parse JSON state from Dapr.

        Args:
            key: State store key.

        Returns:
            Parsed JSON dictionary or None if not found.

        Raises:
            SessionException: If state is corrupted or read fails.
        """
        try:
            response = self._dapr_client.get_state(
                store_name=self._state_store_name,
                key=key,
                state_metadata=self._get_read_metadata(),
            )

            if not response.data:
                return None

            content = response.data.decode("utf-8")
            return cast(Dict[str, Any], json.loads(content))

        except json.JSONDecodeError as e:
            raise SessionException(f"Invalid JSON in state key {key}: {e}") from e
        except Exception as e:
            raise SessionException(f"Failed to read state key {key}: {e}") from e

    def _write_state(self, key: str, data: Dict[str, Any]) -> None:
        """Write JSON state to Dapr.

        Args:
            key: State store key.
            data: Dictionary to serialize and store.

        Raises:
            SessionException: If write fails.
        """
        try:
            content = json.dumps(data, ensure_ascii=False)
            self._dapr_client.save_state(
                store_name=self._state_store_name,
                key=key,
                value=content,
                state_metadata=self._get_write_metadata(),
                options=self._get_state_options(),
            )
        except Exception as e:
            raise SessionException(f"Failed to write state key {key}: {e}") from e

    def _delete_state(self, key: str) -> None:
        """Delete state from Dapr.

        Args:
            key: State store key.

        Raises:
            SessionException: If delete fails.
        """
        try:
            self._dapr_client.delete_state(
                store_name=self._state_store_name,
                key=key,
                state_metadata=self._get_read_metadata(),
                options=self._get_state_options(),
            )
        except Exception as e:
            raise SessionException(f"Failed to delete state key {key}: {e}") from e

    def _get_manifest_key(self, session_id: str) -> str:
        """Get session manifest key (tracks agent_ids for deletion)."""
        session_id = _identifier.validate(session_id, _identifier.Identifier.SESSION)
        return f"{session_id}:manifest"

    def create_session(self, session: Session, **kwargs: Any) -> Session:
        """Create a new session.

        Args:
            session: Session to create.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Created session.

        Raises:
            SessionException: If session already exists or creation fails.
        """
        session_key = self._get_session_key(session.session_id)

        # Check if session already exists
        existing = self.read_session(session.session_id)
        if existing is not None:
            raise SessionException(f"Session {session.session_id} already exists")

        # Write session data
        session_dict = session.to_dict()
        self._write_state(session_key, session_dict)
        return session

    def read_session(self, session_id: str, **kwargs: Any) -> Optional[Session]:
        """Read session data.

        Args:
            session_id: ID of the session to read.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Session if found, None otherwise.

        Raises:
            SessionException: If read fails.
        """
        session_key = self._get_session_key(session_id)

        session_data = self._read_state(session_key)
        if session_data is None:
            return None

        return Session.from_dict(session_data)

    def delete_session(self, session_id: str, **kwargs: Any) -> None:
        """Delete session and all associated data.

        Uses a session manifest to discover agent IDs for cleanup.
        """
        session_key = self._get_session_key(session_id)
        manifest_key = self._get_manifest_key(session_id)

        # Read manifest (may be missing if no agents created)
        manifest = self._read_state(manifest_key)
        agent_ids: list[str] = manifest.get("agents", []) if manifest else []

        # Delete agent and message keys
        for agent_id in agent_ids:
            agent_key = self._get_agent_key(session_id, agent_id)
            messages_key = self._get_messages_key(session_id, agent_id)
            self._delete_state(agent_key)
            self._delete_state(messages_key)

        # Delete manifest and session
        self._delete_state(manifest_key)
        self._delete_state(session_key)

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Create a new agent in the session.

        Args:
            session_id: ID of the session.
            session_agent: Agent to create.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            SessionException: If creation fails.
        """
        agent_key = self._get_agent_key(session_id, session_agent.agent_id)
        agent_dict = session_agent.to_dict()

        self._write_state(agent_key, agent_dict)

        # Initialize empty messages list
        messages_key = self._get_messages_key(session_id, session_agent.agent_id)
        self._write_state(messages_key, {"messages": []})

        # Update manifest with this agent
        manifest_key = self._get_manifest_key(session_id)
        manifest = self._read_state(manifest_key) or {"agents": []}
        if session_agent.agent_id not in manifest["agents"]:
            manifest["agents"].append(session_agent.agent_id)
        self._write_state(manifest_key, manifest)

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> Optional[SessionAgent]:
        """Read agent data.

        Args:
            session_id: ID of the session.
            agent_id: ID of the agent.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            SessionAgent if found, None otherwise.

        Raises:
            SessionException: If read fails.
        """
        agent_key = self._get_agent_key(session_id, agent_id)

        agent_data = self._read_state(agent_key)
        if agent_data is None:
            return None

        return SessionAgent.from_dict(agent_data)

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Update agent data.

        Args:
            session_id: ID of the session.
            session_agent: Agent to update.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            SessionException: If agent doesn't exist or update fails.
        """
        previous_agent = self.read_agent(session_id=session_id, agent_id=session_agent.agent_id)
        if previous_agent is None:
            raise SessionException(f"Agent {session_agent.agent_id} in session {session_id} does not exist")

        # Preserve creation timestamp
        session_agent.created_at = previous_agent.created_at

        agent_key = self._get_agent_key(session_id, session_agent.agent_id)

        self._write_state(agent_key, session_agent.to_dict())

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Create a new message for the agent.

        Args:
            session_id: ID of the session.
            agent_id: ID of the agent.
            session_message: Message to create.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            SessionException: If creation fails.
        """
        messages_key = self._get_messages_key(session_id, agent_id)

        # Read existing messages
        messages_data = self._read_state(messages_key)
        if messages_data is None:
            messages_list = []
        else:
            messages_list = messages_data.get("messages", [])
            if not isinstance(messages_list, list):
                messages_list = []

        # Append new message
        messages_list.append(session_message.to_dict())

        # Write back
        self._write_state(messages_key, {"messages": messages_list})

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs: Any) -> Optional[SessionMessage]:
        """Read message data.

        Args:
            session_id: ID of the session.
            agent_id: ID of the agent.
            message_id: Index of the message.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            SessionMessage if found, None otherwise.

        Raises:
            ValueError: If message_id is not an integer.
            SessionException: If read fails.
        """
        if not isinstance(message_id, int):
            raise ValueError(f"message_id=<{message_id}> | message id must be an integer")

        messages_key = self._get_messages_key(session_id, agent_id)

        messages_data = self._read_state(messages_key)
        if messages_data is None:
            return None

        messages_list = messages_data.get("messages", [])
        if not isinstance(messages_list, list):
            messages_list = []

        # Find message by ID
        for msg_dict in messages_list:
            if msg_dict.get("message_id") == message_id:
                return SessionMessage.from_dict(msg_dict)

        return None

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Update message data.

        Args:
            session_id: ID of the session.
            agent_id: ID of the agent.
            session_message: Message to update.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            SessionException: If message doesn't exist or update fails.
        """
        previous_message = self.read_message(
            session_id=session_id, agent_id=agent_id, message_id=session_message.message_id
        )
        if previous_message is None:
            raise SessionException(f"Message {session_message.message_id} does not exist")

        # Preserve creation timestamp
        session_message.created_at = previous_message.created_at

        messages_key = self._get_messages_key(session_id, agent_id)

        # Read existing messages
        messages_data = self._read_state(messages_key)
        if messages_data is None:
            raise SessionException(f"Messages not found for agent {agent_id} in session {session_id}")

        messages_list = messages_data.get("messages", [])
        if not isinstance(messages_list, list):
            messages_list = []

        # Find and update message
        updated = False
        for i, msg_dict in enumerate(messages_list):
            if msg_dict.get("message_id") == session_message.message_id:
                messages_list[i] = session_message.to_dict()
                updated = True
                break

        if not updated:
            raise SessionException(f"Message {session_message.message_id} not found in list")

        # Write back
        self._write_state(messages_key, {"messages": messages_list})

    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0, **kwargs: Any
    ) -> List[SessionMessage]:
        """List messages for an agent with pagination.

        Args:
            session_id: ID of the session.
            agent_id: ID of the agent.
            limit: Maximum number of messages to return.
            offset: Number of messages to skip.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            List of SessionMessage objects.

        Raises:
            SessionException: If read fails.
        """
        messages_key = self._get_messages_key(session_id, agent_id)

        messages_data = self._read_state(messages_key)
        if messages_data is None:
            return []

        messages_list = messages_data.get("messages", [])
        if not isinstance(messages_list, list):
            messages_list = []

        # Apply pagination
        if limit is not None:
            messages_list = messages_list[offset : offset + limit]
        else:
            messages_list = messages_list[offset:]

        # Convert to SessionMessage objects
        return [SessionMessage.from_dict(msg_dict) for msg_dict in messages_list]

    def close(self) -> None:
        """Close the Dapr client if owned by this manager."""
        if self._owns_client:
            self._dapr_client.close()
