"""S3-based session DAO for cloud storage."""

import json
from typing import Any, Dict, List, Optional, cast

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

from .exceptions import SessionException
from .session_dao import SessionDAO
from .session_models import Session, SessionAgent, SessionMessage

SESSION_PREFIX = "session_"
AGENT_PREFIX = "agent_"
MESSAGE_PREFIX = "message_"


class S3SessionDAO(SessionDAO):
    """S3-based session DAO for cloud storage."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        boto_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        region_name: Optional[str] = None,
    ):
        """Initialize S3SessionDAO with S3 storage.

        Args:
            bucket: S3 bucket name (required)
            prefix: S3 key prefix for storage organization
            boto_session: Optional boto3 session
            boto_client_config: Optional boto3 client configuration
            region_name: AWS region for S3 storage
        """
        self.bucket = bucket
        self.prefix = prefix

        session = boto_session or boto3.Session(region_name=region_name)

        # Add strands-agents to the request user agent
        if boto_client_config:
            existing_user_agent = getattr(boto_client_config, "user_agent_extra", None)
            # Append 'strands-agents' to existing user_agent_extra or set it if not present
            if existing_user_agent:
                new_user_agent = f"{existing_user_agent} strands-agents"
            else:
                new_user_agent = "strands-agents"
            client_config = boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            client_config = BotocoreConfig(user_agent_extra="strands-agents")

        self.client = session.client(service_name="s3", config=client_config)

    def _get_session_path(self, session_id: str) -> str:
        """Get session S3 prefix."""
        return f"{self.prefix}{SESSION_PREFIX}{session_id}/"

    def _get_agent_path(self, session_id: str, agent_id: str) -> str:
        """Get agent S3 prefix."""
        session_path = self._get_session_path(session_id)
        return f"{session_path}agents/{AGENT_PREFIX}{agent_id}/"

    def _get_message_path(self, session_id: str, agent_id: str, message_id: str) -> str:
        """Get message S3 key."""
        agent_path = self._get_agent_path(session_id, agent_id)
        return f"{agent_path}messages/{MESSAGE_PREFIX}{message_id}.json"

    def _read_s3_object(self, key: str) -> Dict[str, Any]:
        """Read JSON object from S3."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            return cast(dict[str, Any], json.loads(content))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise SessionException(f"Object not found: {key}") from e
            else:
                raise SessionException(f"S3 error reading {key}: {e}") from e
        except json.JSONDecodeError as e:
            raise SessionException(f"Invalid JSON in S3 object {key}: {e}") from e

    def _write_s3_object(self, key: str, data: Dict[str, Any]) -> None:
        """Write JSON object to S3."""
        try:
            content = json.dumps(data, indent=2, ensure_ascii=False)
            self.client.put_object(
                Bucket=self.bucket, Key=key, Body=content.encode("utf-8"), ContentType="application/json"
            )
        except ClientError as e:
            raise SessionException(f"Failed to write S3 object {key}: {e}") from e

    def create_session(self, session: Session) -> Session:
        """Create a new session in S3."""
        session_key = f"{self._get_session_path(session.session_id)}session.json"

        # Check if session already exists
        try:
            self.client.head_object(Bucket=self.bucket, Key=session_key)
            raise SessionException(f"Session {session.session_id} already exists")
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise SessionException(f"S3 error checking session existence: {e}") from e

        # Write session object
        session_data = session.to_dict()
        self._write_s3_object(session_key, session_data)
        return session

    def read_session(self, session_id: str) -> Session:
        """Read session data from S3."""
        session_key = f"{self._get_session_path(session_id)}session.json"
        session_data = self._read_s3_object(session_key)
        return Session.from_dict(session_data)

    def update_session(self, session: Session) -> None:
        """Update session data in S3."""
        from datetime import datetime, timezone

        session.updated_at = datetime.now(timezone.utc).isoformat()
        session_data = session.to_dict()
        session_key = f"{self._get_session_path(session.session_id)}session.json"
        self._write_s3_object(session_key, session_data)

    def list_sessions(self) -> List[Session]:
        """List all sessions in S3."""
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix, Delimiter="/")

            sessions = []
            for page in pages:
                if "CommonPrefixes" in page:
                    for prefix_info in page["CommonPrefixes"]:
                        prefix = prefix_info["Prefix"]
                        # Extract session ID from prefix like "session_123/"
                        session_part = prefix.rstrip("/").split("/")[-1]
                        if session_part.startswith(SESSION_PREFIX):
                            session_id = session_part[len(SESSION_PREFIX) :]  # Remove "session_" prefix
                            sessions.append(self.read_session(session_id))

            return sessions

        except ClientError as e:
            raise SessionException(f"S3 error listing sessions: {e}") from e

    def delete_session(self, session_id: str) -> None:
        """Delete session and all associated data from S3."""
        session_prefix = self._get_session_path(session_id)
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=session_prefix)

            objects_to_delete = []
            for page in pages:
                if "Contents" in page:
                    objects_to_delete.extend([{"Key": obj["Key"]} for obj in page["Contents"]])

            if not objects_to_delete:
                raise SessionException(f"Session {session_id} does not exist")

            # Delete objects in batches
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i : i + 1000]
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})

        except ClientError as e:
            raise SessionException(f"S3 error deleting session {session_id}: {e}") from e

    def create_agent(self, session_id: str, session_agent: SessionAgent) -> None:
        """Create a new agent in S3."""
        agent_id = session_agent.agent_id
        agent_data = session_agent.to_dict()
        agent_key = f"{self._get_agent_path(session_id, agent_id)}agent.json"
        self._write_s3_object(agent_key, agent_data)

    def read_agent(self, session_id: str, agent_id: str) -> SessionAgent:
        """Read agent data from S3."""
        agent_key = f"{self._get_agent_path(session_id, agent_id)}agent.json"
        agent_data = self._read_s3_object(agent_key)
        return SessionAgent.from_dict(agent_data)

    def update_agent(self, session_id: str, SessionAgent: SessionAgent) -> None:
        """Update agent data in S3."""
        agent_id = SessionAgent.agent_id
        agent_data = SessionAgent.to_dict()
        agent_key = f"{self._get_agent_path(session_id, agent_id)}agent.json"
        self._write_s3_object(agent_key, agent_data)

    def delete_agent(self, session_id: str, agent_id: str) -> None:
        """Delete an agent from S3."""
        agent_prefix = self._get_agent_path(session_id, agent_id)
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=agent_prefix)

            objects_to_delete = []
            for page in pages:
                if "Contents" in page:
                    objects_to_delete.extend([{"Key": obj["Key"]} for obj in page["Contents"]])

            if not objects_to_delete:
                raise SessionException(f"Agent {agent_id} does not exist in session {session_id}")

            # Delete objects in batches
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i : i + 1000]
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})

        except ClientError as e:
            raise SessionException(f"S3 error deleting agent {agent_id}: {e}") from e

    def list_agents(self, session_id: str) -> List[SessionAgent]:
        """List all agents in S3."""
        agents_prefix = f"{self._get_session_path(session_id)}agents/"
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=agents_prefix, Delimiter="/")

            agents = []
            for page in pages:
                if "CommonPrefixes" in page:
                    for prefix_info in page["CommonPrefixes"]:
                        prefix = prefix_info["Prefix"]
                        # Extract agent ID from prefix like "agents/agent_123/"
                        agent_part = prefix.split("/")[-2]  # Get "agent_123"
                        if agent_part.startswith(AGENT_PREFIX):
                            agent_id = agent_part[len(AGENT_PREFIX) :]  # Remove "agent_" prefix
                            agents.append(self.read_agent(session_id, agent_id))

            return agents

        except ClientError as e:
            raise SessionException(f"S3 error listing agents: {e}") from e

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        """Create a new message in S3."""
        message_id = session_message.message_id
        message_data = session_message.to_dict()
        message_key = self._get_message_path(session_id, agent_id, message_id)
        self._write_s3_object(message_key, message_data)

    def read_message(self, session_id: str, agent_id: str, message_id: str) -> SessionMessage:
        """Read message data from S3."""
        message_key = self._get_message_path(session_id, agent_id, message_id)
        message_data = self._read_s3_object(message_key)
        return SessionMessage.from_dict(message_data)

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        """Update message data in S3."""
        message_id = session_message.message_id
        message_data = session_message.to_dict()
        message_key = self._get_message_path(session_id, agent_id, message_id)
        self._write_s3_object(message_key, message_data)

    def delete_message(self, session_id: str, agent_id: str, message_id: str) -> None:
        """Delete a message from S3."""
        message_key = self._get_message_path(session_id, agent_id, message_id)
        try:
            # Check if message exists
            self.client.head_object(Bucket=self.bucket, Key=message_key)
            # Delete the message
            self.client.delete_object(Bucket=self.bucket, Key=message_key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise SessionException(
                    f"Message {message_id} does not exist for agent {agent_id} in session {session_id}"
                ) from e
            else:
                raise SessionException(f"S3 error deleting message {message_id}: {e}") from e

    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[SessionMessage]:
        """List messages for an agent with pagination from S3."""
        messages_prefix = f"{self._get_agent_path(session_id, agent_id)}messages/"
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=messages_prefix)

            # Collect all message objects with timestamps
            message_objects = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if obj["Key"].endswith(".json"):
                            message_objects.append((obj["Key"], obj["LastModified"]))

            # Sort by last modified (oldest first)
            message_objects.sort(key=lambda x: x[1])

            # Apply pagination
            if limit is not None:
                message_objects = message_objects[offset : offset + limit]
            else:
                message_objects = message_objects[offset:]

            # Read message data
            messages: List[SessionMessage] = []
            for key, _ in message_objects:
                message_data = self._read_s3_object(key)
                messages.append(SessionMessage.from_dict(message_data))

            return messages

        except ClientError as e:
            raise SessionException(f"S3 error reading messages: {e}") from e
