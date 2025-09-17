"""DynamoDB-based session manager for cloud storage."""

import logging
from decimal import Decimal
from typing import Any, List, Optional

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

from .. import _identifier
from ..types.exceptions import SessionException
from ..types.session import Session, SessionAgent, SessionMessage
from .repository_session_manager import RepositorySessionManager
from .session_repository import SessionRepository

logger = logging.getLogger(__name__)


def _convert_decimals_to_native_types(obj: Any) -> Any:
    """Convert Decimal objects to native Python types recursively.

    DynamoDB's TypeDeserializer returns Decimal objects for numeric values,
    but other AWS services expect native Python int/float types.
    """
    if isinstance(obj, Decimal):
        # Convert to int if it's a whole number, otherwise float
        return int(obj) if obj % 1 == 0 else float(obj)
    elif isinstance(obj, dict):
        return {key: _convert_decimals_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_decimals_to_native_types(item) for item in obj]
    else:
        return obj


class DynamoDBSessionManager(RepositorySessionManager, SessionRepository):
    """DynamoDB-based session manager for cloud storage.

    Uses a single table design with the following structure:
    - PK (HASH): session_<session_id>
    - SK (RANGE): session | agent_<agent_id> | agent_<agent_id>#message_<message_id>

    Example:
    ```
    ┌─────────────────┬──────────────────────────┬─────────────────┬──────────────────┐
    │ PK              │ SK                       │ entity_type     │ data             │
    ├─────────────────┼──────────────────────────┼─────────────────┼──────────────────┤
    │ session_abc123  │ session                  │ SESSION         │ {session_json}   │
    │ session_abc123  │ agent_agent1             │ AGENT           │ {agent_json}     │
    │ session_abc123  │ agent_agent1#message_0   │ MESSAGE         │ {message_json}   │
    │ session_abc123  │ agent_agent1#message_1   │ MESSAGE         │ {message_json}   │
    └─────────────────┴──────────────────────────┴─────────────────┴──────────────────┘
    ```
    """

    def __init__(
        self,
        session_id: str,
        table_name: str,
        boto_session: Optional[boto3.Session] = None,
        boto_client_config: Optional[BotocoreConfig] = None,
        region_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize DynamoDBSessionManager.

        Args:
            session_id: ID for the session
            table_name: DynamoDB table name
            boto_session: Optional boto3 session
            boto_client_config: Optional boto3 client configuration
            region_name: AWS region for DynamoDB
            **kwargs: Additional keyword arguments for future extensibility.
        """
        self.table_name = table_name

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

        self.client = session.client(service_name="dynamodb", config=client_config)
        self.serializer = TypeSerializer()
        self.deserializer = TypeDeserializer()
        super().__init__(session_id=session_id, session_repository=self)

    def _validate_dynamodb_id(self, id_: str, id_type: _identifier.Identifier) -> str:
        """Validate ID for DynamoDB key structure.

        Args:
            id_: ID to validate
            id_type: Type of ID for error messages

        Returns:
            Validated ID

        Raises:
            ValueError: If ID contains characters that would break DynamoDB key structure
        """
        if "_" in id_ or "#" in id_:
            raise ValueError(f"{id_type.value}_id={id_} | id cannot contain underscore (_) or hash (#) characters")
        return id_

    def _get_session_pk(self, session_id: str) -> str:
        """Get session partition key."""
        session_id = self._validate_dynamodb_id(session_id, _identifier.Identifier.SESSION)
        return f"session_{session_id}"

    def _get_session_sk(self) -> str:
        """Get session sort key."""
        return "session"

    def _get_agent_sk(self, agent_id: str) -> str:
        """Get agent sort key."""
        agent_id = self._validate_dynamodb_id(agent_id, _identifier.Identifier.AGENT)
        return f"agent_{agent_id}"

    def _get_message_sk(self, agent_id: str, message_id: int) -> str:
        """Get message sort key."""
        if not isinstance(message_id, int):
            raise ValueError(f"message_id=<{message_id}> | message id must be an integer")
        agent_id = self._validate_dynamodb_id(agent_id, _identifier.Identifier.AGENT)
        return f"agent_{agent_id}#message_{message_id}"

    def create_session(self, session: Session, **kwargs: Any) -> Session:
        """Create a new session in DynamoDB."""
        pk = self._get_session_pk(session.session_id)
        sk = self._get_session_sk()

        try:
            self.client.put_item(
                TableName=self.table_name,
                Item={
                    "PK": {"S": pk},
                    "SK": {"S": sk},
                    "entity_type": {"S": "SESSION"},
                    "data": self.serializer.serialize(session.to_dict()),
                },
                ConditionExpression="attribute_not_exists(PK)",
            )
            return session
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise SessionException(f"Session {session.session_id} already exists") from e
            raise SessionException(f"DynamoDB error creating session: {e}") from e

    def read_session(self, session_id: str, **kwargs: Any) -> Optional[Session]:
        """Read session data from DynamoDB."""
        pk = self._get_session_pk(session_id)
        sk = self._get_session_sk()

        try:
            response = self.client.get_item(TableName=self.table_name, Key={"PK": {"S": pk}, "SK": {"S": sk}})
            if "Item" not in response:
                return None

            data = self.deserializer.deserialize(response["Item"]["data"])
            data = _convert_decimals_to_native_types(data)
            return Session.from_dict(data)
        except ClientError as e:
            raise SessionException(f"DynamoDB error reading session: {e}") from e

    def delete_session(self, session_id: str, **kwargs: Any) -> None:
        """Delete session and all associated data from DynamoDB."""
        pk = self._get_session_pk(session_id)

        try:
            # Query all items for this session
            response = self.client.query(
                TableName=self.table_name,
                KeyConditionExpression="PK = :pk",
                ExpressionAttributeValues={":pk": {"S": pk}},
            )

            if not response["Items"]:
                raise SessionException(f"Session {session_id} does not exist")

            # Delete all items in batches
            for i in range(0, len(response["Items"]), 25):
                batch = response["Items"][i : i + 25]
                delete_requests = [{"DeleteRequest": {"Key": {"PK": item["PK"], "SK": item["SK"]}}} for item in batch]
                self.client.batch_write_item(RequestItems={self.table_name: delete_requests})

        except ClientError as e:
            raise SessionException(f"DynamoDB error deleting session: {e}") from e

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Create a new agent in DynamoDB."""
        pk = self._get_session_pk(session_id)
        sk = self._get_agent_sk(session_agent.agent_id)

        try:
            self.client.put_item(
                TableName=self.table_name,
                Item={
                    "PK": {"S": pk},
                    "SK": {"S": sk},
                    "entity_type": {"S": "AGENT"},
                    "data": self.serializer.serialize(session_agent.to_dict()),
                },
            )
        except ClientError as e:
            raise SessionException(f"DynamoDB error creating agent: {e}") from e

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> Optional[SessionAgent]:
        """Read agent data from DynamoDB."""
        pk = self._get_session_pk(session_id)
        sk = self._get_agent_sk(agent_id)

        try:
            response = self.client.get_item(TableName=self.table_name, Key={"PK": {"S": pk}, "SK": {"S": sk}})
            if "Item" not in response:
                return None

            data = self.deserializer.deserialize(response["Item"]["data"])
            data = _convert_decimals_to_native_types(data)
            return SessionAgent.from_dict(data)
        except ClientError as e:
            raise SessionException(f"DynamoDB error reading agent: {e}") from e

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        """Update agent data in DynamoDB."""
        previous_agent = self.read_agent(session_id=session_id, agent_id=session_agent.agent_id)
        if previous_agent is None:
            raise SessionException(f"Agent {session_agent.agent_id} in session {session_id} does not exist")

        # Preserve creation timestamp
        session_agent.created_at = previous_agent.created_at

        pk = self._get_session_pk(session_id)
        sk = self._get_agent_sk(session_agent.agent_id)

        try:
            self.client.put_item(
                TableName=self.table_name,
                Item={
                    "PK": {"S": pk},
                    "SK": {"S": sk},
                    "entity_type": {"S": "AGENT"},
                    "data": self.serializer.serialize(session_agent.to_dict()),
                },
            )
        except ClientError as e:
            raise SessionException(f"DynamoDB error updating agent: {e}") from e

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Create a new message in DynamoDB."""
        pk = self._get_session_pk(session_id)
        sk = self._get_message_sk(agent_id, session_message.message_id)

        try:
            self.client.put_item(
                TableName=self.table_name,
                Item={
                    "PK": {"S": pk},
                    "SK": {"S": sk},
                    "entity_type": {"S": "MESSAGE"},
                    "data": self.serializer.serialize(session_message.to_dict()),
                },
            )
        except ClientError as e:
            raise SessionException(f"DynamoDB error creating message: {e}") from e

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs: Any) -> Optional[SessionMessage]:
        """Read message data from DynamoDB."""
        pk = self._get_session_pk(session_id)
        sk = self._get_message_sk(agent_id, message_id)

        try:
            response = self.client.get_item(TableName=self.table_name, Key={"PK": {"S": pk}, "SK": {"S": sk}})
            if "Item" not in response:
                return None

            data = self.deserializer.deserialize(response["Item"]["data"])
            data = _convert_decimals_to_native_types(data)
            return SessionMessage.from_dict(data)
        except ClientError as e:
            raise SessionException(f"DynamoDB error reading message: {e}") from e

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        """Update message data in DynamoDB."""
        previous_message = self.read_message(
            session_id=session_id, agent_id=agent_id, message_id=session_message.message_id
        )
        if previous_message is None:
            raise SessionException(f"Message {session_message.message_id} does not exist")

        # Preserve creation timestamp
        session_message.created_at = previous_message.created_at

        pk = self._get_session_pk(session_id)
        sk = self._get_message_sk(agent_id, session_message.message_id)

        try:
            self.client.put_item(
                TableName=self.table_name,
                Item={
                    "PK": {"S": pk},
                    "SK": {"S": sk},
                    "entity_type": {"S": "MESSAGE"},
                    "data": self.serializer.serialize(session_message.to_dict()),
                },
            )
        except ClientError as e:
            raise SessionException(f"DynamoDB error updating message: {e}") from e

    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0, **kwargs: Any
    ) -> List[SessionMessage]:
        """List messages for an agent with pagination from DynamoDB."""
        pk = self._get_session_pk(session_id)
        agent_prefix = f"agent_{self._validate_dynamodb_id(agent_id, _identifier.Identifier.AGENT)}#message_"

        try:
            # Query messages for this agent
            response = self.client.query(
                TableName=self.table_name,
                KeyConditionExpression="PK = :pk AND begins_with(SK, :sk_prefix)",
                ExpressionAttributeValues={":pk": {"S": pk}, ":sk_prefix": {"S": agent_prefix}},
            )

            # Sort by message ID (extracted from SK)
            items = sorted(response["Items"], key=lambda x: int(x["SK"]["S"].split("_")[-1]))

            # Apply pagination
            if limit is not None:
                items = items[offset : offset + limit]
            else:
                items = items[offset:]

            # Convert to SessionMessage objects
            messages = []
            for item in items:
                data = self.deserializer.deserialize(item["data"])
                data = _convert_decimals_to_native_types(data)
                messages.append(SessionMessage.from_dict(data))

            return messages

        except ClientError as e:
            raise SessionException(f"DynamoDB error listing messages: {e}") from e
