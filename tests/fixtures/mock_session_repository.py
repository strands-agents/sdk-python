from strands.session.session_repository import SessionRepository
from strands.types.exceptions import SessionException


class MockedSessionRepository(SessionRepository):
    """Mock repository for testing."""

    def __init__(self):
        """Initialize with empty storage."""
        self.sessions = {}
        self.agents = {}
        self.messages = {}

    def create_session(self, session):
        """Create a session."""
        session_id = session.session_id
        if session_id in self.sessions:
            raise SessionException(f"Session {session_id} already exists")
        self.sessions[session_id] = session
        self.agents[session_id] = {}
        self.messages[session_id] = {}
        return session

    def read_session(self, session_id):
        """Read a session."""
        return self.sessions.get(session_id)

    def create_agent(self, session_id, session_agent):
        """Create an agent."""
        agent_id = session_agent.agent_id
        if session_id not in self.sessions:
            raise SessionException(f"Session {session_id} does not exist")
        if agent_id in self.agents.get(session_id, {}):
            raise SessionException(f"Agent {agent_id} already exists in session {session_id}")
        self.agents.setdefault(session_id, {})[agent_id] = session_agent
        self.messages.setdefault(session_id, {}).setdefault(agent_id, [])
        return session_agent

    def read_agent(self, session_id, agent_id):
        """Read an agent."""
        if session_id not in self.sessions:
            return None
        return self.agents.get(session_id, {}).get(agent_id)

    def update_agent(self, session_id, session_agent):
        """Update an agent."""
        agent_id = session_agent.agent_id
        if session_id not in self.sessions:
            raise SessionException(f"Session {session_id} does not exist")
        if agent_id not in self.agents.get(session_id, {}):
            raise SessionException(f"Agent {agent_id} does not exist in session {session_id}")
        self.agents[session_id][agent_id] = session_agent

    def create_message(self, session_id, agent_id, session_message):
        """Create a message."""
        if session_id not in self.sessions:
            raise SessionException(f"Session {session_id} does not exist")
        if agent_id not in self.agents.get(session_id, {}):
            raise SessionException(f"Agent {agent_id} does not exist in session {session_id}")
        self.messages.setdefault(session_id, {}).setdefault(agent_id, []).append(session_message)

    def read_message(self, session_id, agent_id, message_id):
        """Read a message."""
        if session_id not in self.sessions:
            return None
        if agent_id not in self.agents.get(session_id, {}):
            return None
        for message in self.messages.get(session_id, {}).get(agent_id, []):
            if message.message_id == message_id:
                return message
        return None

    def update_message(self, session_id, agent_id, session_message):
        """Update a message."""
        message_id = session_message.message_id
        if session_id not in self.sessions:
            raise SessionException(f"Session {session_id} does not exist")
        if agent_id not in self.agents.get(session_id, {}):
            raise SessionException(f"Agent {agent_id} does not exist in session {session_id}")

        for i, message in enumerate(self.messages.get(session_id, {}).get(agent_id, [])):
            if message.message_id == message_id:
                self.messages[session_id][agent_id][i] = session_message
                return

        raise SessionException(f"Message {message_id} does not exist")

    def list_messages(self, session_id, agent_id, limit=None, offset=0):
        """List messages."""
        if session_id not in self.sessions:
            return []
        if agent_id not in self.agents.get(session_id, {}):
            return []

        messages = self.messages.get(session_id, {}).get(agent_id, [])
        if limit is not None:
            return messages[offset : offset + limit]
        return messages[offset:]
