"""Tests for agent-level default storage."""

from unittest.mock import AsyncMock, MagicMock

from strands import Agent
from strands.session.snapshot_session_manager import SnapshotSessionManager
from strands.storage.in_memory_storage import InMemoryStorage as UnifiedInMemoryStorage
from strands.storage.storage import _NAMESPACED, _NamespacedStorage
from strands.vended_plugins.context_offloader import ContextOffloader
from strands.vended_plugins.context_offloader.storage import InMemoryStorage as OffloaderInMemoryStorage
from tests.fixtures.mocked_model_provider import MockedModelProvider

SIMPLE_RESPONSE = [{"role": "assistant", "content": [{"text": "ok"}]}]


class TestAgentStorageProperty:
    def test_storage_defaults_to_none(self):
        agent = Agent(model=MockedModelProvider(SIMPLE_RESPONSE))
        assert agent.storage is None

    def test_storage_returns_configured_value(self):
        storage = MagicMock()
        storage.write = AsyncMock()
        storage.read = AsyncMock(return_value=None)
        storage.delete = AsyncMock()
        storage.list = AsyncMock(return_value=[])
        agent = Agent(model=MockedModelProvider(SIMPLE_RESPONSE), storage=storage)
        assert agent.storage is storage


class TestOffloaderResolvesAgentStorage:
    def test_offloader_uses_agent_storage_when_no_explicit_storage(self):
        storage = UnifiedInMemoryStorage()
        offloader = ContextOffloader(include_retrieval_tool=False)
        agent = MagicMock()
        agent.storage = storage

        offloader.init_agent(agent)

        assert offloader._storage is not None
        assert isinstance(offloader._storage, _NamespacedStorage)
        assert offloader._storage._namespaced is _NAMESPACED

    def test_offloader_falls_back_to_in_memory_when_no_agent_storage(self):
        offloader = ContextOffloader(include_retrieval_tool=False)
        agent = MagicMock()
        agent.storage = None

        offloader.init_agent(agent)

        assert offloader._storage is not None
        assert isinstance(offloader._storage, OffloaderInMemoryStorage)

    def test_explicit_offloader_storage_overrides_agent_storage(self):
        agent_storage = UnifiedInMemoryStorage()
        explicit_storage = OffloaderInMemoryStorage()
        offloader = ContextOffloader(storage=explicit_storage, include_retrieval_tool=False)
        agent = MagicMock()
        agent.storage = agent_storage

        offloader.init_agent(agent)

        assert offloader._storage is explicit_storage

    def test_explicit_unified_storage_overrides_agent_storage(self):
        agent_storage = UnifiedInMemoryStorage()
        explicit_storage = UnifiedInMemoryStorage()
        offloader = ContextOffloader(storage=explicit_storage, include_retrieval_tool=False)
        agent = MagicMock()
        agent.storage = agent_storage

        offloader.init_agent(agent)

        assert offloader._storage is not agent_storage
        assert isinstance(offloader._storage, _NamespacedStorage)

    def test_offloader_namespaces_agent_storage_under_offloader(self):
        storage = UnifiedInMemoryStorage()
        offloader = ContextOffloader(include_retrieval_tool=False)
        agent = MagicMock()
        agent.storage = storage

        offloader.init_agent(agent)

        assert isinstance(offloader._storage, _NamespacedStorage)
        assert offloader._storage._prefix == "offloader/"


class TestContextManagerUsesAgentStorage:
    def test_auto_context_manager_offloader_resolves_agent_storage(self):
        storage = UnifiedInMemoryStorage()
        agent = Agent(model=MockedModelProvider(SIMPLE_RESPONSE), storage=storage, context_manager="auto")

        offloader = None
        for plugin in agent._plugin_registry._plugins.values():
            if isinstance(plugin, ContextOffloader):
                offloader = plugin
                break

        assert offloader is not None
        assert isinstance(offloader._storage, _NamespacedStorage)
        assert offloader._storage._prefix == "offloader/"


class TestSnapshotSessionManagerResolvesAgentStorage:
    def test_session_manager_uses_agent_storage_when_no_explicit_storage(self):
        storage = UnifiedInMemoryStorage()
        session_mgr = SnapshotSessionManager("test-session")
        agent = MagicMock()
        agent.storage = storage
        agent.agent_id = "agent-1"
        agent.messages = []
        agent.model = MagicMock()
        agent.model.stateful = False
        agent.take_snapshot = MagicMock(return_value=MagicMock())
        agent.load_snapshot = MagicMock()

        session_mgr.initialize(agent)

        assert session_mgr._storage is not None
        assert isinstance(session_mgr._storage, _NamespacedStorage)

    def test_session_manager_falls_back_to_local_file_when_no_agent_storage(self):
        session_mgr = SnapshotSessionManager("test-session")
        agent = MagicMock()
        agent.storage = None
        agent.agent_id = "agent-1"
        agent.messages = []
        agent.model = MagicMock()
        agent.model.stateful = False
        agent.take_snapshot = MagicMock(return_value=MagicMock())
        agent.load_snapshot = MagicMock()

        session_mgr.initialize(agent)

        assert session_mgr._storage is not None
        assert isinstance(session_mgr._storage, _NamespacedStorage)

    def test_explicit_session_storage_overrides_agent_storage(self):
        agent_storage = UnifiedInMemoryStorage()
        explicit_storage = UnifiedInMemoryStorage()
        session_mgr = SnapshotSessionManager("test-session", storage=explicit_storage)
        agent = MagicMock()
        agent.storage = agent_storage
        agent.agent_id = "agent-1"
        agent.messages = []
        agent.model = MagicMock()
        agent.model.stateful = False
        agent.take_snapshot = MagicMock(return_value=MagicMock())
        agent.load_snapshot = MagicMock()

        session_mgr.initialize(agent)

        assert isinstance(session_mgr._storage, _NamespacedStorage)
        # Should be wrapping the explicit storage, not the agent storage
        assert session_mgr._storage._storage is explicit_storage
