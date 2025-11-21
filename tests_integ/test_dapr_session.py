"""Integration tests for DaprSessionManager with real Dapr and Redis."""

import os
import shutil
import tempfile
import time
import urllib.request
from typing import Any
from urllib.error import URLError
from uuid import uuid4

# pytestmark = [pytest.mark.asyncio]
import dotenv
import pytest
from testcontainers.core.container import DockerContainer  # type: ignore[import-untyped]
from testcontainers.core.network import Network  # type: ignore[import-untyped]
from testcontainers.redis import RedisContainer  # type: ignore[import-untyped]

from strands import Agent
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.agent.conversation_manager.summarizing_conversation_manager import SummarizingConversationManager
from strands.session.dapr_session_manager import DAPR_CONSISTENCY_STRONG, DaprSessionManager
from tests.fixtures.mocked_model_provider import MockedModelProvider

dotenv.load_dotenv()


@pytest.fixture(scope="module")
def docker_network():
    """Create a Docker network for container-to-container communication."""
    network = Network()
    network.create()
    try:
        yield network
    finally:
        try:
            network.remove()
        except Exception:
            pass


@pytest.fixture(scope="module")
def redis_container(docker_network: Any) -> Any:
    """Redis container on shared network with network alias."""
    container = RedisContainer("redis:7-alpine")
    container = container.with_network(docker_network)
    container = container.with_network_aliases("redis")
    container.start()
    yield container
    try:
        container.stop()
    except Exception:
        pass


@pytest.fixture(scope="module")
def dapr_container(redis_container: Any, docker_network: Any) -> Any:
    """Dapr sidecar container with Redis state store."""
    # Create Dapr component config
    temp_dir = tempfile.mkdtemp()
    component_path = os.path.join(temp_dir, "statestore.yaml")

    with open(component_path, "w") as f:
        f.write(
            """apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: statestore
spec:
  type: state.redis
  version: v1
  metadata:
  - name: redisHost
    value: redis:6379
  - name: redisPassword
    value: ""
  - name: enableTLS
    value: "false"
"""
        )

    # Start Dapr container
    container = DockerContainer("daprio/daprd:latest")
    container = container.with_network(docker_network)
    container = container.with_volume_mapping(temp_dir, "/components", mode="ro")
    container = container.with_command(
        "./daprd "
        "--app-id test-app "
        "--dapr-grpc-port 50001 "
        "--dapr-http-port 3500 "
        "--components-path /components "
        "--log-level debug"
    )
    container = container.with_exposed_ports(50001, 3500)
    container.start()

    # Wait for Dapr to be ready
    http_host = container.get_container_host_ip()
    http_port = container.get_exposed_port(3500)
    if not _wait_for_dapr_health(http_host, http_port, timeout=60):
        container.stop()
        pytest.fail("Dapr container failed to become healthy")

    # Set environment variables for Dapr SDK
    os.environ["DAPR_HTTP_PORT"] = str(http_port)
    os.environ["DAPR_RUNTIME_HOST"] = http_host

    yield container

    container.stop()
    os.environ.pop("DAPR_HTTP_PORT", None)
    os.environ.pop("DAPR_RUNTIME_HOST", None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def _wait_for_dapr_health(host: str, port: int, timeout: int = 60) -> bool:
    """Poll Dapr HTTP health endpoint until ready."""
    health_url = f"http://{host}:{port}/v1.0/healthz/outbound"
    start_time = time.time()
    print(f"Waiting for Dapr health at {health_url}")

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return True
                print(f"Dapr health check failed with status {response.status}")
        except URLError:
            print("Dapr health check failed with URLError")
            pass
        except Exception as e:
            print(f"Dapr health check failed with exception {e}")
        print("Dapr health check failed with timeout")
        time.sleep(1)
    return False


def test_agent_with_dapr_session(dapr_container: Any, monkeypatch: Any):
    """Test agent with DaprSessionManager using real Dapr and Redis."""
    # Bypass SDK's internal health check (already done in fixture)
    from dapr.clients.health import DaprHealth

    monkeypatch.setattr(DaprHealth, "wait_until_ready", lambda: None)

    dapr_host = dapr_container.get_container_host_ip()
    dapr_port = dapr_container.get_exposed_port(50001)
    test_session_id = str(uuid4())

    session_manager = DaprSessionManager.from_address(
        session_id=test_session_id,
        state_store_name="statestore",
        dapr_address=f"{dapr_host}:{dapr_port}",
        consistency=DAPR_CONSISTENCY_STRONG,
    )

    session_manager_2 = None
    try:
        # Use mocked model to avoid real provider calls
        model1 = MockedModelProvider(
            [
                {"role": "assistant", "content": [{"text": "ok"}]},
            ]
        )
        agent = Agent(session_manager=session_manager, model=model1)
        agent("Hello!")
        assert len(session_manager.list_messages(test_session_id, agent.agent_id)) == 2

        # After agent is persisted and run, restore the agent and run it again
        session_manager_2 = DaprSessionManager.from_address(
            session_id=test_session_id,
            state_store_name="statestore",
            dapr_address=f"{dapr_host}:{dapr_port}",
            consistency=DAPR_CONSISTENCY_STRONG,
        )
        model2 = MockedModelProvider(
            [
                {"role": "assistant", "content": [{"text": "ok"}]},
            ]
        )
        agent_2 = Agent(session_manager=session_manager_2, model=model2)
        assert len(agent_2.messages) == 2
        agent_2("Hello again!")
        assert len(agent_2.messages) == 4
        assert len(session_manager_2.list_messages(test_session_id, agent_2.agent_id)) == 4
    finally:
        # Cleanup
        session_manager.close()
        if session_manager_2 is not None:
            session_manager_2.close()


def test_agent_with_dapr_session_and_conversation_manager(dapr_container: Any, monkeypatch: Any):
    """Test agent with DaprSessionManager and SlidingWindowConversationManager."""
    from dapr.clients.health import DaprHealth

    monkeypatch.setattr(DaprHealth, "wait_until_ready", lambda: None)

    dapr_host = dapr_container.get_container_host_ip()
    dapr_port = dapr_container.get_exposed_port(50001)
    test_session_id = str(uuid4())

    session_manager = DaprSessionManager.from_address(
        session_id=test_session_id,
        state_store_name="statestore",
        dapr_address=f"{dapr_host}:{dapr_port}",
        consistency=DAPR_CONSISTENCY_STRONG,
    )

    session_manager_2 = None
    try:
        model1 = MockedModelProvider(
            [
                {"role": "assistant", "content": [{"text": "ok"}]},
            ]
        )
        agent = Agent(
            session_manager=session_manager,
            model=model1,
            conversation_manager=SlidingWindowConversationManager(window_size=1),
        )
        agent("Hello!")
        assert len(session_manager.list_messages(test_session_id, agent.agent_id)) == 2
        # Conversation Manager reduced messages
        assert len(agent.messages) == 1

        # After agent is persisted and run, restore the agent and run it again
        session_manager_2 = DaprSessionManager.from_address(
            session_id=test_session_id,
            state_store_name="statestore",
            dapr_address=f"{dapr_host}:{dapr_port}",
            consistency=DAPR_CONSISTENCY_STRONG,
        )
        model2 = MockedModelProvider(
            [
                {"role": "assistant", "content": [{"text": "ok"}]},
            ]
        )
        agent_2 = Agent(
            session_manager=session_manager_2,
            model=model2,
            conversation_manager=SlidingWindowConversationManager(window_size=1),
        )
        assert len(agent_2.messages) == 1
        assert agent_2.conversation_manager.removed_message_count == 1
        agent_2("Hello again!")
        assert len(agent_2.messages) == 1
        assert len(session_manager_2.list_messages(test_session_id, agent_2.agent_id)) == 4
    finally:
        # Cleanup
        session_manager.close()
        if session_manager_2 is not None:
            session_manager_2.close()


def test_agent_with_dapr_session_with_image(dapr_container: Any, yellow_img: bytes, monkeypatch: Any):
    """Test agent with DaprSessionManager handling image content."""
    from dapr.clients.health import DaprHealth

    monkeypatch.setattr(DaprHealth, "wait_until_ready", lambda: None)

    dapr_host = dapr_container.get_container_host_ip()
    dapr_port = dapr_container.get_exposed_port(50001)
    test_session_id = str(uuid4())

    session_manager = DaprSessionManager.from_address(
        session_id=test_session_id,
        state_store_name="statestore",
        dapr_address=f"{dapr_host}:{dapr_port}",
        consistency=DAPR_CONSISTENCY_STRONG,
    )

    session_manager_2 = None
    try:
        model1 = MockedModelProvider(
            [
                {"role": "assistant", "content": [{"text": "ok"}]},
            ]
        )
        agent = Agent(session_manager=session_manager, model=model1)
        agent([{"image": {"format": "png", "source": {"bytes": yellow_img}}}])
        assert len(session_manager.list_messages(test_session_id, agent.agent_id)) == 2

        # After agent is persisted and run, restore the agent and run it again
        session_manager_2 = DaprSessionManager.from_address(
            session_id=test_session_id,
            state_store_name="statestore",
            dapr_address=f"{dapr_host}:{dapr_port}",
            consistency=DAPR_CONSISTENCY_STRONG,
        )
        model2 = MockedModelProvider(
            [
                {"role": "assistant", "content": [{"text": "ok"}]},
            ]
        )
        agent_2 = Agent(session_manager=session_manager_2, model=model2)
        assert len(agent_2.messages) == 2
        agent_2("Hello!")
        assert len(agent_2.messages) == 4
        assert len(session_manager_2.list_messages(test_session_id, agent_2.agent_id)) == 4
    finally:
        # Cleanup
        session_manager.close()
        if session_manager_2 is not None:
            session_manager_2.close()


def test_agent_with_dapr_session_forced_summarization(dapr_container: Any, monkeypatch: Any):
    """Force summarization via SummarizingConversationManager and verify persistence/restoration with Dapr."""
    from dapr.clients.health import DaprHealth

    monkeypatch.setattr(DaprHealth, "wait_until_ready", lambda: None)

    dapr_host = dapr_container.get_container_host_ip()
    dapr_port = dapr_container.get_exposed_port(50001)
    test_session_id = str(uuid4())

    session_manager = DaprSessionManager.from_address(
        session_id=test_session_id,
        state_store_name="statestore",
        dapr_address=f"{dapr_host}:{dapr_port}",
        consistency=DAPR_CONSISTENCY_STRONG,
    )

    session_manager_2 = None
    try:
        # Use a separate summarizer Agent without a session manager to avoid persisting summarization messages
        summarizer_agent = Agent(model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Summary"}]}]))
        convo_manager = SummarizingConversationManager(
            summarization_agent=summarizer_agent, summary_ratio=0.5, preserve_recent_messages=1
        )
        model1 = MockedModelProvider(
            [
                {"role": "assistant", "content": [{"text": "ok"}]},
                {"role": "assistant", "content": [{"text": "ok"}]},
                {"role": "assistant", "content": [{"text": "ok"}]},
            ]
        )
        agent = Agent(session_manager=session_manager, conversation_manager=convo_manager, model=model1)

        # Add enough messages
        agent("m1")
        agent("m2")
        agent("m3")

        # Explicitly trigger summarization and persist the updated state
        agent.conversation_manager.reduce_context(agent)
        session_manager.sync_agent(agent)

        # Validate summary inserted and state updated
        assert agent.conversation_manager.removed_message_count > 0
        assert agent.messages[0]["role"] == "user"
        assert "Summary" in str(agent.messages[0]["content"])  # summary message

        # Restore with a new manager and agent
        session_manager_2 = DaprSessionManager.from_address(
            session_id=test_session_id,
            state_store_name="statestore",
            dapr_address=f"{dapr_host}:{dapr_port}",
            consistency=DAPR_CONSISTENCY_STRONG,
        )
        agent_2 = Agent(session_manager=session_manager_2, conversation_manager=convo_manager)

        # After restore, messages should reflect trimmed history (summary + remaining)
        assert len(agent_2.messages) <= 4
        assert agent_2.conversation_manager.removed_message_count == agent.conversation_manager.removed_message_count
    finally:
        # Cleanup
        session_manager.delete_session(test_session_id)
        if session_manager_2 is not None:
            session_manager_2.delete_session(test_session_id)
        session_manager.close()
        if session_manager_2 is not None:
            session_manager_2.close()


def test_agent_with_dapr_session_and_summarizing_conversation_manager(dapr_container: Any, monkeypatch: Any):
    """Test agent with DaprSessionManager and SummarizingConversationManager."""
    from dapr.clients.health import DaprHealth

    monkeypatch.setattr(DaprHealth, "wait_until_ready", lambda: None)

    dapr_host = dapr_container.get_container_host_ip()
    dapr_port = dapr_container.get_exposed_port(50001)
    test_session_id = str(uuid4())

    session_manager = DaprSessionManager.from_address(
        session_id=test_session_id,
        state_store_name="statestore",
        dapr_address=f"{dapr_host}:{dapr_port}",
        consistency=DAPR_CONSISTENCY_STRONG,
    )

    session_manager_2 = None
    try:
        # Create agent with summarizing conversation manager
        model1 = MockedModelProvider(
            [
                {"role": "assistant", "content": [{"text": "ok"}]},
            ]
        )
        agent = Agent(
            session_manager=session_manager,
            model=model1,
            conversation_manager=SummarizingConversationManager(summary_ratio=0.5, preserve_recent_messages=2),
        )
        agent("Hello!")
        messages_count = len(session_manager.list_messages(test_session_id, agent.agent_id))
        assert messages_count == 2

        # Restore the agent with the same conversation manager
        session_manager_2 = DaprSessionManager.from_address(
            session_id=test_session_id,
            state_store_name="statestore",
            dapr_address=f"{dapr_host}:{dapr_port}",
            consistency=DAPR_CONSISTENCY_STRONG,
        )
        model2 = MockedModelProvider(
            [
                {"role": "assistant", "content": [{"text": "ok"}]},
                {"role": "assistant", "content": [{"text": "ok"}]},
            ]
        )
        agent_2 = Agent(
            session_manager=session_manager_2,
            model=model2,
            conversation_manager=SummarizingConversationManager(summary_ratio=0.5, preserve_recent_messages=2),
        )

        # Verify state was restored correctly
        assert len(agent_2.messages) == 2
        assert isinstance(agent_2.conversation_manager, SummarizingConversationManager)

        # Add more messages to trigger summarization if needed
        agent_2("Tell me a story")
        agent_2("Continue the story")

        # Verify messages were persisted
        final_messages_count = len(session_manager_2.list_messages(test_session_id, agent_2.agent_id))
        assert final_messages_count >= 4
    finally:
        # Cleanup
        session_manager.close()
        if session_manager_2 is not None:
            session_manager_2.close()
