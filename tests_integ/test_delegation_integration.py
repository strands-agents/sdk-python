"""Integration tests for agent delegation.

This module tests end-to-end delegation flows using the actual implementation:
- _handle_delegation() in event_loop.py
- AgentDelegationException from types/exceptions.py
- Delegation tool generation from agent.py
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from strands import Agent
from strands.types.exceptions import AgentDelegationException
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.mark.asyncio
class TestDelegationIntegration:
    """Integration tests for end-to-end delegation flows."""

    async def test_end_to_end_delegation_flow(self):
        """Test complete delegation pipeline from tool call to sub-agent execution."""
        # Create sub-agent with multiple responses (sub-agent runs full event loop)
        sub_agent = Agent(
            name="SubAgent",
            model=MockedModelProvider(
                [
                    {"role": "assistant", "content": [{"text": "Sub-agent response"}]},
                    {"role": "assistant", "content": [{"text": "Sub-agent final response"}]},
                    {"role": "assistant", "content": [{"text": "Extra response if needed"}]},
                    {"role": "assistant", "content": [{"text": "Another extra response"}]},
                ]
            ),
        )

        # Create orchestrator with sub-agent
        orchestrator = Agent(
            name="Orchestrator",
            model=MockedModelProvider(
                [
                    # Orchestrator calls delegation tool - delegation will terminate execution
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test123",
                                    "name": "handoff_to_subagent",
                                    "input": {"message": "Handle this task"},
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[sub_agent],
            system_prompt="Delegate tasks when needed",
        )

        orchestrator.messages = [{"role": "user", "content": [{"text": "Test request"}]}]

        # Execute - delegation should occur
        result = await orchestrator.invoke_async()

        # Verify sub-agent was called
        assert result is not None
        assert sub_agent.messages  # Sub-agent received messages
        # Verify delegation context was added
        delegation_msg_found = any(
            "Delegated from Orchestrator" in str(msg.get("content", [])) for msg in sub_agent.messages
        )
        assert delegation_msg_found

    async def test_delegation_exception_raised_in_tool(self):
        """Test that delegation tools raise AgentDelegationException."""
        sub_agent = Agent(
            name="Target", model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Target response"}]}])
        )

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Orch response"}]}]),
            sub_agents=[sub_agent],
        )

        # Get the generated delegation tool
        delegation_tool = orchestrator.tool_registry.registry.get("handoff_to_target")
        assert delegation_tool is not None

        # Calling the tool should raise AgentDelegationException
        with pytest.raises(AgentDelegationException) as exc_info:
            # Call the tool directly using __call__
            delegation_tool(message="Test message", context={"key": "value"})

        # Verify exception contents
        exc = exc_info.value
        assert exc.target_agent == "Target"
        assert exc.message == "Test message"
        assert exc.context == {"key": "value"}
        assert "Orch" in exc.delegation_chain

    async def test_state_transfer_is_deep_copy(self):
        """Verify state is deep copied - mutations don't affect original."""
        sub_agent = Agent(name="Sub", model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Done"}]}]))

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test123",
                                    "name": "handoff_to_sub",
                                    "input": {"message": "Transfer state"},
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[sub_agent],
            delegation_state_transfer=True,
        )

        # Setup with nested mutable state
        orchestrator.state = {"user_data": {"name": "Alice", "scores": [10, 20, 30]}, "config": {"enabled": True}}
        orchestrator.messages = [{"role": "user", "content": [{"text": "Test"}]}]

        # Trigger delegation (transfers state)
        await orchestrator.invoke_async()

        # MUTATE the sub-agent's state
        sub_agent.state["user_data"]["scores"].append(40)
        sub_agent.state["config"]["enabled"] = False

        # VERIFY original is unchanged (proves deep copy)
        assert orchestrator.state["user_data"]["scores"] == [10, 20, 30]
        assert orchestrator.state["config"]["enabled"] is True

        # VERIFY sub-agent has different state
        assert sub_agent.state["user_data"]["scores"] == [10, 20, 30, 40]
        assert sub_agent.state["config"]["enabled"] is False

    async def test_message_filtering_integration(self):
        """Test that internal tool chatter is actually filtered out."""
        sub_agent = Agent(
            name="Sub", model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Response"}]}])
        )

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {"toolUse": {"toolUseId": "t1", "name": "handoff_to_sub", "input": {"message": "Test"}}}
                        ],
                    }
                ]
            ),
            sub_agents=[sub_agent],
            delegation_message_transfer=True,
        )

        # Setup orchestrator with noise
        orchestrator.messages = [
            {"role": "system", "content": [{"text": "System prompt"}]},
            {"role": "user", "content": [{"text": "Calculate 2+2"}]},
            # Internal tool noise that should be FILTERED
            {
                "role": "assistant",
                "content": [{"toolUse": {"name": "calculator", "toolUseId": "t1", "input": {"expr": "2+2"}}}],
            },
            {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "4"}]}}]},
            # Meaningful response that should be KEPT
            {"role": "assistant", "content": [{"type": "text", "text": "The answer is 4"}]},
        ]

        # Trigger delegation
        await orchestrator.invoke_async()

        # VERIFY FILTERING ACTUALLY WORKED
        sub_messages = sub_agent.messages

        # 1. System prompt should be PRESENT
        system_msgs = [m for m in sub_messages if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        assert "System prompt" in str(system_msgs[0])

        # 2. Internal calculator tool should be ABSENT
        for msg in sub_messages:
            msg_str = str(msg)
            if msg.get("role") == "assistant":
                assert "calculator" not in msg_str, "Internal tool should be filtered"
                assert "toolUse" not in msg_str, "Tool uses should be filtered"

        # 3. Meaningful assistant response should be PRESENT
        assistant_msgs = [m for m in sub_messages if m.get("role") == "assistant"]
        meaningful_msgs = [m for m in assistant_msgs if "answer is 4" in str(m).lower()]
        assert len(meaningful_msgs) >= 1, "Meaningful responses should be kept"

        # 4. Delegation context should be PRESENT
        user_msgs = [m for m in sub_messages if m.get("role") == "user"]
        delegation_msgs = [m for m in user_msgs if "Delegated from" in str(m)]
        assert len(delegation_msgs) >= 1, "Delegation context should be added"

    async def test_delegation_timeout_enforcement(self):
        """Test timeout is enforced during delegation."""
        # Create sub-agent that takes too long
        sub_agent = Agent(name="Slow", model=MockedModelProvider([]))

        # Create a mock that returns a coroutine that will never complete
        async def never_respond():
            await asyncio.sleep(10)  # This will never finish due to timeout
            return {"role": "assistant", "content": [{"text": "Too late"}]}

        sub_agent.invoke_async = AsyncMock(side_effect=never_respond)

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test123",
                                    "name": "handoff_to_slow",
                                    "input": {"message": "This will timeout"},
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[sub_agent],
            delegation_timeout=1.0,  # 1 second timeout
        )

        orchestrator.messages = [{"role": "user", "content": [{"text": "Test"}]}]

        # Should timeout - note that the timeout gets wrapped in EventLoopException
        from strands.types.exceptions import EventLoopException

        with pytest.raises((EventLoopException, asyncio.TimeoutError, TimeoutError)):
            await orchestrator.invoke_async()

    async def test_target_agent_not_found_error(self):
        """Test clear error when target agent not found."""
        orchestrator = Agent(name="Orch", model=MockedModelProvider([]))

        # Simulate a delegation exception for non-existent agent
        from strands.event_loop.event_loop import _handle_delegation

        fake_exception = AgentDelegationException(target_agent="NonExistent", message="test", delegation_chain=[])

        with pytest.raises(ValueError, match="not found"):
            await _handle_delegation(
                agent=orchestrator,
                delegation_exception=fake_exception,
                invocation_state={},
                cycle_trace=Mock(id="test", add_child=Mock(), add_event=Mock()),
                cycle_span=None,
            )

    async def test_circular_delegation_prevention(self):
        """Test circular delegation is detected and prevented."""
        orchestrator = Agent(name="Orch", model=MockedModelProvider([]))

        # Simulate circular delegation
        from strands.event_loop.event_loop import _handle_delegation

        circular_exception = AgentDelegationException(
            target_agent="Orch",  # Trying to delegate back to self
            message="circular",
            delegation_chain=["Orch"],  # Already in chain
        )

        orchestrator._sub_agents["Orch"] = orchestrator

        with pytest.raises(ValueError, match="Circular delegation"):
            await _handle_delegation(
                agent=orchestrator,
                delegation_exception=circular_exception,
                invocation_state={},
                cycle_trace=Mock(id="test", add_child=Mock(), add_event=Mock()),
                cycle_span=None,
            )

    async def test_delegation_context_always_added(self):
        """Test delegation message is always appended to sub-agent."""
        sub_agent = Agent(name="Sub", model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Done"}]}]))

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test123",
                                    "name": "handoff_to_sub",
                                    "input": {"message": "Important task"},
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[sub_agent],
            delegation_message_transfer=False,  # Even with no message transfer
        )

        orchestrator.messages = [{"role": "user", "content": [{"text": "Test"}]}]

        # Execute delegation
        await orchestrator.invoke_async()

        # Delegation context should still be added
        delegation_msg_found = any(
            "Delegated from Orch: Important task" in str(msg.get("content", [])) for msg in sub_agent.messages
        )
        assert delegation_msg_found

    async def test_additional_context_transfer(self):
        """Test additional context is passed to sub-agent."""
        sub_agent = Agent(name="Sub", model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Done"}]}]))

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test123",
                                    "name": "handoff_to_sub",
                                    "input": {
                                        "message": "Handle with context",
                                        "context": {"user_id": "123", "priority": "high"},
                                    },
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[sub_agent],
        )

        orchestrator.messages = [{"role": "user", "content": [{"text": "Test"}]}]

        # Execute delegation
        await orchestrator.invoke_async()

        # Context should be in sub-agent messages
        context_msg_found = any(
            "Additional context:" in str(msg.get("content", [])) and "user_id" in str(msg.get("content", []))
            for msg in sub_agent.messages
        )
        assert context_msg_found

    async def test_max_delegation_depth_enforcement(self):
        """Test maximum delegation depth is enforced."""
        sub_agent = Agent(name="Sub", model=MockedModelProvider([]))

        orchestrator = Agent(name="Orch", model=MockedModelProvider([]), sub_agents=[sub_agent], max_delegation_depth=2)

        # Get delegation tool
        delegation_tool = orchestrator.tool_registry.registry.get("handoff_to_sub")

        # Try to exceed max depth
        with pytest.raises(ValueError, match="Maximum delegation depth"):
            delegation_tool(
                message="test",
                delegation_chain=["A", "B"],  # Length 2, adding one more would exceed max
            )

    async def test_streaming_proxy_integration(self):
        """Test streaming proxy functionality for delegation."""

        sub_agent = Agent(
            name="StreamingSub",
            model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Streaming response"}]}]),
        )

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test123",
                                    "name": "handoff_to_streamingsub",
                                    "input": {"message": "Stream this"},
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[sub_agent],
            delegation_streaming_proxy=True,
        )

        orchestrator.messages = [{"role": "user", "content": [{"text": "Test"}]}]

        # Test that streaming proxy is enabled
        assert orchestrator.delegation_streaming_proxy is True
        # Test basic delegation flow (streaming proxy tested in unit tests)
        result = await orchestrator.invoke_async()
        assert result is not None

    async def test_session_persistence_integration(self):
        """Test session persistence during delegation."""
        # This test verifies that the delegation mechanism handles session management correctly
        # We test the basic delegation flow with session context tracking

        sub_agent = Agent(
            name="SessionSub",
            model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Session response"}]}]),
        )

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test123",
                                    "name": "handoff_to_sessionsub",
                                    "input": {"message": "Persistent session"},
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[sub_agent],
        )

        orchestrator.messages = [{"role": "user", "content": [{"text": "Test"}]}]

        # Test that delegation works even without explicit session management
        # The delegation system should gracefully handle cases where session managers are not set up
        result = await orchestrator.invoke_async()

        # Verify delegation completed successfully
        assert result is not None
        # Verify sub-agent received the delegation context
        assert len(sub_agent.messages) > 0

        # Check that delegation message was properly added
        delegation_msg_found = any("Delegated from Orch" in str(msg.get("content", [])) for msg in sub_agent.messages)
        assert delegation_msg_found

    async def test_nested_delegation_chain_integration(self):
        """Test multi-level nested delegation chains."""
        # Create 3-level hierarchy: Orchestrator -> Level1 -> Level2
        level2_agent = Agent(
            name="Level2", model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Level2 response"}]}])
        )

        level1_agent = Agent(
            name="Level1",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test456",
                                    "name": "handoff_to_level2",
                                    "input": {"message": "Delegate to Level2"},
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[level2_agent],
        )

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test123",
                                    "name": "handoff_to_level1",
                                    "input": {"message": "Delegate to Level1"},
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[level1_agent],
            max_delegation_depth=3,
        )

        orchestrator.messages = [{"role": "user", "content": [{"text": "Nested test"}]}]

        # Execute nested delegation
        await orchestrator.invoke_async()

        # Verify delegation chain worked
        assert level1_agent.messages  # Level1 received delegation
        assert level2_agent.messages  # Level2 received delegation

        # Verify delegation context propagation
        level1_delegation = any("Delegated from Orch" in str(msg.get("content", [])) for msg in level1_agent.messages)
        level2_delegation = any("Delegated from Level1" in str(msg.get("content", [])) for msg in level2_agent.messages)
        assert level1_delegation
        assert level2_delegation

    async def test_event_loop_delegation_handling(self):
        """Test event loop yields delegation completion event."""
        from strands.event_loop.event_loop import event_loop_cycle
        from strands.types._events import DelegationCompleteEvent, EventLoopStopEvent

        sub_agent = Agent(
            name="SubAgent",
            model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Sub-agent response"}]}]),
        )

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test123",
                                    "name": "handoff_to_subagent",
                                    "input": {"message": "Handle this task"},
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[sub_agent],
        )

        orchestrator.messages = [{"role": "user", "content": [{"text": "Test request"}]}]

        # Collect events from event loop cycle
        events = []
        async for event in event_loop_cycle(orchestrator, {}):
            events.append(event)

        # Verify delegation completion and stop events
        delegation_complete_found = any(isinstance(e, DelegationCompleteEvent) for e in events)
        event_loop_stop_found = any(isinstance(e, EventLoopStopEvent) for e in events)

        assert delegation_complete_found, "DelegationCompleteEvent should be yielded"
        assert event_loop_stop_found, "EventLoopStopEvent should be yielded"

    async def test_sub_agent_failure_propagates(self):
        """Test errors from sub-agents bubble up."""
        sub_agent = Agent(name="SubAgent", model=MockedModelProvider([]))

        # Mock invoke_async to raise an exception
        async def failing_invoke():
            raise RuntimeError("Sub-agent failed")

        sub_agent.invoke_async = failing_invoke

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test123",
                                    "name": "handoff_to_subagent",
                                    "input": {"message": "This will fail"},
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[sub_agent],
        )

        orchestrator.messages = [{"role": "user", "content": [{"text": "Test"}]}]

        # Should propagate the sub-agent error (may be wrapped in EventLoopException)
        from strands.types.exceptions import EventLoopException

        with pytest.raises((RuntimeError, EventLoopException), match="Sub-agent failed"):
            await orchestrator.invoke_async()

    async def test_tool_executor_delegation_exception_handling(self):
        """Test tool executor re-raises delegation exceptions."""
        from unittest.mock import Mock

        from strands.tools.executors._executor import ToolExecutor

        # Create a mock agent
        agent = Mock()
        agent.tool_registry = Mock()
        agent.hooks = Mock()
        agent.event_loop_metrics = Mock()
        agent.model = Mock()
        agent.messages = []
        agent.system_prompt = "Test prompt"

        # Mock the tool registry methods
        agent.tool_registry.get_all_tool_specs.return_value = []

        # Create async generator that raises delegation exception
        async def raising_stream(tool_use, invocation_state, **kwargs):
            raise AgentDelegationException(target_agent="TestTarget", message="Test delegation")
            yield  # Never reached, but makes it a generator

        # Create a delegating tool with proper async stream
        delegating_tool = Mock()
        delegating_tool.stream = raising_stream

        # Mock hooks to return the tool
        agent.hooks.invoke_callbacks.return_value = Mock(
            selected_tool=delegating_tool,
            tool_use={"name": "test_tool", "toolUseId": "123"},
            invocation_state={},
            result=Mock(),
        )

        agent.tool_registry.registry = {"test_tool": delegating_tool}
        agent.tool_registry.dynamic_tools = {}

        # Tool executor should re-raise delegation exceptions
        with pytest.raises(AgentDelegationException, match="TestTarget"):
            async for _ in ToolExecutor._stream(
                agent=agent, tool_use={"name": "test_tool", "toolUseId": "123"}, tool_results=[], invocation_state={}
            ):
                pass

    async def test_custom_state_serializer(self):
        """Verify custom state serializer is invoked and applied."""
        sub_agent = Agent(
            name="SubAgent",
            model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Sub-agent response"}]}]),
        )

        serializer_calls = []

        def custom_serializer(state):
            """Example: Exclude private fields starting with underscore."""
            serializer_calls.append(state)
            return {k: v for k, v in state.items() if not k.startswith("_")}

        orchestrator = Agent(
            name="Orch",
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "t1",
                                    "name": "handoff_to_subagent",
                                    "input": {"message": "Test"},
                                }
                            }
                        ],
                    }
                ]
            ),
            sub_agents=[sub_agent],
            delegation_state_serializer=custom_serializer,
            delegation_state_transfer=True,
        )

        # Set state with public and private fields
        orchestrator.state = {"public_data": "visible", "_private_key": "secret", "_internal_cache": [1, 2, 3]}
        orchestrator.messages = [{"role": "user", "content": [{"text": "Test"}]}]

        # Trigger delegation
        await orchestrator.invoke_async()

        # VERIFY serializer was called
        assert len(serializer_calls) == 1, "Serializer should be called once"
        assert "public_data" in serializer_calls[0]

        # VERIFY private fields were excluded
        assert "public_data" in sub_agent.state
        assert "_private_key" not in sub_agent.state
        assert "_internal_cache" not in sub_agent.state
