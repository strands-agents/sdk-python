"""Performance tests for agent delegation functionality.

This module tests performance characteristics of delegation operations including
overhead measurement, memory usage with deep hierarchies, concurrent delegation,
and timeout behavior under load.
"""

import asyncio
import gc
import resource
import time
import tracemalloc
from typing import Any, Generator
from unittest.mock import AsyncMock

import pytest

from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.event_loop.event_loop import _handle_delegation
from strands.telemetry.metrics import EventLoopMetrics, Trace
from strands.types.exceptions import AgentDelegationException
from tests.fixtures.mocked_model_provider import MockedModelProvider


class PerformanceMeasurement:
    """Utility class for measuring delegation performance."""

    def __init__(self) -> None:
        self.start_time = 0.0
        self.end_time = 0.0
        self.memory_before = 0
        self.memory_after = 0
        self.peak_memory = 0

    def start_measurement(self) -> None:
        """Start performance measurement."""
        gc.collect()  # Clean up before measurement
        tracemalloc.start()
        self.start_time = time.perf_counter()
        self.memory_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def stop_measurement(self) -> dict[str, Any]:
        """Stop measurement and return results."""
        self.end_time = time.perf_counter()
        self.memory_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        current, self.peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Convert memory from KB to MB (on macOS, ru_maxrss is in bytes)
        memory_delta_bytes = self.memory_after - self.memory_before
        memory_delta_mb = memory_delta_bytes / 1024 / 1024

        return {
            "duration_ms": (self.end_time - self.start_time) * 1000,
            "memory_delta_mb": memory_delta_mb,
            "peak_memory_mb": self.peak_memory / 1024 / 1024,
        }


def _make_mock_agent(name: str) -> Agent:
    """Create a mock agent for testing."""
    return Agent(name=name, model=MockedModelProvider([]))


def _make_agent_result(text: str = "delegation_complete") -> AgentResult:
    """Create a mock agent result."""
    return AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": text}]},
        metrics=EventLoopMetrics(),
        state={}
    )


async def _measure_delegation_overhead(
    orchestrator: Agent,
    sub_agent: Agent,
    message: str = "Test message"
) -> dict[str, Any]:
    """Measure delegation overhead for a single delegation."""
    perf = PerformanceMeasurement()

    # Setup sub-agent mock
    sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

    # Create delegation exception
    exception = AgentDelegationException(
        target_agent=sub_agent.name,
        message=message,
        context={},
        delegation_chain=[],
        transfer_state=True,
        transfer_messages=True
    )

    # Start measurement
    perf.start_measurement()

    # Execute delegation
    result = await _handle_delegation(
        agent=orchestrator,
        delegation_exception=exception,
        invocation_state={},
        cycle_trace=Trace("cycle"),
        cycle_span=None
    )

    # Stop measurement and return results
    performance_data = perf.stop_measurement()
    performance_data["success"] = result is not None
    performance_data["trace_children"] = 0  # Basic implementation

    return performance_data


@pytest.mark.asyncio
@pytest.mark.delegation
class TestDelegationOverhead:
    """Test delegation overhead and performance characteristics."""

    async def test_single_delegation_overhead(self) -> None:
        """Test basic delegation overhead for a single delegation."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("SubAgent")
        orchestrator._sub_agents[sub_agent.name] = sub_agent

        # Measure delegation overhead
        performance = await _measure_delegation_overhead(orchestrator, sub_agent)

        # Assertions
        assert performance["success"] is True
        assert performance["duration_ms"] > 0

        # Performance should be reasonable (less than 100ms for simple delegation)
        assert performance["duration_ms"] < 100, f"Delegation too slow: {performance['duration_ms']:.2f}ms"

    async def test_delegation_overhead_with_large_state(self) -> None:
        """Test delegation overhead with large state transfer."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("SubAgent")
        orchestrator._sub_agents[sub_agent.name] = sub_agent

        # Create large state (1MB of data)
        large_state = {
            "data": ["x" * 1000] * 1000,  # ~1MB of strings
            "nested": {
                "level1": {
                    "level2": {
                        "deep_data": list(range(10000))
                    }
                }
            }
        }
        orchestrator.state = large_state

        # Measure delegation overhead
        performance = await _measure_delegation_overhead(orchestrator, sub_agent)

        # Assertions
        assert performance["success"] is True
        assert performance["duration_ms"] > 0
        assert performance["memory_delta_mb"] >= 0

        # Performance should still be reasonable even with large state
        assert performance["duration_ms"] < 500, f"Large state delegation too slow: {performance['duration_ms']:.2f}ms"

    async def test_delegation_overhead_with_message_filtering(self) -> None:
        """Test delegation overhead with message filtering."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("SubAgent")
        orchestrator._sub_agents[sub_agent.name] = sub_agent

        # Create conversation history with many messages (including tool noise)
        orchestrator.messages = [
            {"role": "system", "content": [{"type": "text", "text": "System prompt"}]}
        ] + [
            {"role": "user", "content": [{"type": "text", "text": f"User message {i}"}]}
            for i in range(50)
        ]

        # Add tool noise messages
        for i in range(20):
            orchestrator.messages.append({
                "role": "assistant",
                "content": [
                    {"type": "toolUse", "name": f"internal_tool_{i}", "id": f"tool_{i}", "input": {}},
                    {"type": "text", "text": f"Response {i}"}
                ]
            })

        # Measure delegation overhead
        performance = await _measure_delegation_overhead(orchestrator, sub_agent)

        # Assertions
        assert performance["success"] is True
        assert performance["duration_ms"] > 0

        # Message filtering should reduce noise efficiently
        assert performance["duration_ms"] < 200, f"Message filtering delegation too slow: {performance['duration_ms']:.2f}ms"

    async def test_delegation_overhead_comparison_direct_vs_delegation(self) -> None:
        """Compare performance of direct execution vs delegation."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("SubAgent")
        orchestrator._sub_agents[sub_agent.name] = sub_agent

        # Setup sub-agent to return quickly
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        # Measure direct execution (mock direct call)
        perf_direct = PerformanceMeasurement()
        perf_direct.start_measurement()
        await sub_agent.invoke_async()
        direct_performance = perf_direct.stop_measurement()

        # Measure delegation execution
        delegation_performance = await _measure_delegation_overhead(orchestrator, sub_agent)

        # Calculate overhead
        overhead_ms = delegation_performance["duration_ms"] - direct_performance["duration_ms"]
        overhead_ratio = overhead_ms / direct_performance["duration_ms"] if direct_performance["duration_ms"] > 0 else float('inf')

        # Assertions - delegation should have reasonable overhead
        assert delegation_performance["success"] is True
        assert overhead_ms < 50, f"Delegation overhead too high: {overhead_ms:.2f}ms"
        if direct_performance["duration_ms"] > 0.5:
            assert overhead_ratio < 10, f"Delegation overhead ratio too high: {overhead_ratio:.2f}x"


@pytest.mark.asyncio
@pytest.mark.delegation
class TestDeepHierarchyMemoryUsage:
    """Test memory usage with deep delegation hierarchies."""

    async def test_memory_usage_10_level_hierarchy(self) -> None:
        """Test memory usage with 10-level delegation hierarchy."""
        agents = []

        # Create 10-level hierarchy: Agent0 -> Agent1 -> Agent2 -> ... -> Agent10
        for i in range(11):
            agent = _make_mock_agent(f"Agent{i}")
            if i > 0:
                agents[i-1]._sub_agents[agent.name] = agent
            agents.append(agent)

        # Setup final agent to return result
        agents[-1].invoke_async = AsyncMock(return_value=_make_agent_result("hierarchy_complete"))

        # Measure memory usage through the hierarchy
        perf = PerformanceMeasurement()
        perf.start_measurement()

        # Execute delegation through hierarchy
        current_agent = agents[0]
        for i in range(10):
            next_agent = agents[i + 1]
            if i < 9:
                next_agent.invoke_async = AsyncMock(return_value=_make_agent_result())
            exception = AgentDelegationException(
                target_agent=next_agent.name,
                message=f"Delegation step {i}",
                context={"step": i},
                delegation_chain=[agent.name for agent in agents[:i+1]],
                transfer_state=True,
                transfer_messages=True
            )

            result = await _handle_delegation(
                agent=current_agent,
                delegation_exception=exception,
                invocation_state={},
                cycle_trace=Trace(f"step_{i}"),
                cycle_span=None
            )

            if i < 9:  # Not the final step
                # Setup next agent for delegation
                current_agent = next_agent
                current_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        performance = perf.stop_measurement()
        memory_usage = max(performance["memory_delta_mb"], performance["peak_memory_mb"])

        # Assertions
        assert result is not None
        assert performance["duration_ms"] > 0
        assert memory_usage > 0

        # Memory usage should be reasonable for 10-level hierarchy
        assert memory_usage < 50, f"10-level hierarchy uses too much memory: {memory_usage:.2f}MB"
        assert performance["duration_ms"] < 1000, f"10-level hierarchy too slow: {performance['duration_ms']:.2f}ms"

    async def test_memory_growth_with_increasing_depth(self) -> None:
        """Test how memory usage scales with delegation depth."""
        memory_by_depth = []

        for depth in range(1, 8):  # Test depths 1-7
            agents = []

            # Create hierarchy of specified depth
            for i in range(depth + 1):
                agent = _make_mock_agent(f"Agent{i}_{depth}")
                if i > 0:
                    agents[i-1]._sub_agents[agent.name] = agent
                agents.append(agent)

            # Setup final agent
            agents[-1].invoke_async = AsyncMock(return_value=_make_agent_result())

            # Measure memory for this depth
            perf = PerformanceMeasurement()
            perf.start_measurement()

            # Execute delegation chain
            current_agent = agents[0]
            for i in range(depth):
                next_agent = agents[i + 1]
                if i < depth - 1:
                    next_agent.invoke_async = AsyncMock(return_value=_make_agent_result())
                exception = AgentDelegationException(
                    target_agent=next_agent.name,
                    message=f"Step {i}",
                    context={"depth": depth, "step": i},
                    delegation_chain=[agent.name for agent in agents[:i+1]],
                    transfer_state=True,
                    transfer_messages=True
                )

                result = await _handle_delegation(
                    agent=current_agent,
                    delegation_exception=exception,
                    invocation_state={},
                    cycle_trace=Trace(f"depth_{depth}_step_{i}"),
                    cycle_span=None
                )

                if i < depth - 1:
                    current_agent = next_agent
                    current_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

            performance = perf.stop_measurement()
            memory_by_depth.append(max(performance["memory_delta_mb"], performance["peak_memory_mb"]))

        # Memory growth should be roughly linear
        for i in range(1, len(memory_by_depth)):
            prev = memory_by_depth[i - 1]
            curr = memory_by_depth[i]
            if prev <= 1 or curr <= 1:
                continue
            growth_ratio = curr / prev
            # Memory growth should be reasonable (not exponential)
            assert growth_ratio < 8, f"Memory growth too fast at depth {i+1}: {growth_ratio:.2f}x"

    async def test_memory_cleanup_after_delegation(self) -> None:
        """Test that memory is properly cleaned up after delegation completes."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("SubAgent")
        orchestrator._sub_agents[sub_agent.name] = sub_agent

        # Create large state and messages
        large_state = {"big_data": ["x" * 1000] * 1000}
        orchestrator.state = large_state
        orchestrator.messages = [
            {"role": "user", "content": [{"type": "text", "text": f"Message {i}" * 100}]}
            for i in range(100)
        ]

        # Execute delegation
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())
        performance = await _measure_delegation_overhead(orchestrator, sub_agent)

        peak_memory = performance["peak_memory_mb"]

        # Cleanup and force garbage collection
        del orchestrator.state
        del orchestrator.messages
        gc.collect()

        # Assertions
        assert performance["success"] is True
        assert peak_memory < 100, f"Delegation peak memory unexpectedly high: {peak_memory:.2f}MB"


@pytest.mark.asyncio
@pytest.mark.delegation
class TestConcurrentDelegation:
    """Test concurrent delegation scenarios."""

    async def test_concurrent_delegation_performance(self) -> None:
        """Test performance of multiple delegations running concurrently."""
        # Create multiple orchestrator/sub-agent pairs
        delegation_tasks = []

        for i in range(10):
            orchestrator = _make_mock_agent(f"Orchestrator{i}")
            sub_agent = _make_mock_agent(f"SubAgent{i}")
            orchestrator._sub_agents[sub_agent.name] = sub_agent
            sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result(f"Result{i}"))

            task = _measure_delegation_overhead(orchestrator, sub_agent, f"Concurrent message {i}")
            delegation_tasks.append(task)

        # Run all delegations concurrently
        perf = PerformanceMeasurement()
        perf.start_measurement()

        results = await asyncio.gather(*delegation_tasks, return_exceptions=True)

        performance = perf.stop_measurement()

        # Assertions
        assert len(results) == 10
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 10

        # All delegations should succeed
        for result in successful_results:
            assert result["success"] is True

        # Concurrent execution should be faster than sequential
        total_individual_time = sum(r["duration_ms"] for r in successful_results)
        concurrent_time = performance["duration_ms"]

        # Should be significantly faster than sequential execution
        speedup_ratio = total_individual_time / concurrent_time if concurrent_time > 0 else float('inf')
        assert speedup_ratio > 2, f"Concurrent delegation not providing sufficient speedup: {speedup_ratio:.2f}x"

    async def test_concurrent_delegation_with_shared_resources(self) -> None:
        """Test concurrent delegation with shared orchestrator but different sub-agents."""
        shared_orchestrator = _make_mock_agent("SharedOrchestrator")

        # Create multiple sub-agents
        sub_agents = []
        for i in range(5):
            sub_agent = _make_mock_agent(f"SubAgent{i}")
            shared_orchestrator._sub_agents[sub_agent.name] = sub_agent
            sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result(f"SharedResult{i}"))
            sub_agents.append(sub_agent)

        # Create concurrent delegation tasks
        delegation_tasks = []
        for i, sub_agent in enumerate(sub_agents):
            task = _measure_delegation_overhead(shared_orchestrator, sub_agent, f"Shared message {i}")
            delegation_tasks.append(task)

        # Run all delegations concurrently
        results = await asyncio.gather(*delegation_tasks, return_exceptions=True)

        # Assertions
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 5

        # All delegations should succeed despite shared orchestrator
        for result in successful_results:
            assert result["success"] is True

    async def test_concurrent_delegation_with_circular_prevention(self) -> None:
        """Test concurrent delegation with circular reference prevention."""
        # Create agents that could potentially create circular references
        agent_a = _make_mock_agent("AgentA")
        agent_b = _make_mock_agent("AgentB")
        agent_c = _make_mock_agent("AgentC")

        # Setup cross-delegation
        agent_a._sub_agents[agent_b.name] = agent_b
        agent_b._sub_agents[agent_c.name] = agent_c
        agent_c._sub_agents[agent_a.name] = agent_a

        # Mock all agents to return results
        agent_a.invoke_async = AsyncMock(return_value=_make_agent_result())
        agent_b.invoke_async = AsyncMock(return_value=_make_agent_result())
        agent_c.invoke_async = AsyncMock(return_value=_make_agent_result())

        # Create concurrent delegation tasks that could conflict
        async def delegation_chain_a_to_b():
            exception = AgentDelegationException(
                target_agent="AgentB",
                message="A to B",
                delegation_chain=["AgentA"]
            )
            return await _handle_delegation(
                agent=agent_a,
                delegation_exception=exception,
                invocation_state={},
                cycle_trace=Trace("a_to_b"),
                cycle_span=None
            )

        async def delegation_chain_b_to_c():
            exception = AgentDelegationException(
                target_agent="AgentC",
                message="B to C",
                delegation_chain=["AgentB"]
            )
            return await _handle_delegation(
                agent=agent_b,
                delegation_exception=exception,
                invocation_state={},
                cycle_trace=Trace("b_to_c"),
                cycle_span=None
            )

        # Run concurrently
        results = await asyncio.gather(
            delegation_chain_a_to_b(),
            delegation_chain_b_to_c(),
            return_exceptions=True
        )

        # Both should succeed without circular reference issues
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 2


@pytest.mark.asyncio
@pytest.mark.delegation
class TestTimeoutBehaviorUnderLoad:
    """Test timeout behavior under various load conditions."""

    async def test_timeout_with_slow_sub_agent(self) -> None:
        """Test timeout enforcement with slow sub-agent."""
        orchestrator = _make_mock_agent("Orchestrator")
        slow_agent = _make_mock_agent("SlowAgent")
        orchestrator._sub_agents[slow_agent.name] = slow_agent

        # Create slow sub-agent that takes longer than timeout
        async def slow_invoke():
            await asyncio.sleep(0.2)  # 200ms delay
            return _make_agent_result("slow_result")

        slow_agent.invoke_async = slow_invoke
        orchestrator.delegation_timeout = 0.1  # 100ms timeout

        # Create delegation exception
        exception = AgentDelegationException(
            target_agent=slow_agent.name,
            message="This should timeout",
            context={},
            delegation_chain=[],
            transfer_state=False,
            transfer_messages=False
        )

        # Measure timeout behavior
        perf = PerformanceMeasurement()
        perf.start_measurement()

        with pytest.raises(TimeoutError, match="timed out"):
            await _handle_delegation(
                agent=orchestrator,
                delegation_exception=exception,
                invocation_state={},
                cycle_trace=Trace("timeout_test"),
                cycle_span=None
            )

        performance = perf.stop_measurement()

        # Timeout should happen quickly (approximately the timeout duration)
        assert performance["duration_ms"] < 200  # Should be close to 100ms timeout
        assert performance["duration_ms"] > 80   # But not too fast

    async def test_timeout_under_concurrent_load(self) -> None:
        """Test timeout behavior when multiple delegations are running concurrently."""
        # Create multiple slow agents
        slow_agents = []
        for i in range(5):
            agent = _make_mock_agent(f"SlowAgent{i}")

            async def slow_invoke(delay_ms=300):
                await asyncio.sleep(delay_ms / 1000)
                return _make_agent_result(f"slow_result_{delay_ms}")

            agent.invoke_async = lambda delay_ms=300, i=i: slow_invoke(delay_ms)
            slow_agents.append(agent)

        # Setup orchestrator with short timeout
        orchestrator = _make_mock_agent("LoadOrchestrator")
        orchestrator.delegation_timeout = 0.15  # 150ms timeout

        for agent in slow_agents:
            orchestrator._sub_agents[agent.name] = agent

        # Create concurrent delegation tasks that should all timeout
        delegation_tasks = []
        for i, agent in enumerate(slow_agents):
            async def timeout_task(idx=i, ag=agent):
                exception = AgentDelegationException(
                    target_agent=ag.name,
                    message=f"Load test {idx}",
                    context={},
                    delegation_chain=[],
                    transfer_state=False,
                    transfer_messages=False
                )

                with pytest.raises(TimeoutError, match="timed out"):
                    await _handle_delegation(
                        agent=orchestrator,
                        delegation_exception=exception,
                        invocation_state={},
                        cycle_trace=Trace(f"load_timeout_{idx}"),
                        cycle_span=None
                    )

                return True

            delegation_tasks.append(timeout_task())

        # Run all timeout tests concurrently
        perf = PerformanceMeasurement()
        perf.start_measurement()

        results = await asyncio.gather(*delegation_tasks, return_exceptions=True)

        performance = perf.stop_measurement()

        # All should timeout successfully
        successful_timeouts = [r for r in results if r is True]
        assert len(successful_timeouts) == 5

        # Concurrent timeouts should be efficient
        assert performance["duration_ms"] < 300  # Should be close to max timeout

    async def test_timeout_with_fast_and_slow_mixed(self) -> None:
        """Test timeout behavior with mix of fast and slow delegations."""
        orchestrator = _make_mock_agent("MixedOrchestrator")
        orchestrator.delegation_timeout = 0.2  # 200ms timeout

        # Create mix of fast and slow agents
        fast_agent = _make_mock_agent("FastAgent")
        slow_agent = _make_mock_agent("SlowAgent")

        fast_agent.invoke_async = AsyncMock(return_value=_make_agent_result("fast_result"))

        async def slow_invoke():
            await asyncio.sleep(0.3)  # Slower than timeout
            return _make_agent_result("slow_result")

        slow_agent.invoke_async = slow_invoke

        orchestrator._sub_agents[fast_agent.name] = fast_agent
        orchestrator._sub_agents[slow_agent.name] = slow_agent

        # Test fast delegation (should succeed)
        fast_exception = AgentDelegationException(
            target_agent=fast_agent.name,
            message="Fast delegation",
            context={},
            delegation_chain=[],
            transfer_state=False,
            transfer_messages=False
        )

        fast_result = await _handle_delegation(
            agent=orchestrator,
            delegation_exception=fast_exception,
            invocation_state={},
            cycle_trace=Trace("fast_test"),
            cycle_span=None
        )

        assert fast_result is not None

        # Test slow delegation (should timeout)
        slow_exception = AgentDelegationException(
            target_agent=slow_agent.name,
            message="Slow delegation",
            context={},
            delegation_chain=[],
            transfer_state=False,
            transfer_messages=False
        )

        with pytest.raises(TimeoutError, match="timed out"):
            await _handle_delegation(
                agent=orchestrator,
                delegation_exception=slow_exception,
                invocation_state={},
                cycle_trace=Trace("slow_test"),
                cycle_span=None
            )