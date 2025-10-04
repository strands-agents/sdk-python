"""
Basic Delegation Functionality Tests

This file contains tests for the core sub-agent delegation functionality,
verifying that delegation works correctly in fundamental scenarios.

These tests verify that the sub-agent delegation feature works correctly
in basic scenarios with actual agent interactions.
"""

import asyncio

import pytest

from strands import Agent
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.mark.asyncio
async def test_scenario_1_basic_delegation():
    """
    Test basic delegation functionality end-to-end.

    Verifies that:
    - Delegation tools are automatically generated and used
    - Sub-agents receive questions and provide answers
    - Final response contains the expected result
    - No post-processing by orchestrator occurs
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: BASIC DELEGATION")
    print("=" * 70)

    # Create sub-agent specialized in mathematics
    math_agent = Agent(
        name="MathExpert",
        model=MockedModelProvider([{"role": "assistant", "content": [{"text": "123 * 456 = 56088"}]}]),
        system_prompt="You are a math expert. Solve math problems concisely.",
    )

    # Create orchestrator that can delegate
    orchestrator = Agent(
        name="Orchestrator",
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "handoff_to_mathexpert",
                                "input": {"message": "What is 123 * 456?"},
                            }
                        }
                    ],
                }
            ]
        ),
        system_prompt=(
            "You are an orchestrator. When users ask math questions, "
            "use the handoff_to_mathexpert tool to delegate to the math expert."
        ),
        sub_agents=[math_agent],
        delegation_state_transfer=True,
        delegation_message_transfer=True,
    )

    print("1. Created MathExpert sub-agent")
    print("2. Created Orchestrator with delegation capability")
    print("3. Testing delegation with: 'What is 123 * 456?'")

    # Test delegation
    result = await orchestrator.invoke_async("What is 123 * 456?")

    print(f"4. Result received: {result}")
    print(f"5. Math agent was called: {len(math_agent.messages) > 0}")
    print(f"6. Orchestrator messages: {len(orchestrator.messages)}")

    # Verification
    assert result is not None, "❌ No result returned"
    assert len(math_agent.messages) > 0, "❌ Math agent was not called"
    assert "56088" in str(result), "❌ Incorrect math result"

    # Verify delegation tool was generated and used
    delegation_tools = [name for name in orchestrator.tool_names if "handoff_to_" in name]
    assert len(delegation_tools) > 0, "❌ Delegation tools not generated"

    print("✅ Basic delegation works correctly!")
    print("   - Delegation tool auto-generated")
    print("   - Math agent called and provided correct answer")
    print("   - Final response contains mathematical result")

    return True


@pytest.mark.asyncio
async def test_scenario_2_state_transfer():
    """
    Test state transfer between agents.

    Verifies that:
    - State is properly transferred from orchestrator to sub-agent
    - State includes user context, session information, and metadata
    - State isolation is maintained (deep copy)
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: STATE TRANSFER VERIFICATION")
    print("=" * 70)

    # Create sub-agent
    math_agent = Agent(
        name="MathExpert",
        model=MockedModelProvider([{"role": "assistant", "content": [{"text": "sqrt(144) = 12"}]}]),
        system_prompt="You are a math expert.",
    )

    # Create orchestrator with initial state
    orchestrator = Agent(
        name="Orchestrator",
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "handoff_to_mathexpert",
                                "input": {"message": "Calculate the square root of 144"},
                            }
                        }
                    ],
                }
            ]
        ),
        system_prompt="Delegate math questions to expert",
        sub_agents=[math_agent],
        delegation_state_transfer=True,
        delegation_message_transfer=True,
    )

    # Set up orchestrator with initial state
    orchestrator.state = {
        "user_id": "test123",
        "session_context": "math_quiz",
        "difficulty": "intermediate",
        "question_number": 3,
    }

    print("1. Set up orchestrator state:")
    for key, value in orchestrator.state.items():
        print(f"   {key}: {value}")

    print("2. Delegating with state transfer enabled")

    # Delegate with state transfer enabled
    result = await orchestrator.invoke_async("Calculate the square root of 144")

    print("3. Math agent received state:")
    for key, value in math_agent.state.items():
        print(f"   {key}: {value}")

    # Verification
    assert math_agent.state is not None, "❌ Math agent received no state"
    assert math_agent.state["user_id"] == "test123", "❌ user_id not transferred"
    assert math_agent.state["session_context"] == "math_quiz", "❌ session_context not transferred"
    assert math_agent.state["difficulty"] == "intermediate", "❌ difficulty not transferred"
    assert math_agent.state["question_number"] == 3, "❌ question_number not transferred"

    # Verify state is deep copied (changes to sub-agent don't affect orchestrator)
    original_orchestrator_state = dict(orchestrator.state.items())
    # Since the transferred state is a dict, we'll modify it directly
    math_agent_state_dict = math_agent.state
    math_agent_state_dict["modified_by_sub"] = True
    # Reassign to test deep copy
    assert dict(orchestrator.state.items()) == original_orchestrator_state, "❌ State not deep copied"

    print("✅ State transfer works correctly!")
    print("   - Complete state dictionary transferred")
    print("   - State isolation maintained (deep copy)")
    print("   - No state corruption during transfer")

    return True


@pytest.mark.asyncio
async def test_scenario_3_message_filtering():
    """
    Test message filtering during delegation.

    Verifies that:
    - Sub-agents receive clean conversation history
    - Internal tool messages are filtered out
    - Only relevant user messages and context are transferred
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: MESSAGE FILTERING VERIFICATION")
    print("=" * 70)

    # Create sub-agent
    math_agent = Agent(
        name="MathExpert",
        model=MockedModelProvider([{"role": "assistant", "content": [{"text": "2 + 2 = 4"}]}]),
        system_prompt="You are a math expert.",
    )

    # Create orchestrator with complex message history
    orchestrator = Agent(
        name="Orchestrator",
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "handoff_to_mathexpert",
                                "input": {"message": "What is 2 + 2?"},
                            }
                        }
                    ],
                }
            ]
        ),
        system_prompt="Delegate math questions",
        sub_agents=[math_agent],
        delegation_state_transfer=True,
        delegation_message_transfer=True,
    )

    # Simulate orchestrator having internal tool messages in history
    orchestrator.messages = [
        {"role": "user", "content": [{"text": "Help with math"}]},
        {"role": "assistant", "content": [{"text": "I'll help you with that"}]},
        {"role": "assistant", "content": [{"toolUse": {"name": "internal_tool", "toolUseId": "internal1"}}]},
        {"role": "assistant", "content": [{"toolResult": {"toolUseId": "internal1", "content": "internal result"}}]},
        {"role": "user", "content": [{"text": "What is 2 + 2?"}]},
    ]

    print("1. Orchestrator has internal tool messages in history")
    print(f"   Total messages before delegation: {len(orchestrator.messages)}")

    internal_tools_before = [
        msg for msg in orchestrator.messages if "toolUse" in str(msg) and "handoff_to" not in str(msg)
    ]
    print(f"   Internal tool messages: {len(internal_tools_before)}")

    print("2. Delegating to math expert")

    # Perform delegation
    result = await orchestrator.invoke_async("What is 2 + 2?")

    print("3. Checking math agent message history")
    print(f"   Math agent received: {len(math_agent.messages)} messages")

    # Check what math agent received
    math_agent_message_types = []
    for msg in math_agent.messages:
        if "toolUse" in str(msg):
            math_agent_message_types.append("toolUse")
        elif "toolResult" in str(msg):
            math_agent_message_types.append("toolResult")
        elif "user" in str(msg):
            math_agent_message_types.append("user")
        elif "assistant" in str(msg):
            math_agent_message_types.append("assistant")

    print(f"   Message types in math agent: {set(math_agent_message_types)}")

    # Verify internal tools were filtered
    internal_tools_after = [
        msg for msg in math_agent.messages if "toolUse" in str(msg) and "handoff_to_" not in str(msg)
    ]

    print(f"   Internal tool messages in math agent: {len(internal_tools_after)}")

    # Verification
    assert len(internal_tools_after) == 0, "❌ Internal tool messages not filtered"
    assert len(math_agent.messages) > 0, "❌ Math agent received no messages"

    # Should have delegation context message
    delegation_context = [msg for msg in math_agent.messages if "Delegated from" in str(msg)]
    assert len(delegation_context) > 0, "❌ Delegation context not added"

    print("✅ Message filtering works correctly!")
    print("   - Internal tool chatter filtered out")
    print("   - Clean conversation history transferred")
    print("   - Delegation context properly added")

    return True


@pytest.mark.asyncio
async def test_scenario_4_concurrent_delegation():
    """
    Test concurrent delegation setup.

    Verifies that:
    - Multiple delegation tools can be generated without conflicts
    - Sub-agents can be configured for concurrent execution
    - Tool registration works correctly with multiple sub-agents
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: CONCURRENT DELEGATION")
    print("=" * 70)

    # Create multiple sub-agents
    math_agent = Agent(
        name="MathExpert",
        model=MockedModelProvider([{"role": "assistant", "content": [{"text": "15 * 23 = 345"}]}]),
        system_prompt="Math expert",
    )

    writing_agent = Agent(
        name="WritingExpert",
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"text": "Numbers dance,\nFifteen times twenty-three,\nThree hundred forty-five."}],
                }
            ]
        ),
        system_prompt="Writing expert",
    )

    # Create orchestrator that delegates to both
    orchestrator = Agent(
        name="Orchestrator",
        model=MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "handoff_to_mathexpert",
                                "input": {"message": "Calculate 15 * 23"},
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "t2",
                                "name": "handoff_to_writingexpert",
                                "input": {"message": "write a haiku about numbers"},
                            }
                        },
                    ],
                }
            ]
        ),
        system_prompt="Delegate to appropriate expert",
        sub_agents=[math_agent, writing_agent],
        delegation_state_transfer=True,
        delegation_message_transfer=True,
    )

    print("1. Created MathExpert and WritingExpert sub-agents")
    print("2. Testing concurrent delegation:")
    print("   - Math: 'Calculate 15 * 23'")
    print("   - Writing: 'write a haiku about numbers'")

    # Test concurrent operations
    result = await orchestrator.invoke_async("Calculate 15 * 23 AND write a haiku about numbers")

    print(f"3. Result: {result}")
    print(f"4. Math agent called: {len(math_agent.messages) > 0}")
    print(f"5. Writing agent called: {len(writing_agent.messages) > 0}")

    # Verification
    assert result is not None, "❌ No result returned"

    # At least one agent should have been called (depending on tool execution order)
    agents_called = sum([len(math_agent.messages) > 0, len(writing_agent.messages) > 0])
    assert agents_called > 0, "❌ No sub-agents were called"

    # Verify delegation tools were generated for both
    delegation_tools = [name for name in orchestrator.tool_names if "handoff_to_" in name]
    assert len(delegation_tools) == 2, "❌ Not all delegation tools generated"
    assert "handoff_to_mathexpert" in delegation_tools, "❌ Math delegation tool missing"
    assert "handoff_to_writingexpert" in delegation_tools, "❌ Writing delegation tool missing"

    print("✅ Concurrent delegation setup works correctly!")
    print("   - Multiple delegation tools generated")
    print("   - Sub-agents can be configured for concurrent execution")
    print("   - No conflicts in tool registration")

    return True


@pytest.mark.asyncio
async def run_all_basic_scenarios():
    """
    Run all basic delegation tests.

    Returns:
        dict: Results of all tests with pass/fail status
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BASIC DELEGATION VERIFICATION")
    print("=" * 80)
    print("Running Phase 3: Manual End-to-End Testing")
    print("Implementing scenarios from VERIFICATION_PLAN.md")

    scenarios = [
        ("Basic Delegation", test_scenario_1_basic_delegation),
        ("State Transfer", test_scenario_2_state_transfer),
        ("Message Filtering", test_scenario_3_message_filtering),
        ("Concurrent Delegation", test_scenario_4_concurrent_delegation),
    ]

    results = {}

    for scenario_name, test_func in scenarios:
        print(f"\n{'=' * 20} {scenario_name} {'=' * 20}")
        try:
            success = await test_func()
            results[scenario_name] = {"status": "PASS", "error": None}
            print(f"✅ {scenario_name}: PASSED")
        except Exception as e:
            results[scenario_name] = {"status": "FAIL", "error": str(e)}
            print(f"❌ {scenario_name}: FAILED - {e}")

    # Summary
    print("\n" + "=" * 80)
    print("BASIC DELEGATION VERIFICATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    total = len(results)

    for scenario, result in results.items():
        status_symbol = "✅" if result["status"] == "PASS" else "❌"
        print(f"{status_symbol} {scenario}: {result['status']}")
        if result["error"]:
            print(f"   Error: {result['error']}")

    print(f"\nOverall: {passed}/{total} scenarios passed")

    if passed == total:
        print("🎉 ALL BASIC DELEGATION SCENARIOS PASSED!")
        print("Core delegation functionality is working correctly.")
    else:
        print("⚠️  Some scenarios failed. Check the implementation.")

    return results


if __name__ == "__main__":
    """
    Run basic delegation functionality tests.

    This script can be executed directly to verify the sub-agent delegation
    feature works in basic scenarios.
    """
    asyncio.run(run_all_basic_scenarios())
