from typing import Any

from typing_extensions import assert_type

from strands import Agent, LocalAgent, ToolContext, tool
from strands.experimental.bidi import BidiAgent
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent


@tool(context=True)
def unparameterized_tool(tool_context: ToolContext) -> str:
    assert_type(tool_context.agent, Any)
    return "unparameterized"


@tool(context=True)
def agent_tool(tool_context: ToolContext[Agent]) -> str:
    assert_type(tool_context.agent, Agent)
    return tool_context.agent.name


@tool(context=True)
def local_agent_tool(tool_context: ToolContext[LocalAgent]) -> str:
    assert_type(tool_context.agent, LocalAgent)
    return tool_context.agent.name


def before_tool_call(event: BeforeToolCallEvent) -> None:
    assert_type(event.agent, Agent)


def before_local_tool_call(event: BeforeToolCallEvent[LocalAgent]) -> None:
    assert_type(event.agent, LocalAgent)


async def after_local_tool_call(event: AfterToolCallEvent[LocalAgent]) -> None:
    assert_type(event.agent, LocalAgent)


def local_tool_call(event: BeforeToolCallEvent[LocalAgent] | AfterToolCallEvent[LocalAgent]) -> None:
    assert_type(event.agent, LocalAgent)


def register_hooks(agent: Agent, bidi_agent: BidiAgent, local_agent: LocalAgent) -> None:
    shared_agent: LocalAgent = agent
    assert_type(shared_agent, LocalAgent)
    shared_bidi_agent: LocalAgent = bidi_agent
    assert_type(shared_bidi_agent, LocalAgent)

    agent.add_hook(before_tool_call)
    agent.add_hook(before_local_tool_call)
    agent.add_hook(after_local_tool_call)
    agent.add_hook(local_tool_call)

    bidi_agent.add_hook(before_local_tool_call)
    bidi_agent.add_hook(after_local_tool_call)
    bidi_agent.add_hook(local_tool_call)

    local_agent.add_hook(before_local_tool_call)
    local_agent.add_hook(after_local_tool_call)
    local_agent.add_hook(local_tool_call)

    local_agent.add_hook(before_local_tool_call, BeforeToolCallEvent)
    local_agent.add_hook(local_tool_call, [BeforeToolCallEvent, AfterToolCallEvent])


def local_agent_excludes_agent_only_members(local_agent: LocalAgent) -> None:
    local_agent.cleanup()  # type: ignore[attr-defined]
