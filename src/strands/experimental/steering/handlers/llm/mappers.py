"""LLM steering prompt mappers for generating evaluation prompts.

Extends the original mappers module with model output steering support.
"""

import json
from typing import Any, Protocol

from .....types.content import Message
from .....types.streaming import StopReason
from .....types.tools import ToolUse
from ...core.context import SteeringContext

# Agent SOP format - see https://github.com/strands-agents/agent-sop
_STEERING_PROMPT_TEMPLATE = """# Steering Evaluation

## Overview

You are a STEERING AGENT that evaluates a {action_type} that ANOTHER AGENT is attempting to make.
Your job is to provide contextual guidance to help the other agent navigate workflows effectively.
You act as a safety net that can intervene when patterns in the context data suggest the agent
should try a different approach or get human input.

**YOUR ROLE:**
- Analyze context data for concerning patterns (repeated failures, inappropriate timing, etc.)
- Provide just-in-time guidance when the agent is going down an ineffective path
- Allow normal operations to proceed when context shows no issues

**CRITICAL CONSTRAINTS:**
- Base decisions ONLY on the context data provided below
- Do NOT use external knowledge about domains, URLs, or tool purposes
- Do NOT make assumptions about what tools "should" or "shouldn't" do
- Focus ONLY on patterns in the context data

## Context

{context_str}

### Understanding Ledger Tool States

If the context includes a ledger with tool_calls, the "status" field indicates:

- **"pending"**: The tool is CURRENTLY being evaluated by you (the steering agent).
This is NOT a duplicate call - it's the tool you're deciding whether to approve.
The tool has NOT started executing yet.
- **"success"**: The tool completed successfully in a previous turn
- **"error"**: The tool failed or was cancelled in a previous turn

**IMPORTANT**: When you see a tool with status="pending" that matches the tool you're evaluating,
that IS the current tool being evaluated.
It is NOT already executing or a duplicate.

## Event to Evaluate

{event_description}

## Steps

### 1. Analyze the {action_type_title}

Review ONLY the context data above. Look for patterns in the data that indicate:

- Previous failures or successes with this tool
- Frequency of attempts
- Any relevant tracking information

**Constraints:**
- You MUST base analysis ONLY on the provided context data
- You MUST NOT use external knowledge about tool purposes or domains
- You SHOULD identify patterns in the context data
- You MAY reference relevant context data to inform your decision

### 2. Make Steering Decision

**Constraints:**
- You MUST respond with exactly one of: "proceed", "guide", or "interrupt"
- You MUST base the decision ONLY on context data patterns
- Your reason will be shown to the AGENT as guidance

**Decision Options:**
- "proceed" if context data shows no concerning patterns
- "guide" if context data shows patterns requiring intervention
- "interrupt" if context data shows patterns requiring human input
"""


class LLMPromptMapper(Protocol):
    """Protocol for mapping context and events to LLM evaluation prompts."""

    def create_steering_prompt(
        self, steering_context: SteeringContext, tool_use: ToolUse | None = None, **kwargs: Any
    ) -> str:
        """Create steering prompt for LLM evaluation.

        Args:
            steering_context: Steering context with populated data
            tool_use: Tool use object for tool call events (None for other events)
            **kwargs: Additional event data for other steering events

        Returns:
            Formatted prompt string for LLM evaluation
        """
        ...


class DefaultPromptMapper(LLMPromptMapper):
    """Default prompt mapper for steering evaluation."""

    def create_steering_prompt(
        self, steering_context: SteeringContext, tool_use: ToolUse | None = None, **kwargs: Any
    ) -> str:
        """Create default steering prompt using Agent SOP structure.

        Uses Agent SOP format for structured, constraint-based prompts.
        See: https://github.com/strands-agents/agent-sop
        """
        context_str = (
            json.dumps(steering_context.data.get(), indent=2) if steering_context.data.get() else "No context available"
        )

        if tool_use:
            event_description = (
                f"Tool: {tool_use['name']}\nArguments: {json.dumps(tool_use.get('input', {}), indent=2)}"
            )
            action_type = "tool call"
        else:
            event_description = "General evaluation"
            action_type = "action"

        return _STEERING_PROMPT_TEMPLATE.format(
            action_type=action_type,
            action_type_title=action_type.title(),
            context_str=context_str,
            event_description=event_description,
        )


# Agent SOP format for model output evaluation
_MODEL_STEERING_PROMPT_TEMPLATE = """# Steering Evaluation

## Overview

You are a STEERING AGENT that evaluates a {action_type} that ANOTHER AGENT has produced.
Your job is to check whether the agent's output follows the instructions it was given.
You act as a quality gate that catches instruction violations before the response is returned.

**YOUR ROLE:**
- Compare the agent's output against its system prompt instructions
- Allow output that follows the instructions to proceed without modification
- Reject output that violates any explicit instruction and explain what to fix

**CRITICAL CONSTRAINTS:**
- Base decisions ONLY on the agent's system prompt and the output provided below
- Do NOT inject your own style preferences or opinions
- Do NOT reject output just because you would have written it differently
- Focus ONLY on clear violations of explicit instructions in the system prompt

## Context

{context_str}

## Agent's System Prompt

{agent_system_prompt}

## Event to Evaluate

{event_description}

## Steps

### 1. Analyze the {action_type_title}

Review the agent's system prompt and compare it against the model output. Look for:

- Direct violations of explicit rules or constraints in the system prompt
- Content that contradicts specific instructions
- Patterns the system prompt explicitly prohibits
- Required elements that are missing from the output

**Constraints:**
- You MUST only check rules stated in the agent's system prompt
- You MUST NOT apply rules that are not in the system prompt
- You SHOULD quote the specific violation when rejecting
- You MAY reference the relevant system prompt rule to justify your decision

### 2. Make Steering Decision

**Constraints:**
- You MUST respond with exactly one of: "proceed" or "guide"
- You MUST quote the specific violation if rejecting
- Your reason will be shown to the AGENT as guidance for a rewrite

**Decision Options:**
- "proceed" if the output follows the agent's system prompt instructions
- "guide" if the output violates a specific instruction (explain what to fix)
"""


class LLMModelPromptMapper(Protocol):
    """Protocol for mapping model output to an LLM evaluation prompt."""

    def create_steering_prompt(
        self,
        steering_context: SteeringContext,
        message: Message,
        stop_reason: StopReason,
        agent_system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Create steering prompt for model output evaluation.

        Args:
            steering_context: Steering context with populated data
            message: The model's generated message
            stop_reason: The reason the model stopped generating
            agent_system_prompt: The agent's system prompt (instructions to check against)
            **kwargs: Additional event data

        Returns:
            Formatted prompt string for LLM evaluation
        """
        ...


class DefaultModelPromptMapper(LLMModelPromptMapper):
    """Default prompt mapper for model output steering evaluation.

    Uses the Agent SOP template with the event description containing the
    agent's system prompt and model output. The steering agent checks
    whether the output follows the instructions.
    """

    def create_steering_prompt(
        self,
        steering_context: SteeringContext,
        message: Message,
        stop_reason: StopReason,
        agent_system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Create model output steering prompt using Agent SOP structure."""
        context_str = (
            json.dumps(steering_context.data.get(), indent=2) if steering_context.data.get() else "No context available"
        )

        # Extract text from message
        text_parts = []
        for block in message.get("content", []):
            if "text" in block:
                text_parts.append(block["text"])
        model_output = "\n".join(text_parts) if text_parts else "(no text content)"

        system_prompt_str = agent_system_prompt or "(system prompt not available)"

        event_description = f"Model Output:\n{model_output}"

        return _MODEL_STEERING_PROMPT_TEMPLATE.format(
            action_type="model response",
            action_type_title="Model Response",
            context_str=context_str,
            agent_system_prompt=system_prompt_str,
            event_description=event_description,
        )
