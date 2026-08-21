"""Reasoner protocol and built-in implementations."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Reasoner(Protocol):
    """Protocol for reasoning implementations.

    Args:
        messages: Conversation messages (USER/ASSISTANT turns).
            Format: [{"role": "USER", "content": [{"text": "..."}]}, ...]

    Yields:
        Text chunks (sentences) to stream back for TTS.
    """

    async def reason(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream reasoning response sentence by sentence."""
        ...


class BedrockConverseReasoner:
    """Reasoner backed by Bedrock's ConverseStream API."""

    def __init__(
        self,
        model: Any,
        system_prompt: str = "You are a helpful voice assistant.",
        tools: list[Any] | None = None,
        max_iterations: int = 5,
    ):
        """Initialize BedrockConverseReasoner.

        Args:
            model: A Strands BedrockModel instance.
            system_prompt: System prompt for the reasoning LLM.
            tools: List of @tool decorated functions.
            max_iterations: Max tool call loops per turn.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self._tools = tools or []
        self._tool_specs = self._build_tool_specs(self._tools)
        self._tool_executors = self._build_tool_executors(self._tools)
        self._conversation_history: list[dict[str, Any]] = []

    async def reason(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream reasoning with automatic tool calling loops.

        Maintains conversation history across invocations.
        """
        logger.debug("reasoner_model=<%s> | invoked", self.model.config["model_id"])

        new_messages = self._convert_messages(messages)
        self._conversation_history.extend(new_messages)
        working_messages = list(self._conversation_history)

        for _ in range(self.max_iterations):
            full_text = ""
            buffer = ""
            tool_requests: list[dict[str, Any]] = []
            current_tool: dict[str, Any] | None = None

            request_kwargs = self._build_request(working_messages)
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda kw=request_kwargs: self.model.client.converse_stream(**kw)
            )

            for event in response.get("stream", []):
                if "contentBlockStart" in event:
                    start = event["contentBlockStart"].get("start", {})
                    if "toolUse" in start:
                        current_tool = {
                            "toolUseId": start["toolUse"]["toolUseId"],
                            "name": start["toolUse"]["name"],
                            "_raw": "",
                        }
                elif "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        token = delta["text"]
                        full_text += token
                        buffer += token
                        while "\n" in buffer:
                            sentence, buffer = buffer.split("\n", 1)
                            if sentence.strip():
                                yield sentence.strip()
                    elif "toolUse" in delta and current_tool is not None:
                        inp = delta["toolUse"].get("input", "")
                        if isinstance(inp, str):
                            current_tool["_raw"] += inp
                elif "contentBlockStop" in event:
                    if current_tool is not None:
                        raw = current_tool.pop("_raw", "{}")
                        try:
                            current_tool["input"] = json.loads(raw)
                        except json.JSONDecodeError:
                            current_tool["input"] = {}
                        tool_requests.append(current_tool)
                        current_tool = None

            if buffer.strip():
                yield buffer.strip()

            if not tool_requests:
                if full_text:
                    self._conversation_history.append({"role": "assistant", "content": [{"text": full_text}]})
                return

            assistant_content: list[dict[str, Any]] = []
            if full_text:
                assistant_content.append({"text": full_text})
            for tr in tool_requests:
                assistant_content.append(
                    {"toolUse": {"toolUseId": tr["toolUseId"], "name": tr["name"], "input": tr["input"]}}
                )
            working_messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for tr in tool_requests:
                result = await self._execute_tool(tr["name"], tr["input"])
                tool_results.append({"toolResult": {"toolUseId": tr["toolUseId"], "content": [{"text": result}]}})
            working_messages.append({"role": "user", "content": tool_results})

        yield "I'm sorry, I wasn't able to complete that request."

    def _build_request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "modelId": self.model.config["model_id"],
            "system": [{"text": self.system_prompt}],
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.model.config.get("max_tokens", 1024),
                "temperature": self.model.config.get("temperature", 0.7),
            },
        }
        if self._tool_specs:
            kwargs["toolConfig"] = {"tools": self._tool_specs}
        if self.model.config.get("guardrail_id") and self.model.config.get("guardrail_version"):
            kwargs["guardrailConfig"] = {
                "guardrailIdentifier": self.model.config["guardrail_id"],
                "guardrailVersion": self.model.config["guardrail_version"],
                "trace": self.model.config.get("guardrail_trace", "enabled"),
            }
        return kwargs

    def _convert_messages(self, sonic_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Bedrock Converse format, stripping leading assistant messages."""
        converted = []
        for msg in sonic_messages:
            role = msg.get("role", "USER").lower()
            bedrock_content = [{"text": b["text"]} for b in msg.get("content", []) if "text" in b]
            if bedrock_content:
                converted.append({"role": role, "content": bedrock_content})
        while converted and converted[0]["role"] == "assistant":
            converted.pop(0)
        return converted

    async def _execute_tool(self, name: str, input_data: dict[str, Any]) -> str:
        executor = self._tool_executors.get(name)
        if executor:
            try:
                result = executor(**input_data)
                return result if isinstance(result, str) else json.dumps(result)
            except Exception as e:
                logger.error("Tool execution error: %s - %s", name, e)
                return json.dumps({"error": str(e)})
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _build_tool_specs(self, tools: list[Any]) -> list[dict[str, Any]]:
        specs = []
        for t in tools:
            if hasattr(t, "tool_spec"):
                specs.append({"toolSpec": t.tool_spec})
            elif hasattr(t, "TOOL_SPEC"):
                specs.append({"toolSpec": t.TOOL_SPEC})
        return specs

    def _build_tool_executors(self, tools: list[Any]) -> dict[str, Any]:
        executors: dict[str, Any] = {}
        for t in tools:
            if hasattr(t, "tool_spec"):
                executors[t.tool_spec.get("name", "")] = t
            elif hasattr(t, "TOOL_SPEC"):
                executors[t.TOOL_SPEC.get("name", "")] = t
        return executors


class StrandsAgentReasoner:
    """Wraps a Strands Agent as a Reasoner."""

    def __init__(self, agent: Any):
        """Initialize with a Strands Agent instance.

        Args:
            agent: A Strands Agent instance (with model, tools, system_prompt configured).
        """
        self.agent = agent

    async def reason(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Invoke the Strands Agent and yield response sentences.

        Args:
            messages: Conversation messages.

        Yields:
            Text sentences from the agent response.
        """
        user_text = self._extract_latest_user_text(messages)
        if not user_text:
            yield "I didn't catch that. Could you say it again?"
            return

        result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.agent(user_text))

        response_text = str(result)
        for sentence in self._split_sentences(response_text):
            yield sentence

    def _extract_latest_user_text(self, messages: list[dict[str, Any]]) -> str:
        """Extract the most recent user text from messages."""
        for m in reversed(messages):
            if m.get("role", "").upper() == "USER":
                parts = m.get("content", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
        return ""

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences for TTS-friendly streaming."""
        sentences = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped:
                sentences.append(stripped)
        return sentences if sentences else [text.strip()]
