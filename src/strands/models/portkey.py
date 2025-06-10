"""Implementation of the Portkey model provider integration."""

import json
import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, cast

from portkey_ai import Portkey
from typing_extensions import TypedDict, override

from ..types.content import Messages
from ..types.exceptions import ContextWindowOverflowException
from ..types.models import Model
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec

# Configure logger for debug-level output
logger = logging.getLogger(__name__)


class PortkeyModel(Model):
    """Portkey model provider implementation."""

    class PortkeyConfig(TypedDict, total=False):
        """Configuration schema for the Portkey model."""

        api_key: str
        virtual_key: str
        base_url: str
        model_id: str
        provider: str
        streaming: bool

    def __init__(self, **model_config: PortkeyConfig):
        """Initialize the Portkey model provider.

        Sets up the model configuration and initializes the Portkey client.

        Args:
            **model_config (PortkeyConfig): Configuration parameters for the model.
        """
        self.config = PortkeyModel.PortkeyConfig()
        self.config["streaming"] = True
        self.update_config(**model_config)

        # Extract provider(bedrock, openai, anthropic, etc) from model_config or infer from model_id.
        self.provider: str = str(model_config["provider"])

        logger.debug("PortkeyModel initialized with config: %s", self.config)

        self.client = Portkey(
            api_key=self.config["api_key"],
            virtual_key=self.config["virtual_key"],
            base_url=self.config["base_url"],
            model=self.config["model_id"],
        )
        self._current_tool_use_id: Optional[str] = None
        self._current_tool_name: Optional[str] = None
        self._current_tool_args = ""

    @override
    def update_config(self, **model_config: PortkeyConfig) -> None:
        """Update the model configuration.

        Args:
            **model_config (PortkeyConfig): Configuration parameters to update.
        """
        logger.debug("Updating config with: %s", model_config)
        self.config.update(cast(PortkeyModel.PortkeyConfig, model_config))

    @override
    def get_config(self) -> PortkeyConfig:
        """Retrieve the current model configuration.

        Returns:
            PortkeyConfig: The current configuration dictionary.
        """
        logger.debug("Retrieving current model config")
        return self.config

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format the input messages and tool specifications into a request dictionary.

        Prepares the messages, system prompt, and tool specifications into the format
        required by the Portkey client for streaming chat completions.

        Args:
            messages (Messages): List of messages to format.
            tool_specs (Optional[List[ToolSpec]]): Optional list of tool specifications.
            system_prompt (Optional[str]): Optional system prompt string.

        Returns:
            Dict[str, Any]: Formatted request dictionary.
        """
        formatted_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                formatted_messages.extend(self._format_message_parts(role, content))

        if system_prompt:
            formatted_messages.insert(0, {"role": "system", "content": system_prompt})

        request = {
            "messages": formatted_messages,
            "model": self.config["model_id"],
            "stream": True,
        }

        allow_tools = self._allow_tool_use()

        if tool_specs and allow_tools:
            tool_calls = self._map_tools(tool_specs)
        else:
            tool_calls = None

        if tool_calls:
            request["tools"] = tool_calls
            request["tool_choice"] = "auto"
        logger.debug("Formatted Portkey request: %s", json.dumps(request, default=str)[:300])
        return request

    def _allow_tool_use(self) -> bool:
        """Determine whether tool use is allowed based on provider and model.

        Returns:
            bool: True if tool use is allowed for the current provider and model.
        """
        provider = str(self.provider).lower()
        if provider == "openai":
            return True
        if provider == "bedrock":
            model_id = self.config.get("model_id", "").lower()
            return "anthropic" in model_id
        return False

    @override
    def stream(self, request: Dict[str, Any]) -> Iterable[Any]:
        """Stream responses from the Portkey client based on the request.

        Args:
            request (Dict[str, Any]): The formatted request dictionary.

        Returns:
            Iterable[Any]: An iterable stream of response events.

        Raises:
            ContextWindowOverflowException: If the context window is exceeded.
        """
        try:
            return iter(self.client.chat.completions.create(**request))
        except ContextWindowOverflowException:
            logger.error("Context window exceeded for request: %s", request)
            raise

    @override
    def format_chunk(self, event: Any) -> StreamEvent:
        """Format a single response event into a stream event for Strands Agents.

        Converts the raw event from the Portkey client into the structured stream event
        format expected downstream.

        Args:
            event (Any): The raw response event from the model.

        Returns:
            StreamEvent: The formatted stream event dictionary.
        """
        choice = event.get("choices", [{}])[0]
        delta = choice.get("delta", {})

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call.get("function", {}).get("name")
            call_type = tool_call.get("type")
            arguments_chunk = tool_call.get("function", {}).get("arguments", "")
            if tool_name and call_type and not self._current_tool_name:
                self._current_tool_name = tool_name
                self._current_tool_use_id = f"{tool_name}-{uuid.uuid4().hex[:6]}"
                self._current_tool_args = arguments_chunk
                return cast(
                    StreamEvent,
                    {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": self._current_tool_name,
                                    "toolUseId": self._current_tool_use_id,
                                }
                            }
                        }
                    },
                )

            if arguments_chunk:
                return cast(StreamEvent, {"contentBlockDelta": {"delta": {"toolUse": {"input": arguments_chunk}}}})

        if choice.get("finish_reason") == "tool_calls" or choice.get("finish_reason") == "tool_use":
            return cast(
                StreamEvent,
                {
                    "contentBlockStop": {
                        "name": self._current_tool_name,
                        "toolUseId": self._current_tool_use_id,
                    }
                },
            )

        if delta.get("content"):
            return cast(StreamEvent, {"contentBlockDelta": {"delta": {"text": delta["content"]}}})
        elif event.get("usage"):
            usage_data = event["usage"]
            return cast(
                StreamEvent,
                {
                    "metadata": {
                        "metrics": {"latencyMs": 0},
                        "usage": {
                            "inputTokens": usage_data["prompt_tokens"],
                            "outputTokens": usage_data["completion_tokens"],
                            "totalTokens": usage_data["total_tokens"],
                        },
                    }
                },
            )
        return cast(StreamEvent, {})

    @override
    def converse(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterable[StreamEvent]:
        """Converse with the model by streaming formatted message chunks.

        Handles the full lifecycle of conversing with the model, including formatting
        the request, sending it, and yielding formatted response chunks.

        Args:
            messages (Messages): List of message objects to be processed by the model.
            tool_specs (Optional[list[ToolSpec]]): List of tool specifications available to the model.
            system_prompt (Optional[str]): System prompt to provide context to the model.

        Yields:
            Iterable[StreamEvent]: Formatted message chunks from the model.

        Raises:
            ModelThrottledException: When the model service is throttling requests from the client.
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt)

        logger.debug("invoking model %s", request)
        response = self.stream(request)
        logger.debug("streaming response from model %s", response)

        yield cast(StreamEvent, {"messageStart": {"role": "assistant"}})

        for event in response:
            yield self.format_chunk(event)

            if self._should_terminate_with_tool_use(event):
                yield cast(StreamEvent, {"messageStop": {"stopReason": "tool_use"}})
                logger.debug("finished streaming response from model")

        self._current_tool_use_id = None
        self._current_tool_name = None
        self._current_tool_args = ""

    @staticmethod
    def _should_terminate_with_tool_use(event: dict) -> bool:
        """Determine whether the stream should terminate due to a tool use.

        This accounts for inconsistencies across providers: some may return a 'tool_calls'
        payload but label the finish_reason as 'stop' instead of 'tool_calls'.

        Args:
            event (dict): The raw event from the model.

        Returns:
            bool: True if the event indicates a tool use termination.
        """
        choice = event.get("choices", [{}])[0]
        finish_reason = (choice.get("finish_reason") or "").lower()
        return finish_reason in ["tool_calls", "tool_use"]

    def _format_tool_use_part(self, part: dict) -> dict:
        """Format a tool use part of a message into the standard dictionary format.

        Args:
            part (dict): The part of the message representing a tool use.

        Returns:
            dict: Formatted dictionary representing the tool use.
        """
        logger.debug("Formatting tool use part: %s", part)
        self._current_tool_use_id = part["toolUse"]["toolUseId"]
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": self._current_tool_use_id,
                    "type": "function",
                    "function": {"name": part["toolUse"]["name"], "arguments": json.dumps(part["toolUse"]["input"])},
                }
            ],
            "content": None,
        }

    def _format_tool_result_part(self, part: dict) -> dict:
        """Format a tool result part of a message into the standard dictionary format.

        Args:
            part (dict): The part of the message representing a tool result.

        Returns:
            dict: Formatted dictionary representing the tool result.
        """
        logger.debug("Formatting tool result part: %s", part)
        result_text = " ".join([c["text"] for c in part["toolResult"]["content"] if "text" in c])
        return {"role": "tool", "tool_call_id": self._current_tool_use_id, "content": result_text}

    def _format_message_parts(self, role: str, content: Any) -> List[Dict[str, Any]]:
        """Format message parts into a list of standardized message dictionaries.

        Handles plain text content as well as structured parts including tool uses and results.

        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (Any): The content of the message, can be string or list of parts.

        Returns:
            List[Dict[str, Any]]: List of formatted message dictionaries.
        """
        logger.debug("Formatting message parts for role '%s' with content: %s", role, content)
        parts = []
        if isinstance(content, str):
            parts.append({"role": role, "content": content})
        elif isinstance(content, list):
            for part in content:
                if "text" in part and isinstance(part["text"], str):
                    parts.append({"role": role, "content": part["text"]})
                elif "toolUse" in part:
                    parts.append(self._format_tool_use_part(part))
                elif "toolResult" in part and self._current_tool_use_id:
                    parts.append(self._format_tool_result_part(part))
        return parts

    @staticmethod
    def _map_tools(tool_specs: List[ToolSpec]) -> List[Dict[str, Any]]:
        """Map tool specifications to the format expected by Portkey.

        Args:
            tool_specs (List[ToolSpec]): List of tool specifications.

        Returns:
            List[Dict[str, Any]]: Mapped list of tool dictionaries.
        """
        logger.debug("Mapping tool specs: %s", tool_specs)
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            k: {key: value for key, value in v.items() if key != "default" or value is not None}
                            for k, v in spec["inputSchema"]["json"].get("properties", {}).items()
                        },
                        "required": spec["inputSchema"]["json"].get("required", []),
                    },
                },
            }
            for spec in tool_specs
        ]
