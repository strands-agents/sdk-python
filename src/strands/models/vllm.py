"""vLLM model provider.

- Docs: https://docs.vllm.ai/en/latest/index.html
"""
import json
import logging
import re
from collections import namedtuple
from typing import Any, Iterable, Optional

import requests
from typing_extensions import TypedDict, Unpack, override

from ..types.content import Messages
from ..types.models import Model
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec

logger = logging.getLogger(__name__)


class VLLMModel(Model):
    """vLLM model provider implementation for OpenAI compatible /v1/chat/completions endpoint."""

    class VLLMConfig(TypedDict, total=False):
        """Configuration options for vLLM models.

        Attributes:
            model_id: Model ID (e.g., "Qwen/Qwen3-4B").
            temperature: Optional[float]
            top_p: Optional[float]
            max_tokens: Optional[int]
            stop_sequences: Optional[list[str]]
            additional_args: Optional[dict[str, Any]]
        """

        model_id: str
        temperature: Optional[float]
        top_p: Optional[float]
        max_tokens: Optional[int]
        stop_sequences: Optional[list[str]]
        additional_args: Optional[dict[str, Any]]

    def __init__(self, host: str, **model_config: Unpack[VLLMConfig]) -> None:
        """Initialize provider instance.

        Args:
            host: Host and port of the vLLM Inference Server
            **model_config: Configuration options for the LiteLLM model.
        """
        self.config = VLLMModel.VLLMConfig(**model_config)
        self.host = host.rstrip("/")
        logger.debug("Initializing vLLM provider with config: %s", self.config)

    @override
    def update_config(self, **model_config: Unpack[VLLMConfig]) -> None:
        """Update the vLLM model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> VLLMConfig:
        """Get the vLLM model configuration.

        Returns:
            The vLLM model configuration.
        """
        return self.config

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Format a vLLM chat streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A vLLM chat streaming request.
        """

        def format_message(msg: dict[str, Any], content: dict[str, Any]) -> dict[str, Any]:
            if "text" in content:
                return {"role": msg["role"], "content": content["text"]}
            if "toolUse" in content:
                return {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": content["toolUse"]["toolUseId"],
                            "type": "function",
                            "function": {
                                "name": content["toolUse"]["name"],
                                "arguments": json.dumps(content["toolUse"]["input"]),
                            },
                        }
                    ],
                }
            if "toolResult" in content:
                return {
                    "role": "tool",
                    "tool_call_id": content["toolResult"]["toolUseId"],
                    "content": json.dumps(content["toolResult"]["content"]),
                }
            return {"role": msg["role"], "content": json.dumps(content)}

        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            for content in msg["content"]:
                chat_messages.append(format_message(msg, content))

        payload = {
            "model": self.config["model_id"],
            "messages": chat_messages,
            "temperature": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 1.0),
            "max_tokens": self.config.get("max_tokens", 2048),
            "stream": True,
        }

        if self.config.get("stop_sequences"):
            payload["stop"] = self.config["stop_sequences"]

        if tool_specs:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["inputSchema"]["json"],
                    },
                }
                for tool in tool_specs
            ]

        if self.config.get("additional_args"):
            payload.update(self.config["additional_args"])

        logger.debug("Formatted vLLM Request:\n%s", json.dumps(payload, indent=2))
        return payload

    @override
    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format the vLLM response events into standardized message chunks.

        Args:
            event: A response event from the vLLM model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
                This error should never be encountered as we control chunk_type in the stream method.
        """
        from collections import namedtuple

        Function = namedtuple("Function", ["name", "arguments"])

        if event.get("chunk_type") == "message_start":
            return {"messageStart": {"role": "assistant"}}

        if event.get("chunk_type") == "content_start":
            if event["data_type"] == "text":
                return {"contentBlockStart": {"start": {}}}

            tool: Function = event["data"]
            return {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "name": tool.name,
                            "toolUseId": tool.name,
                        }
                    }
                }
            }

        if event.get("chunk_type") == "content_delta":
            if event["data_type"] == "text":
                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

            tool: Function = event["data"]
            return {
                "contentBlockDelta": {
                    "delta": {
                        "toolUse": {
                            "input": json.dumps(tool.arguments)  # This is already a dict
                        }
                    }
                }
            }

        if event.get("chunk_type") == "content_stop":
            return {"contentBlockStop": {}}

        if event.get("chunk_type") == "message_stop":
            reason = event["data"]
            if reason == "tool_use":
                return {"messageStop": {"stopReason": "tool_use"}}
            elif reason == "length":
                return {"messageStop": {"stopReason": "max_tokens"}}
            else:
                return {"messageStop": {"stopReason": "end_turn"}}

        if event.get("chunk_type") == "metadata":
            usage = event.get("data", {})
            return {
                "metadata": {
                    "usage": {
                        "inputTokens": usage.get("prompt_eval_count", 0),
                        "outputTokens": usage.get("eval_count", 0),
                        "totalTokens": usage.get("prompt_eval_count", 0) + usage.get("eval_count", 0),
                    },
                    "metrics": {
                        "latencyMs": usage.get("total_duration", 0) / 1e6,
                    },
                }
            }

        raise RuntimeError(f"chunk_type=<{event.get('chunk_type')}> | unknown type")

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Send the request to the vLLM model and get the streaming response.

        Args:
            request: The formatted request to send to the vLLM model.

        Returns:
            An iterable of response events from the vLLM model.
        """

        Function = namedtuple("Function", ["name", "arguments"])

        headers = {"Content-Type": "application/json"}
        url = f"{self.host}/v1/chat/completions"

        accumulated_content = []
        tool_requested = False

        try:
            with requests.post(url, headers=headers, data=json.dumps(request), stream=True) as response:
                if response.status_code != 200:
                    logger.error("vLLM server error: %d - %s", response.status_code, response.text)
                    raise Exception(f"Request failed: {response.status_code} - {response.text}")

                yield {"chunk_type": "message_start"}
                yield {"chunk_type": "content_start", "data_type": "text"}

                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    line = line[len("data: ") :].strip()

                    if line == "[DONE]":
                        break

                    try:
                        event = json.loads(line)
                        choices = event.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                accumulated_content.append(content)

                        yield {"chunk_type": "content_delta", "data_type": "text", "data": content or ""}

                    except json.JSONDecodeError:
                        logger.warning("Failed to parse line: %s", line)
                        continue

                yield {"chunk_type": "content_stop", "data_type": "text"}

                full_content = "".join(accumulated_content)

                tool_call_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", full_content, re.DOTALL)
                for idx, block in enumerate(tool_call_blocks):
                    try:
                        tool_call_data = json.loads(block.strip())
                        func = Function(name=tool_call_data["name"], arguments=tool_call_data.get("arguments", {}))
                        func_str = f"function=Function(name='{func.name}', arguments={func.arguments})"

                        yield {"chunk_type": "content_start", "data_type": "tool", "data": func}
                        yield {"chunk_type": "content_delta", "data_type": "tool", "data": func}
                        yield {"chunk_type": "content_stop", "data_type": "tool", "data": func}
                        tool_requested = True

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool_call block #{idx}: {block}")
                        continue

                yield {"chunk_type": "message_stop", "data": "tool_use" if tool_requested else "end_turn"}

        except requests.RequestException as e:
            logger.error("Streaming request failed: %s", str(e))
            raise Exception("Failed to reach vLLM server") from e
