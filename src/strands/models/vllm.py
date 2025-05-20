import json
import logging
from typing import Any, Iterable, Optional

import requests
from typing_extensions import TypedDict, Unpack, override

from ..types.content import Messages
from ..types.models import Model
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec

logger = logging.getLogger(__name__)


class VLLMModel(Model):
    class VLLMConfig(TypedDict, total=False):
        model_id: str
        temperature: Optional[float]
        top_p: Optional[float]
        max_tokens: Optional[int]
        stop_sequences: Optional[list[str]]
        additional_args: Optional[dict[str, Any]]

    def __init__(self, host: str, **model_config: Unpack[VLLMConfig]) -> None:
        self.config = VLLMModel.VLLMConfig(**model_config)
        self.host = host.rstrip("/")
        logger.debug("----Initializing vLLM provider with config: %s", self.config)

    @override
    def update_config(self, **model_config: Unpack[VLLMConfig]) -> None:
        self.config.update(model_config)

    @override
    def get_config(self) -> VLLMConfig:
        return self.config

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        def format_message(message: dict[str, Any], content: dict[str, Any]) -> dict[str, Any]:
            if "text" in content:
                return {"role": message["role"], "content": content["text"]}
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
            return {"role": message["role"], "content": json.dumps(content)}

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
        choice = event.get("choices", [{}])[0]

        # Streaming delta (streaming mode)
        if "delta" in choice:
            delta = choice["delta"]
            if "content" in delta:
                return {"contentBlockDelta": {"delta": {"text": delta["content"]}}}
            if "tool_calls" in delta:
                return {"toolCall": delta["tool_calls"][0]}

        # Non-streaming response
        if "message" in choice:
            return {"contentBlockDelta": {"delta": {"text": choice["message"].get("content", "")}}}

        # Completion stop
        if "finish_reason" in choice:
            return {"messageStop": {"stopReason": choice["finish_reason"] or "end_turn"}}

        return {}

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Stream from /v1/chat/completions, print content, and yield chunks including tool calls."""
        headers = {"Content-Type": "application/json"}
        url = f"{self.host}/v1/chat/completions"
        request["stream"] = True

        try:
            with requests.post(url, headers=headers, data=json.dumps(request), stream=True) as response:
                if response.status_code != 200:
                    logger.error("vLLM server error: %d - %s", response.status_code, response.text)
                    raise Exception(f"Request failed: {response.status_code} - {response.text}")

                yield {"chunk_type": "message_start"}
                yield {"chunk_type": "content_start", "data_type": "text"}

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    if line.startswith("data: "):
                        line = line[len("data: ") :]

                    if line.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(line)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        tool_calls = delta.get("tool_calls")

                        if content:
                            print(content, end="", flush=True)
                            yield {
                                "chunk_type": "content_delta",
                                "data_type": "text",
                                "data": content,
                            }

                        if tool_calls:
                            for tool_call in tool_calls:
                                tool_call_id = tool_call.get("id")
                                func = tool_call.get("function", {})
                                tool_name = func.get("name", "")
                                args_text = func.get("arguments", "")

                                yield {
                                    "toolCallStart": {
                                        "toolCallId": tool_call_id,
                                        "toolName": tool_name,
                                        "type": "function",
                                    }
                                }
                                yield {
                                    "toolCallDelta": {
                                        "toolCallId": tool_call_id,
                                        "delta": {
                                            "toolName": tool_name,
                                            "argsText": args_text,
                                        },
                                    }
                                }

                    except json.JSONDecodeError:
                        logger.warning("Failed to decode streamed line: %s", line)

                yield {"chunk_type": "content_stop", "data_type": "text"}
                yield {"chunk_type": "message_stop", "data": "end_turn"}

        except requests.RequestException as e:
            logger.error("Request to vLLM failed: %s", str(e))
            raise Exception("Failed to reach vLLM server") from e
