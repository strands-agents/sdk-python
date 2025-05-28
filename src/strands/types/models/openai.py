"""Base OpenAI model provider.

This module provides the base OpenAI model provider class which implements shared logic for formatting requests and
responses to and from the OpenAI specification.

- Docs: https://pypi.org/project/openai
"""

import abc
import base64
import json
import logging
import mimetypes
from typing import Any, Optional

from typing_extensions import override

from ..content import ContentBlock, Messages
from ..streaming import StreamEvent
from ..tools import ToolResult, ToolSpec, ToolUse
from .model import Model

logger = logging.getLogger(__name__)


class OpenAIModel(Model, abc.ABC):
    """Base OpenAI model provider implementation.

    Implements shared logic for formatting requests and responses to and from the OpenAI specification.
    """

    config: dict[str, Any]

    @staticmethod
    def format_request_message_content(content: ContentBlock) -> dict[str, Any]:
        """Format an OpenAI compatible content block.

        Args:
            content: Message content.

        Returns:
            OpenAI compatible content block.
        """
        if "document" in content:
            mime_type = mimetypes.types_map.get(f".{content['document']['format']}", "application/octet-stream")
            file_data = base64.b64encode(content["document"]["source"]["bytes"]).decode("utf-8")
            return {
                "file": {
                    "file_data": f"data:{mime_type};base64,{file_data}",
                    "filename": content["document"]["name"],
                },
                "type": "file",
            }

        if "image" in content:
            mime_type = mimetypes.types_map.get(f".{content['image']['format']}", "application/octet-stream")
            image_data = content["image"]["source"]["bytes"].decode("utf-8")
            return {
                "image_url": {
                    "detail": "auto",
                    "format": mime_type,
                    "url": f"data:{mime_type};base64,{image_data}",
                },
                "type": "image_url",
            }

        if "text" in content:
            return {"text": content["text"], "type": "text"}

        return {"text": json.dumps(content), "type": "text"}

    @staticmethod
    def format_request_message_tool_call(tool_use: ToolUse) -> dict[str, Any]:
        """Format an OpenAI compatible tool call.

        Args:
            tool_use: Tool use requested by the model.

        Returns:
            OpenAI compatible tool call.
        """
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    @staticmethod
    def format_request_tool_message(tool_result: ToolResult) -> dict[str, Any]:
        """Format an OpenAI compatible tool message.

        Args:
            tool_result: Tool result collected from a tool execution.

        Returns:
            OpenAI compatible tool message.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": json.dumps(
                {
                    "content": tool_result["content"],
                    "status": tool_result["status"],
                }
            ),
        }

    @classmethod
    def format_request_messages(cls, messages: Messages, system_prompt: Optional[str] = None) -> list[dict[str, Any]]:
        """Format an OpenAI compatible messages array.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            An OpenAI compatible messages array.
        """
        formatted_messages: list[dict[str, Any]]
        formatted_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []

        for message in messages:
            contents = message["content"]

            formatted_contents = [
                cls.format_request_message_content(content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse"])
            ]
            formatted_tool_calls = [
                cls.format_request_message_tool_call(content["toolUse"]) for content in contents if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls.format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message = {
                "role": message["role"],
                "content": formatted_contents,
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

    @override
    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format an OpenAI compatible chat streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            An OpenAI compatible chat streaming request.
        """
        return {
            "messages": self.format_request_messages(messages, system_prompt),
            "model": self.config["model_id"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs or []
            ],
            **(self.config.get("params") or {}),
        }

    @override
    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format an OpenAI response event into a standardized message chunk.

        Args:
            event: A response event from the OpenAI compatible model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
                This error should never be encountered as chunk_type is controlled in the stream method.
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": event["data"].function.name,
                                    "toolUseId": event["data"].id,
                                }
                            }
                        }
                    }

                return {"contentBlockStart": {"start": {}}}

            case "content_delta":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockDelta": {"delta": {"toolUse": {"input": event["data"].function.arguments or ""}}}
                    }

                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

            case "content_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                match event["data"]:
                    case "tool_calls":
                        return {"messageStop": {"stopReason": "tool_use"}}
                    case "length":
                        return {"messageStop": {"stopReason": "max_tokens"}}
                    case _:
                        return {"messageStop": {"stopReason": "end_turn"}}

            case "metadata":
                return {
                    "metadata": {
                        "usage": {
                            "inputTokens": event["data"].prompt_tokens,
                            "outputTokens": event["data"].completion_tokens,
                            "totalTokens": event["data"].total_tokens,
                        },
                        "metrics": {
                            "latencyMs": 0,  # TODO
                        },
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']} | unknown type")
