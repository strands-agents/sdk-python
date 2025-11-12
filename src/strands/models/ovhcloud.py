"""OVHcloud AI Endpoints model provider.

- Docs: https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/
- API Keys: https://ovh.com/manager (Public Cloud > AI & Machine Learning > AI Endpoints)
"""

import json
import logging
from typing import Any, Optional, TypedDict, cast

from typing_extensions import Unpack, override

from ..types.content import Messages
from ..types.tools import ToolResult
from ._validation import validate_config_keys
from .openai import OpenAIModel

logger = logging.getLogger(__name__)


class OVHcloudModel(OpenAIModel):
    """OVHcloud AI Endpoints model provider implementation.

    OVHcloud AI Endpoints provides OpenAI-compatible API access to various models.
    The service can be used for free with rate limits when no API key is provided,
    or with an API key for higher rate limits.

    To generate an API key:
    1. Go to https://ovh.com/manager
    2. Navigate to Public Cloud > AI & Machine Learning > AI Endpoints
    3. Create an API key

    For a complete list of available models, see:
    https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/
    """

    class OVHcloudConfig(TypedDict, total=False):
        """Configuration options for OVHcloud AI Endpoints models.

        Attributes:
            model_id: Model ID (e.g., "gpt-oss-120b", "gpt-oss-20b", "Qwen3-32B").
                For a complete list of supported models, see
                https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/.
            params: Model parameters (e.g., max_tokens, temperature).
                For a complete list of supported parameters, see OpenAI API documentation.
        """

        model_id: str
        params: Optional[dict[str, Any]]

    def __init__(self, client_args: Optional[dict[str, Any]] = None, **model_config: Unpack[OVHcloudConfig]) -> None:
        """Initialize OVHcloud AI Endpoints provider instance.

        Args:
            client_args: Arguments for the OpenAI client.
                The base_url is automatically set to the OVHcloud endpoint.
                If api_key is not provided or is an empty string, the service will
                be used with free tier rate limits.
                For a complete list of supported arguments, see https://pypi.org/project/openai/.
            **model_config: Configuration options for the OVHcloud model.
        """
        validate_config_keys(model_config, self.OVHcloudConfig)
        self.config = dict(model_config)

        # Set up client args with OVHcloud base URL
        self.client_args = client_args or {}
        self.client_args.setdefault("base_url", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1")

        # Handle API key: if not provided or empty string, set to empty string (free tier)
        # OVHcloud supports free tier usage with an empty API key
        # The OpenAI client requires api_key to be set, so we use empty string for free tier
        api_key = self.client_args.get("api_key")
        if api_key is None or api_key == "":
            # Set to empty string for free tier (OVHcloud accepts this)
            self.client_args["api_key"] = ""

        logger.debug("config=<%s> | initializing", self.config)

    @override
    def update_config(self, **model_config: Unpack[OVHcloudConfig]) -> None:  # type: ignore[override]
        """Update the OVHcloud model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.OVHcloudConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> "OVHcloudModel.OVHcloudConfig":
        """Get the OVHcloud model configuration.

        Returns:
            The OVHcloud model configuration.
        """
        return cast(OVHcloudModel.OVHcloudConfig, self.config)

    @override
    @classmethod
    def format_request_messages(cls, messages: Messages, system_prompt: Optional[str] = None) -> list[dict[str, Any]]:
        """Format an OVHcloud AI Endpoints compatible messages array.

        This method is identical to the base OpenAIModel implementation, but suppresses
        the warning about reasoningContent since it's expected behavior and handled correctly.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            An OVHcloud AI Endpoints compatible messages array.
        """
        formatted_messages: list[dict[str, Any]]
        formatted_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []

        for message in messages:
            contents = message["content"]

            # Note: reasoningContent is filtered out silently (no warning) as it's expected
            # behavior for OpenAI-compatible APIs that don't support it in multi-turn conversations

            formatted_contents = [
                cls.format_request_message_content(content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse", "reasoningContent"])
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
    @classmethod
    def format_request_tool_message(cls, tool_result: ToolResult) -> dict[str, Any]:
        """Format an OVHcloud AI Endpoints compatible tool message.

        OVHcloud expects tool message content as a string, not a list of content blocks.
        We format the content blocks first, then extract the text to create a string.

        Args:
            tool_result: Tool result collected from a tool execution.

        Returns:
            OVHcloud AI Endpoints compatible tool message with content as a string.
        """
        # First format content blocks using the base class method
        from typing import cast

        from ..types.content import ContentBlock

        contents = cast(
            list[ContentBlock],
            [
                {"text": json.dumps(content["json"])} if "json" in content else content
                for content in tool_result["content"]
            ],
        )

        # Format each content block
        formatted_blocks = [cls.format_request_message_content(content) for content in contents]

        # Extract text from formatted blocks and join into a single string
        content_parts = []
        for block in formatted_blocks:
            if isinstance(block, dict):
                if "text" in block:
                    content_parts.append(block["text"])
                elif "type" in block and block["type"] == "text" and "text" in block:
                    content_parts.append(block["text"])
                else:
                    # Fallback: convert the whole block to string
                    content_parts.append(json.dumps(block))

        content_string = " ".join(content_parts) if content_parts else ""

        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": content_string,  # String format for OVHcloud compatibility
        }
