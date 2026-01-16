"""xAI model provider.

- Docs: https://docs.x.ai/docs

This module implements the xAI model provider using the native xAI Python SDK (xai-sdk).
The xAI SDK uses a gRPC-based Chat API pattern:

    from xai_sdk import Client
    from xai_sdk.chat import system, user

    client = Client(api_key="...")
    chat = client.chat.create(model="grok-4", store_messages=False)
    chat.append(system("You are helpful"))
    chat.append(user("Hello"))
    response = chat.sample()  # or: for response, chunk in chat.stream()

Server-Side State Preservation
==============================

xAI's server-side tools (x_search, web_search, code_execution) and encrypted reasoning
content present a unique challenge: their results are returned in an encrypted format
that cannot be reconstructed from plain text. To maintain multi-turn conversation
context, we must preserve this encrypted state across turns.

The solution uses Strands' `reasoningContent.redactedContent` field to store serialized
xAI SDK messages. This field is designed for encrypted/hidden content and is NOT rendered
when printing AgentResult, keeping the output clean for users.

Flow:
1. After each response with server-side tools or encrypted reasoning, we capture the
   SDK's internal protobuf messages (which contain encrypted tool results)
2. Serialize these messages to base64 and wrap them with XAI_STATE markers
3. Store in `reasoningContent.redactedContent` - this field is preserved in message
   history but NOT displayed to users (unlike `text` content blocks)
4. On the next turn, extract and deserialize these messages to rebuild the xAI chat
   with full encrypted context

Why `reasoningContent.redactedContent` instead of `text`?
- `text` content blocks are rendered in AgentResult.__str__(), showing ugly markers
- `redactedContent` is designed for encrypted content and is NOT rendered
- The Strands event loop already handles accumulating `redactedContent` properly
- This keeps the user-facing output clean while preserving internal state
"""

import base64
import json
import logging
import mimetypes
from typing import Any, AsyncGenerator, Optional, Type, TypedDict, TypeVar, Union

import pydantic
from typing_extensions import Required, Unpack, override

from ..types.content import Messages
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._validation import validate_config_keys
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=pydantic.BaseModel)

try:
    from xai_sdk import AsyncClient
    from xai_sdk.chat import chat_pb2 as xai_chat_pb2
    from xai_sdk.chat import image as xai_image
    from xai_sdk.chat import system as xai_system
    from xai_sdk.chat import tool as xai_tool
    from xai_sdk.chat import tool_result as xai_tool_result
    from xai_sdk.chat import user as xai_user
    from xai_sdk.tools import get_tool_call_type
except ImportError as e:
    raise ImportError(
        "The 'xai-sdk' package is required to use xAIModel. Install it with: pip install strands-agents[grok]"
    ) from e

# Markers for xAI state serialization.
# The state is stored in reasoningContent.redactedContent to keep it hidden from users.
# Format: <!--XAI_STATE:{base64_encoded_json}:XAI_STATE-->
# The JSON contains {"messages": [base64_encoded_protobuf_messages...]}
XAI_STATE_MARKER = "<!--XAI_STATE:"
XAI_STATE_MARKER_END = ":XAI_STATE-->"


class xAIModel(Model):
    """xAI model provider implementation.

    This provider uses the native xAI Python SDK (xai-sdk) which provides a gRPC-based
    conversational API with features including server-side agentic tools (web_search,
    x_search, code_execution), reasoning models, and stateful conversations.

    - Docs: https://docs.x.ai/docs
    """

    class xAIConfig(TypedDict, total=False):
        """Configuration options for xAI models.

        Attributes:
            model_id: xAI model ID (e.g., "grok-4", "grok-4-fast", "grok-3-mini").
            params: Additional model parameters (e.g., temperature, max_tokens).
            xai_tools: xAI server-side tools (web_search, x_search, code_execution).
            reasoning_effort: Reasoning effort level ("low" or "high") for grok-3-mini.
            include: Optional xAI features (e.g., ["inline_citations", "verbose_streaming"]).
            use_encrypted_content: Return encrypted reasoning for multi-turn context (grok-4).
        """

        model_id: Required[str]
        params: dict[str, Any]
        xai_tools: list[Any]
        reasoning_effort: str
        include: list[str]
        use_encrypted_content: bool

    def __init__(
        self,
        *,
        client: Optional[AsyncClient] = None,
        client_args: Optional[dict[str, Any]] = None,
        **model_config: Unpack[xAIConfig],
    ) -> None:
        """Initialize provider instance.

        Args:
            client: Pre-configured AsyncClient to reuse across requests.
            client_args: Arguments for the underlying xAI client (e.g., api_key, timeout).
            **model_config: Configuration options for the Grok model.

        Raises:
            ValueError: If both `client` and `client_args` are provided.
        """
        validate_config_keys(model_config, xAIModel.xAIConfig)
        self.config = xAIModel.xAIConfig(**model_config)

        if client is not None and client_args is not None and len(client_args) > 0:
            raise ValueError("Only one of 'client' or 'client_args' should be provided, not both.")

        self._custom_client = client
        self.client_args = client_args or {}

        if "xai_tools" in self.config:
            self._validate_xai_tools(self.config["xai_tools"])
            # Auto-enable encrypted content when using server-side tools
            # This is required to preserve server-side tool state across turns
            if not self.config.get("use_encrypted_content"):
                self.config["use_encrypted_content"] = True
                logger.debug("auto-enabled use_encrypted_content for server-side tool state preservation")

        logger.debug("config=<%s> | initializing", self.config)

    def _get_client(self) -> AsyncClient:
        """Get an xAI AsyncClient for making requests."""
        if self._custom_client is not None:
            return self._custom_client
        return AsyncClient(**self.client_args)

    @staticmethod
    def _validate_xai_tools(xai_tools: list[Any]) -> None:
        """Validate that xai_tools contains only server-side tools."""
        for tool in xai_tools:
            if isinstance(tool, dict) and tool.get("type") == "function":
                raise ValueError(
                    "xai_tools should not contain function-based tools. "
                    "Use the standard tools interface for function calling tools."
                )

    def _format_request_tools(self, tool_specs: Optional[list[ToolSpec]]) -> list[Any]:
        """Format tool specs into xAI SDK compatible tools."""
        tools: list[Any] = []

        for tool_spec in tool_specs or []:
            tools.append(
                xai_tool(
                    name=tool_spec["name"],
                    description=tool_spec["description"],
                    parameters=tool_spec["inputSchema"]["json"],
                )
            )

        if self.config.get("xai_tools"):
            tools.extend(self.config["xai_tools"])

        return tools

    def _format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format xAI response events into standardized StreamEvent format."""
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                if event.get("data_type") == "tool":
                    tool_data = event["data"]
                    return {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": tool_data["name"],
                                    "toolUseId": tool_data["id"],
                                }
                            }
                        }
                    }
                return {"contentBlockStart": {"start": {}}}

            case "content_delta":
                if event.get("data_type") == "tool":
                    return {"contentBlockDelta": {"delta": {"toolUse": {"input": event["data"].get("arguments", "")}}}}
                if event.get("data_type") == "reasoning_content":
                    return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": event["data"]}}}}
                if event.get("data_type") == "server_tool":
                    tool_data = event["data"]
                    tool_text = f"\n[xAI Tool: {tool_data['name']}({tool_data.get('arguments', '{}')})]\n"
                    return {"contentBlockDelta": {"delta": {"text": tool_text}}}
                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

            case "content_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                match event.get("data"):
                    case "tool_use":
                        return {"messageStop": {"stopReason": "tool_use"}}
                    case "max_tokens" | "length":
                        return {"messageStop": {"stopReason": "max_tokens"}}
                    case _:
                        return {"messageStop": {"stopReason": "end_turn"}}

            case "metadata":
                usage_data = event["data"]
                metadata_event: StreamEvent = {
                    "metadata": {
                        "usage": {
                            "inputTokens": getattr(usage_data, "prompt_tokens", 0),
                            "outputTokens": getattr(usage_data, "completion_tokens", 0),
                            "totalTokens": getattr(usage_data, "total_tokens", 0),
                        },
                        "metrics": {"latencyMs": 0},
                    }
                }
                if hasattr(usage_data, "reasoning_tokens") and usage_data.reasoning_tokens:
                    metadata_event["metadata"]["usage"]["reasoningTokens"] = usage_data.reasoning_tokens  # type: ignore[typeddict-unknown-key]
                if event.get("citations"):
                    metadata_event["metadata"]["citations"] = event["citations"]  # type: ignore[typeddict-unknown-key]
                if event.get("server_tool_calls"):
                    metadata_event["metadata"]["serverToolCalls"] = event["server_tool_calls"]  # type: ignore[typeddict-unknown-key]
                return metadata_event

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']}> | unknown type")

    def _build_chat(self, client: AsyncClient, tool_specs: Optional[list[ToolSpec]] = None) -> Any:
        """Build a chat instance with the configured parameters."""
        chat_kwargs: dict[str, Any] = {
            "model": self.config["model_id"],
            "store_messages": False,
        }

        tools = self._format_request_tools(tool_specs)
        if tools:
            chat_kwargs["tools"] = tools

        if self.config.get("reasoning_effort"):
            chat_kwargs["reasoning_effort"] = self.config["reasoning_effort"]

        if self.config.get("include"):
            chat_kwargs["include"] = self.config["include"]

        if self.config.get("use_encrypted_content"):
            chat_kwargs["use_encrypted_content"] = self.config["use_encrypted_content"]

        if self.config.get("params"):
            chat_kwargs.update(self.config["params"])

        logger.debug("chat_kwargs=<%s> | creating xAI chat", chat_kwargs)
        return client.chat.create(**chat_kwargs)

    def _format_image_content(self, content: dict[str, Any]) -> Any:
        """Format image content block into xAI SDK image() helper format."""
        image_data = content["image"]
        mime_type = mimetypes.types_map.get(f".{image_data['format']}", "image/png")
        b64_data = base64.b64encode(image_data["source"]["bytes"]).decode("utf-8")
        image_url = f"data:{mime_type};base64,{b64_data}"
        return xai_image(image_url=image_url, detail="auto")

    def _extract_xai_state(self, messages: Messages) -> Optional[list[bytes]]:
        """Extract serialized xAI SDK messages from Strands message history.

        This method searches for preserved xAI state in the message history. The state
        contains serialized protobuf messages from the xAI SDK that include encrypted
        server-side tool results and reasoning content.

        The state is stored in `reasoningContent.redactedContent` to keep it hidden
        from users (this field is not rendered in AgentResult.__str__). We also check
        `text` content for backwards compatibility.

        Why is this needed?
        - Server-side tools (x_search, web_search) return encrypted results
        - Encrypted reasoning (grok-4 with use_encrypted_content=True) cannot be reconstructed
        - The xAI SDK requires the original protobuf messages to maintain context
        - Without this, multi-turn conversations would lose server-side tool context

        Args:
            messages: The Strands message history to search.

        Returns:
            List of serialized protobuf message bytes if found, None otherwise.
        """
        for message in messages:
            for content in message.get("content", []):
                # Primary location: reasoningContent.redactedContent
                # This field is designed for encrypted content and is NOT rendered to users
                if "reasoningContent" in content:
                    rc = content["reasoningContent"]
                    if "redactedContent" in rc:
                        redacted: Any = rc["redactedContent"]
                        if isinstance(redacted, bytes):
                            redacted_text = redacted.decode("utf-8")
                        else:
                            redacted_text = str(redacted)
                        if redacted_text.startswith(XAI_STATE_MARKER) and redacted_text.endswith(XAI_STATE_MARKER_END):
                            # Extract base64-encoded JSON between markers
                            encoded = redacted_text[len(XAI_STATE_MARKER) : -len(XAI_STATE_MARKER_END)]
                            try:
                                state_data = json.loads(base64.b64decode(encoded).decode("utf-8"))
                                return [base64.b64decode(msg_b64) for msg_b64 in state_data.get("messages", [])]
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.warning("failed to decode xAI state from redactedContent: %s", e)

                # Fallback: check text content (for backwards compatibility with older versions)
                if "text" in content:
                    text_content = content["text"]
                    if text_content.startswith(XAI_STATE_MARKER) and text_content.endswith(XAI_STATE_MARKER_END):
                        encoded = text_content[len(XAI_STATE_MARKER) : -len(XAI_STATE_MARKER_END)]
                        try:
                            state_data = json.loads(base64.b64decode(encoded).decode("utf-8"))
                            return [base64.b64decode(msg_b64) for msg_b64 in state_data.get("messages", [])]
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning("failed to decode xAI state from text: %s", e)
        return None

    def _append_messages_to_chat(
        self,
        chat: Any,
        messages: Messages,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Append Strands messages to an xAI chat.

        This method handles two cases:
        1. If xAI state is present (from previous turns with server-side tools or
           encrypted reasoning), use the serialized protobuf messages directly to
           preserve encrypted content, then append only the new user message.
        2. Otherwise, reconstruct all messages from the Strands format.

        The first case is critical for multi-turn conversations with server-side tools
        because the encrypted tool results cannot be reconstructed from plain text.
        """
        if system_prompt:
            chat.append(xai_system(system_prompt))

        # Check for preserved xAI state (contains server-side tool results)
        xai_state = self._extract_xai_state(messages)
        if xai_state:
            logger.debug("xai_state_count=<%d> | using preserved xAI messages", len(xai_state))
            for serialized_msg in xai_state:
                msg = xai_chat_pb2.Message()
                msg.ParseFromString(serialized_msg)
                chat.append(msg)

            # Append the new user message (last message in the list)
            # The xAI state contains the old conversation, but we need to add the new input
            if messages and messages[-1]["role"] == "user":
                last_message = messages[-1]
                user_parts: list[Any] = []
                for content in last_message["content"]:
                    if "text" in content:
                        text = content["text"]
                        # Skip xAI state markers
                        if not (text.startswith(XAI_STATE_MARKER) and text.endswith(XAI_STATE_MARKER_END)):
                            user_parts.append(text)
                    elif "image" in content:
                        user_parts.append(self._format_image_content(dict(content)))

                if user_parts:
                    if len(user_parts) == 1 and isinstance(user_parts[0], str):
                        chat.append(xai_user(user_parts[0]))
                    else:
                        chat.append(xai_user(*user_parts))
                    logger.debug("appended new user message after xAI state")
            return

        # No preserved state - reconstruct from Strands format
        for message in messages:
            role = message["role"]
            contents = message["content"]

            if role == "user":
                tool_results: list[tuple[str, str]] = []
                user_parts_list: list[Any] = []

                for content in contents:
                    if "toolResult" in content:
                        tr = content["toolResult"]
                        result_parts: list[str] = []
                        for tr_content in tr["content"]:
                            if "json" in tr_content:
                                result_parts.append(json.dumps(tr_content["json"]))
                            elif "text" in tr_content:
                                result_parts.append(tr_content["text"])
                        result_str = "\n".join(result_parts) if result_parts else ""
                        tool_results.append((tr.get("toolUseId", ""), result_str))
                    elif "text" in content:
                        user_parts_list.append(content["text"])
                    elif "image" in content:
                        user_parts_list.append(self._format_image_content(dict(content)))

                for _tool_use_id, result in tool_results:
                    chat.append(xai_tool_result(result))

                if user_parts_list:
                    if len(user_parts_list) == 1 and isinstance(user_parts_list[0], str):
                        chat.append(xai_user(user_parts_list[0]))
                    else:
                        chat.append(xai_user(*user_parts_list))

            elif role == "assistant":
                assistant_msg = xai_chat_pb2.Message()
                assistant_msg.role = xai_chat_pb2.ROLE_ASSISTANT

                text_parts: list[str] = []
                reasoning_parts: list[str] = []
                encrypted_content: Optional[str] = None
                tool_uses: list[dict[str, Any]] = []

                for content in contents:
                    if "text" in content:
                        # Skip xAI state markers - they're only for state preservation
                        text = content["text"]
                        if not (text.startswith(XAI_STATE_MARKER) and text.endswith(XAI_STATE_MARKER_END)):
                            text_parts.append(text)
                    elif "reasoningContent" in content:
                        rc = content["reasoningContent"]
                        # Handle visible reasoning text (grok-3-mini)
                        if "reasoningText" in rc:
                            reasoning_text: Any = rc["reasoningText"]
                            if isinstance(reasoning_text, dict) and "text" in reasoning_text:
                                reasoning_parts.append(reasoning_text["text"])
                            elif isinstance(reasoning_text, str):
                                reasoning_parts.append(reasoning_text)
                        # Handle encrypted reasoning (grok-4 with use_encrypted_content=True)
                        if "redactedContent" in rc:
                            redacted: Any = rc["redactedContent"]
                            if isinstance(redacted, bytes):
                                encrypted_content = redacted.decode("utf-8")
                            else:
                                encrypted_content = str(redacted)
                    elif "toolUse" in content:
                        tool_use_block = content["toolUse"]
                        tool_uses.append(
                            {
                                "id": tool_use_block.get("toolUseId", ""),
                                "name": tool_use_block.get("name", ""),
                                "arguments": tool_use_block.get("input", ""),
                            }
                        )

                # Add reasoning content if present (for grok-3-mini)
                if reasoning_parts:
                    assistant_msg.reasoning_content = " ".join(reasoning_parts)

                # Add encrypted content if present (for grok-4 with use_encrypted_content)
                if encrypted_content:
                    assistant_msg.encrypted_content = encrypted_content

                if text_parts:
                    text_content = assistant_msg.content.add()
                    text_content.text = " ".join(text_parts)

                for tool_use_item in tool_uses:
                    tc = assistant_msg.tool_calls.add()
                    tc.id = tool_use_item["id"]
                    tc.type = xai_chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL
                    tc.function.name = tool_use_item["name"]
                    args = tool_use_item["arguments"]
                    if isinstance(args, dict):
                        tc.function.arguments = json.dumps(args)
                    else:
                        tc.function.arguments = str(args) if args else ""

                chat.append(assistant_msg)

    def _capture_xai_state(self, chat: Any, response: Any) -> list[str]:
        """Capture xAI SDK messages for state preservation across turns.

        This method is called after receiving a response that contains server-side
        tool results or encrypted reasoning content. It serializes the SDK's internal
        protobuf messages so they can be restored on the next turn.

        The flow:
        1. Append the response to the chat (SDK creates proper message structure)
        2. Iterate through all messages in the chat (excluding system)
        3. Serialize each message to protobuf bytes, then base64 encode

        These serialized messages contain encrypted content that cannot be reconstructed
        from plain text, which is why we must preserve them exactly as-is.

        Args:
            chat: The xAI chat instance with the conversation.
            response: The response to append before capturing state.

        Returns:
            List of base64-encoded serialized protobuf messages.
        """
        chat.append(response)

        serialized_messages: list[str] = []
        for msg in chat.messages:
            # Skip system messages - they're added fresh each turn
            if msg.role == xai_chat_pb2.ROLE_SYSTEM:
                continue
            serialized = msg.SerializeToString()
            serialized_messages.append(base64.b64encode(serialized).decode("utf-8"))

        logger.debug("captured_messages=<%d> | preserved xAI state", len(serialized_messages))
        return serialized_messages

    @override
    def update_config(self, **model_config: Unpack[xAIConfig]) -> None:  # type: ignore[override]
        """Update the Grok model configuration."""
        validate_config_keys(model_config, xAIModel.xAIConfig)
        if "xai_tools" in model_config:
            self._validate_xai_tools(model_config["xai_tools"])
            if not self.config.get("use_encrypted_content") and not model_config.get("use_encrypted_content"):
                model_config["use_encrypted_content"] = True
                logger.debug("auto-enabled use_encrypted_content for server-side tool state preservation")
        self.config.update(model_config)

    @override
    def get_config(self) -> xAIConfig:
        """Get the Grok model configuration."""
        return self.config

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the Grok model."""
        client = self._get_client()

        try:
            chat = self._build_chat(client, tool_specs)
            self._append_messages_to_chat(chat, messages, system_prompt)

            yield self._format_chunk({"chunk_type": "message_start"})

            tool_calls_pending: list[dict[str, Any]] = []
            server_tool_calls: list[dict[str, Any]] = []
            current_content_type: Optional[str] = None
            final_response: Any = None
            citations: Any = None

            async for response, chunk in chat.stream():
                final_response = response

                if hasattr(response, "citations") and response.citations:
                    citations = response.citations

                if hasattr(chunk, "reasoning_content") and chunk.reasoning_content:
                    if current_content_type != "reasoning":
                        if current_content_type:
                            yield self._format_chunk({"chunk_type": "content_stop"})
                        yield self._format_chunk({"chunk_type": "content_start", "data_type": "reasoning"})
                        current_content_type = "reasoning"
                    yield self._format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": "reasoning_content",
                            "data": chunk.reasoning_content,
                        }
                    )

                if hasattr(chunk, "content") and chunk.content:
                    if current_content_type != "text":
                        if current_content_type:
                            yield self._format_chunk({"chunk_type": "content_stop"})
                        yield self._format_chunk({"chunk_type": "content_start", "data_type": "text"})
                        current_content_type = "text"
                    yield self._format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": "text",
                            "data": chunk.content,
                        }
                    )

                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        tool_type = get_tool_call_type(tool_call)
                        tool_data = {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments or "",
                        }
                        if tool_type == "client_side_tool":
                            tool_calls_pending.append(tool_data)
                        else:
                            logger.debug(
                                "tool_type=<%s>, tool_name=<%s> | server-side tool executed by xAI",
                                tool_type,
                                tool_call.function.name,
                            )
                            server_tool_calls.append(tool_data)

                            if current_content_type != "text":
                                if current_content_type:
                                    yield self._format_chunk({"chunk_type": "content_stop"})
                                yield self._format_chunk({"chunk_type": "content_start", "data_type": "text"})
                                current_content_type = "text"
                            yield self._format_chunk(
                                {
                                    "chunk_type": "content_delta",
                                    "data_type": "server_tool",
                                    "data": tool_data,
                                }
                            )

            if current_content_type:
                yield self._format_chunk({"chunk_type": "content_stop"})

            # Emit encrypted reasoning content for visibility (grok-4 with use_encrypted_content=True)
            # The actual state preservation happens via xAI state capture below
            if final_response and hasattr(final_response, "encrypted_content") and final_response.encrypted_content:
                encrypted_bytes = (
                    final_response.encrypted_content.encode("utf-8")
                    if isinstance(final_response.encrypted_content, str)
                    else final_response.encrypted_content
                )
                yield self._format_chunk({"chunk_type": "content_start", "data_type": "encrypted_reasoning"})
                yield {"contentBlockDelta": {"delta": {"reasoningContent": {"redactedContent": encrypted_bytes}}}}
                yield self._format_chunk({"chunk_type": "content_stop"})

            # Emit tool call events (client-side only)
            for tool_call in tool_calls_pending:
                yield self._format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_call})
                yield self._format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_call})
                yield self._format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

            # Determine if we need to capture xAI state for multi-turn context preservation
            # State is needed when:
            # 1. Server-side tools were used (encrypted tool results must be preserved)
            # 2. Encrypted reasoning content is present (grok-4 with use_encrypted_content=True)
            # State is NOT needed for client-side tools only (Strands handles those)
            has_encrypted_reasoning = (
                final_response and hasattr(final_response, "encrypted_content") and final_response.encrypted_content
            )
            needs_xai_state = server_tool_calls or has_encrypted_reasoning

            # =================================================================
            # STATE PRESERVATION FOR MULTI-TURN CONVERSATIONS
            # =================================================================
            # When server-side tools or encrypted reasoning are used, we must
            # preserve the xAI SDK's internal state for the next turn. This state
            # contains encrypted content that cannot be reconstructed from text.
            #
            # We store this state in `reasoningContent.redactedContent` because:
            # 1. This field is NOT rendered in AgentResult.__str__() - keeps output clean
            # 2. The Strands event loop properly accumulates redactedContent
            # 3. It's semantically appropriate (encrypted/hidden content)
            #
            # On the next turn, _extract_xai_state() will find this content and
            # restore the full conversation context including encrypted tool results.
            # =================================================================
            if final_response and needs_xai_state:
                xai_state = self._capture_xai_state(chat, final_response)
                if xai_state:
                    # Encode: protobuf bytes -> base64 -> JSON -> base64 -> markers
                    state_json = json.dumps({"messages": xai_state})
                    state_b64 = base64.b64encode(state_json.encode("utf-8")).decode("utf-8")
                    state_text = f"{XAI_STATE_MARKER}{state_b64}{XAI_STATE_MARKER_END}"

                    # Emit as reasoningContent.redactedContent - this is the key!
                    # Unlike text content, redactedContent is NOT shown to users
                    yield self._format_chunk({"chunk_type": "content_start", "data_type": "xai_state"})
                    yield {
                        "contentBlockDelta": {
                            "delta": {"reasoningContent": {"redactedContent": state_text.encode("utf-8")}}
                        }
                    }
                    yield self._format_chunk({"chunk_type": "content_stop"})

            stop_reason = "tool_use" if tool_calls_pending else "end_turn"
            yield self._format_chunk({"chunk_type": "message_stop", "data": stop_reason})

            if final_response and hasattr(final_response, "usage") and final_response.usage:
                yield self._format_chunk(
                    {
                        "chunk_type": "metadata",
                        "data": final_response.usage,
                        "citations": citations,
                        "server_tool_calls": server_tool_calls if server_tool_calls else None,
                    }
                )

        except Exception as error:
            self._handle_stream_error(error)

        logger.debug("finished streaming response from xAI")

    def _handle_stream_error(self, error: Exception) -> None:
        """Handle errors from the xAI API and map them to Strands exceptions."""
        from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException

        error_message = str(error).lower()
        error_str = str(error)

        if any(x in error_message for x in ["rate limit", "rate_limit", "too many requests"]) or "429" in error_str:
            logger.warning("error=<%s> | xAI rate limit error", error)
            raise ModelThrottledException(str(error)) from error

        if any(x in error_message for x in ["context length", "maximum context", "token limit"]):
            logger.warning("error=<%s> | xAI context window overflow error", error)
            raise ContextWindowOverflowException(str(error)) from error

        raise error

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the Grok model using chat.parse()."""
        from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException

        client = self._get_client()

        try:
            chat = self._build_chat(client)
            self._append_messages_to_chat(chat, prompt, system_prompt)

            response, parsed_output = await chat.parse(output_model)
            yield {"output": parsed_output}

        except Exception as error:
            error_message = str(error).lower()
            error_str = str(error)

            if any(x in error_message for x in ["rate limit", "rate_limit", "too many requests"]) or "429" in error_str:
                logger.warning("error=<%s> | xAI rate limit error", error)
                raise ModelThrottledException(str(error)) from error

            if any(x in error_message for x in ["context length", "maximum context", "token limit"]):
                logger.warning("error=<%s> | xAI context window overflow error", error)
                raise ContextWindowOverflowException(str(error)) from error

            raise
