"""Internal helpers shared by OpenAI-compatible model providers.

OpenAI-compatible providers (``openai``, ``openai_responses``, ``llamacpp``) stream vendor
events through the same internal ``chunk_type`` protocol. This module holds the shared
translation from those events into Strands ``StreamEvent`` chunks. Provider-specific events
(metadata usage shapes, citations) stay in each provider, which handles them before
delegating the common cases here.
"""

from typing import Any

from ..types.streaming import StreamEvent


def format_common_chunk(event: dict[str, Any]) -> StreamEvent:
    """Format an OpenAI-compatible response event into a standardized message chunk.

    Args:
        event: A response event from an OpenAI-compatible model.

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
                return {"contentBlockDelta": {"delta": {"toolUse": {"input": event["data"].function.arguments or ""}}}}

            if event["data_type"] == "reasoning_content":
                return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": event["data"]}}}}

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

        case _:
            raise RuntimeError(f"chunk_type=<{event['chunk_type']}> | unknown type")
