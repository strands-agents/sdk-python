"""Lightweight token estimation utilities for conversation managers."""

import json
from collections.abc import Callable
from typing import Any

from ...types.content import Messages

_IMAGE_CHAR_ESTIMATE = 4000

TokenCounter = Callable[[Messages], int]


def estimate_tokens(messages: Messages) -> int:
    """Approximate token count for a message list using a chars/4 heuristic.

    This is deliberately conservative (overestimates for English text, underestimates for CJK).
    For model-specific accuracy, pass a custom ``token_counter`` to the conversation manager.

    Args:
        messages: The conversation message history.

    Returns:
        Estimated token count.
    """
    total_chars = 0
    for msg in messages:
        for block in msg.get("content", []):
            total_chars += _estimate_block_chars(block)
    return total_chars // 4


def _estimate_block_chars(block: Any) -> int:
    """Estimate character count for a single content block."""
    if "text" in block:
        return len(block["text"])

    if "toolResult" in block:
        result = block["toolResult"]
        chars = 0
        for item in result.get("content", []):
            if "text" in item:
                chars += len(item["text"])
            elif "image" in item:
                chars += _IMAGE_CHAR_ESTIMATE
        return chars

    if "toolUse" in block:
        tool_use = block["toolUse"]
        chars = len(tool_use.get("name", ""))
        tool_input = tool_use.get("input", {})
        if isinstance(tool_input, str):
            chars += len(tool_input)
        else:
            chars += len(json.dumps(tool_input, default=str))
        return chars

    if "image" in block:
        return _IMAGE_CHAR_ESTIMATE

    # NOTE (M3): len(bytes) returns raw binary size, not extractable text length.
    # A 100KB PDF may contain only 5KB of text — this overestimates for binary
    # documents but stays in the same accuracy class as the chars/4 heuristic.
    if "document" in block:
        doc = block["document"]
        source = doc.get("source", {})
        data = source.get("bytes", b"")
        return len(data) if data else 200

    # NOTE (L1): Rough placeholder — actual video token cost varies enormously by
    # duration/resolution. Treat as an order-of-magnitude estimate.
    if "video" in block:
        return _IMAGE_CHAR_ESTIMATE * 10

    if "reasoningContent" in block:
        rc = block["reasoningContent"]
        if isinstance(rc, dict) and "reasoningText" in rc:
            return len(rc["reasoningText"].get("text", ""))
        return len(str(rc))

    if "cachePoint" in block or "guardContent" in block or "citationsContent" in block:
        return 0

    return len(str(block))
